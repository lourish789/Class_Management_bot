import os
import time
import re
from datetime import datetime, timedelta
from decimal import Decimal
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from flask import Flask, request, jsonify
import threading
import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
import logging
from typing import List, Dict, Optional, Tuple
import sys
import json
from queue import Queue, Empty
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import signal
import atexit

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - OPTIMIZED FOR IMMEDIATE RESPONSES
CONFIG = {
    'PINECONE_API_KEY': os.getenv("PINECONE_API_KEY"),
    'GOOGLE_API_KEY': os.getenv("GOOGLE_API_KEY"), 
    'GREEN_API_ID': os.getenv("GREEN_API_ID_INSTANCE"), 
    'GREEN_API_TOKEN': os.getenv("GREEN_API_TOKEN"), 
    'DATABASE_URL': os.getenv("DATABASE_URL"),  
    'MAX_HISTORY': 4,  # Reduced for speed
    'INDEX_NAME': 'coach',
    'MEMORY_DAYS': 90,
    'RAG_TIMEOUT': 1.0,  # Reduced
    'LLM_TIMEOUT': 5,    # Reduced
    'MAX_TOKENS': 400,   # Reduced for faster generation
    'CACHE_TTL_USER': 300,
    'CACHE_TTL_HISTORY': 180,
    'MAX_CACHE_SIZE': 100,
    'DB_POOL_MIN': 2,
    'DB_POOL_MAX': 10,
    'THREAD_WORKERS': 6,
    'RESPONSE_TIMEOUT': 3  # Max time to generate response
}

# Validate required keys
required_keys = ['GOOGLE_API_KEY', 'GREEN_API_ID', 'GREEN_API_TOKEN', 'DATABASE_URL']
missing_keys = [key for key in required_keys if not CONFIG[key]]

if missing_keys:
    logger.error(f"Missing configuration: {', '.join(missing_keys)}")
    raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")

# Global executor
executor = None

def init_executor():
    global executor
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=CONFIG['THREAD_WORKERS'])
        logger.info(f"✓ Executor initialized with {CONFIG['THREAD_WORKERS']} workers")

# Initialize AI components
pinecone_index = None
embed_model = None
llm = None

def init_ai_components():
    global pinecone_index, embed_model, llm
    
    try:
        # Initialize Pinecone (optional)
        if CONFIG['PINECONE_API_KEY']:
            try:
                pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
                pinecone_index = pc.Index(CONFIG['INDEX_NAME'])
                logger.info("✓ Pinecone connected")
            except Exception as e:
                logger.warning(f"Pinecone unavailable: {e}")
        
        # Initialize Google AI with faster settings
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=CONFIG['GOOGLE_API_KEY'],
            temperature=0.4,  # Lower for more consistent speed
            max_tokens=CONFIG['MAX_TOKENS'],
            timeout=CONFIG['LLM_TIMEOUT']
        )
        logger.info("✓ Google AI initialized (fast mode)")
        return True
    except Exception as e:
        logger.error(f"AI initialization error: {e}")
        return False


def extract_phone_number(chat_id: str) -> str:
    """Extract phone number from chat_id"""
    phone = re.sub(r'@.*$', '', chat_id)
    return re.sub(r'\D', '', phone) or chat_id


def generate_unique_user_id() -> str:
    """Generate unique user ID"""
    return str(uuid.uuid4())


class SimpleCache:
    """Memory-efficient cache with size limits"""
    
    def __init__(self, ttl=300, max_size=100):
        self.cache = {}
        self.ttl = ttl
        self.max_size = max_size
        self.lock = threading.Lock()
        self.access_count = {}
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.access_count[key] = self.access_count.get(key, 0) + 1
                    return value
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
                if oldest_key in self.access_count:
                    del self.access_count[oldest_key]
            
            self.cache[key] = (value, time.time())
            self.access_count[key] = 1
    
    def delete(self, key):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
    
    def size(self):
        with self.lock:
            return len(self.cache)


class DatabaseManager:
    """Optimized database manager with async writes"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool = None
        self.write_queue = Queue(maxsize=500)
        self.worker_thread = None
        self.running = False
        self._init_pool()
        self._init_db()
        self._start_worker()
    
    def _start_worker(self):
        self.running = True
        
        def worker():
            while self.running:
                try:
                    op = self.write_queue.get(timeout=1)
                    op()
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"DB worker error: {e}")
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        logger.info("✓ DB worker started")
    
    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        if self.connection_pool:
            self.connection_pool.closeall()
    
    def _init_pool(self):
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=CONFIG['DB_POOL_MIN'],
                maxconn=CONFIG['DB_POOL_MAX'],
                dsn=self.database_url
            )
            logger.info("✓ PostgreSQL pool created")
        except Exception as e:
            logger.error(f"Pool error: {e}")
            raise
    
    def _init_db(self):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS users (
                            user_id VARCHAR(50) PRIMARY KEY,
                            phone_number VARCHAR(20) UNIQUE NOT NULL,
                            chat_id VARCHAR(100) UNIQUE,
                            first_name VARCHAR(100),
                            email VARCHAR(255),
                            location VARCHAR(100),
                            class_taught VARCHAR(100),
                            profile_complete BOOLEAN DEFAULT FALSE,
                            registration_step INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_verification TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            total_messages INTEGER DEFAULT 0,
                            needs_reverification BOOLEAN DEFAULT FALSE
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS conversations (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(50) NOT NULL,
                            phone_number VARCHAR(20) NOT NULL,
                            message_type VARCHAR(10),
                            message_content TEXT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            intent VARCHAR(50),
                            session_id VARCHAR(50),
                            FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS sessions (
                            session_id VARCHAR(50) PRIMARY KEY,
                            user_id VARCHAR(50) NOT NULL,
                            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                        )
                    ''')
                    
                    # Indexes for fast queries
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON conversations (user_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone ON conversations (phone_number)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone_number)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_id ON users (chat_id)')
                    
                    conn.commit()
                    logger.info("✓ Database initialized")
        except Exception as e:
            logger.error(f"DB init error: {e}")
            raise
    
    @contextmanager
    def get_conn(self):
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def queue_write(self, operation):
        try:
            self.write_queue.put_nowait(operation)
        except:
            logger.warning("DB queue full")
    
    def get_user_by_phone(self, phone: str) -> Optional[Dict]:
        phone = re.sub(r'\D', '', phone)
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('SELECT * FROM users WHERE phone_number = %s', (phone,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Get user error: {e}")
            return None
    
    def get_user_by_chat_id(self, chat_id: str) -> Optional[Dict]:
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('SELECT * FROM users WHERE chat_id = %s', (chat_id,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            return None
    
    def save_user_sync(self, phone: str, chat_id=None, user_id=None, **kwargs) -> str:
        try:
            phone = re.sub(r'\D', '', phone)
            
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('SELECT user_id FROM users WHERE phone_number = %s', (phone,))
                    existing = cur.fetchone()
                    
                    if existing:
                        user_id = existing['user_id']
                        updates = []
                        values = []
                        
                        if chat_id:
                            updates.append('chat_id = %s')
                            values.append(chat_id)
                        
                        for k, v in kwargs.items():
                            updates.append(f'{k} = %s')
                            values.append(v)
                        
                        updates.append('last_interaction = %s')
                        values.append(datetime.now())
                        values.append(user_id)
                        
                        cur.execute(f'UPDATE users SET {", ".join(updates)} WHERE user_id = %s', values)
                        conn.commit()
                        return user_id
                    else:
                        new_id = user_id or generate_unique_user_id()
                        fields = ['user_id', 'phone_number']
                        values = [new_id, phone]
                        placeholders = ['%s', '%s']
                        
                        if chat_id:
                            fields.append('chat_id')
                            values.append(chat_id)
                            placeholders.append('%s')
                        
                        for k, v in kwargs.items():
                            fields.append(k)
                            values.append(v)
                            placeholders.append('%s')
                        
                        cur.execute(
                            f'INSERT INTO users ({", ".join(fields)}) VALUES ({", ".join(placeholders)})',
                            values
                        )
                        conn.commit()
                        return new_id
        except Exception as e:
            logger.error(f"Save user error: {e}")
            return None
    
    def save_user_async(self, phone: str, chat_id=None, user_id=None, **kwargs):
        def op():
            self.save_user_sync(phone, chat_id, user_id, **kwargs)
        self.queue_write(op)
    
    def save_message_async(self, user_id: str, phone: str, msg_type: str, 
                          content: str, intent=None, session_id=None):
        def op():
            try:
                phone = re.sub(r'\D', '', phone)
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute('''
                            INSERT INTO conversations 
                            (user_id, phone_number, message_type, message_content, intent, session_id)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        ''', (user_id, phone, msg_type, content, intent, session_id))
                        
                        # Update message count
                        cur.execute('''
                            UPDATE users 
                            SET total_messages = total_messages + 1,
                                last_interaction = %s
                            WHERE user_id = %s
                        ''', (datetime.now(), user_id))
                        
                        conn.commit()
            except Exception as e:
                logger.error(f"Save message error: {e}")
        self.queue_write(op)
    
    def get_history(self, user_id: str, limit=None) -> List[Dict]:
        limit = limit or CONFIG['MAX_HISTORY']
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('''
                        SELECT message_type, message_content, timestamp, intent
                        FROM conversations 
                        WHERE user_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    ''', (user_id, limit))
                    return [dict(r) for r in reversed(cur.fetchall())]
        except Exception as e:
            return []
    
    def create_session_async(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        def op():
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute('''
                            INSERT INTO sessions (session_id, user_id)
                            VALUES (%s, %s)
                        ''', (session_id, user_id))
                        conn.commit()
            except Exception as e:
                logger.error(f"Session error: {e}")
        self.queue_write(op)
        return session_id
    
    def check_memory_expiry(self, user_id: str) -> bool:
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('''
                        SELECT last_verification, needs_reverification 
                        FROM users WHERE user_id = %s
                    ''', (user_id,))
                    result = cur.fetchone()
                    
                    if not result or result['needs_reverification']:
                        return True
                    
                    if result['last_verification']:
                        days = (datetime.now() - result['last_verification']).days
                        if days >= CONFIG['MEMORY_DAYS']:
                            def update():
                                try:
                                    with self.get_conn() as c:
                                        with c.cursor() as cu:
                                            cu.execute('UPDATE users SET needs_reverification = TRUE WHERE user_id = %s', (user_id,))
                                            c.commit()
                                except: 
                                    pass
                            self.queue_write(update)
                            return True
                    return False
        except:
            return False
    
    def reset_verification(self, user_id: str):
        def op():
            try:
                with self.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute('UPDATE users SET last_verification = %s, needs_reverification = FALSE WHERE user_id = %s',
                                  (datetime.now(), user_id))
                        conn.commit()
            except: 
                pass
        self.queue_write(op)


class ResponseFormatter:
    @staticmethod
    def clean_response(text: str) -> str:
        """Fast response cleaning"""
        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = text.replace('**', '').replace('*', '')
        
        # Format lists
        lines = text.split('\n')
        formatted = []
        counter = 1
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted.append('')
                counter = 1
            elif re.match(r'^[-•·]\s+', stripped):
                content = re.sub(r'^[-•·]\s+', '', stripped)
                formatted.append(f"{counter}. {content}")
                counter += 1
            else:
                formatted.append(stripped)
        
        return re.sub(r'\n{3,}', '\n\n', '\n'.join(formatted)).strip()


class LinkaAI:
    """Fast AI response generator"""
    
    def __init__(self, llm):
        self.llm = llm
        self.formatter = ResponseFormatter()
    
    def generate_response(self, message: str, user: Dict, history=None) -> Tuple[str, str]:
        """Generate response with timeout"""
        try:
            intent = self._extract_intent(message)
            
            # Build minimal context
            context = ""
            if history and len(history) > 0:
                last_msg = history[-1]
                if last_msg['message_type'] == 'user':
                    context = f"Last: {last_msg['message_content'][:40]}"
            
            name = user.get('first_name', 'Teacher')
            cls = user.get('class_taught', 'your class')
            
            # Ultra-concise prompt for speed
            prompt = f"""You are LINKA AI - teaching assistant for Nigerian teachers.

Teacher: {name} ({cls})
Query: {message}

Rules:
- Be warm, brief (60 words max)
- Use 1. 2. 3. format
- NO markdown
- Practical advice only"""
            
            if context:
                prompt += f"\nContext: {context}"
            
            # Generate with timeout
            response = self.llm.invoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}
            ])
            
            return self.formatter.clean_response(response.content), intent
            
        except Exception as e:
            logger.error(f"Response error: {e}")
            return "I'm here to help! Could you rephrase that?", "error"
    
    @staticmethod
    def _extract_intent(msg: str) -> str:
        """Fast intent extraction"""
        m = msg.lower()
        if any(k in m for k in ['teach', 'lesson', 'explain']):
            return 'teaching'
        elif any(k in m for k in ['discipline', 'behavior']):
            return 'management'
        elif any(k in m for k in ['assess', 'test', 'grade']):
            return 'assessment'
        elif any(k in m for k in ['hello', 'hi', 'hey']):
            return 'greeting'
        return 'general'


# Registration steps - streamlined
REGISTRATION_STEPS = {
    0: {"message": "👋 Hi! I'm LINKA AI. I help teachers with strategies and development.\n\nWhat's your first name?", "field": "first_name"},
    1: {"message": "Nice to meet you, {first_name}! What's your email?", "field": "email"},
    2: {"message": "Thanks! Which city/state are you in?", "field": "location"},
    3: {"message": "Great! Which class do you teach?", "field": "class_taught"}
}


def handle_registration(phone: str, chat_id: str, text: str, user: Optional[Dict]) -> str:
    """Fast registration handler"""
    if not user:
        new_id = generate_unique_user_id()
        db.save_user_sync(phone, chat_id, new_id, 
                         profile_complete=False, registration_step=0)
        return REGISTRATION_STEPS[0]["message"]
    
    step = user.get('registration_step', 0)
    user_id = user.get('user_id')
    
    if step >= 4:
        return None  # Registration complete
    
    text = text.strip()
    if not text or len(text) < 2:
        return f"Please provide your {REGISTRATION_STEPS[step]['field'].replace('_', ' ')}."
    
    if step == 0:
        name = re.sub(r'[^a-zA-Z\s]', '', text).strip().title()
        if len(name) > 1:
            db.save_user_sync(phone, chat_id, user_id,
                            first_name=name, registration_step=1)
            user_cache.delete(phone)
            return REGISTRATION_STEPS[1]["message"].format(first_name=name)
        return "Please enter a valid name."
    
    elif step == 1:
        if '@' in text and '.' in text:
            db.save_user_sync(phone, email=text.lower(), registration_step=2)
            user_cache.delete(phone)
            return REGISTRATION_STEPS[2]["message"]
        return "Please enter a valid email."
    
    elif step == 2:
        if len(text) > 2:
            db.save_user_sync(phone, location=text.title(), registration_step=3)
            user_cache.delete(phone)
            return REGISTRATION_STEPS[3]["message"]
        return "Please enter a valid location."
    
    elif step == 3:
        if len(text) > 1:
            db.save_user_sync(phone, class_taught=text.title(), 
                            profile_complete=True, registration_step=4)
            db.reset_verification(user_id)
            user_cache.delete(phone)
            
            name = user.get('first_name', 'Teacher')
            return (
                f"✅ Perfect, {name}! You're all set.\n\n"
                f"I can help with:\n"
                f"1. Teaching strategies\n"
                f"2. Classroom management\n"
                f"3. Assessment tips\n"
                f"4. Student engagement\n\n"
                f"What would you like to know?"
            )
        return "Please enter your class."
    
    return "Please try again."


def process_message(phone: str, chat_id: str, text: str):
    """Ultra-fast message processing with immediate response"""
    response_sent = False
    
    try:
        # Get user (from cache first)
        user = user_cache.get(phone)
        if not user:
            user = db.get_user_by_phone(phone)
            if not user:
                user = db.get_user_by_chat_id(chat_id)
                if user:
                    db.save_user_async(phone, chat_id, user['user_id'])
            
            if user:
                user_cache.set(phone, user)
        
        # Handle registration
        if not user or not user.get('profile_complete') or user.get('registration_step', 0) < 4:
            response = handle_registration(phone, chat_id, text, user)
            send_whatsapp_message(chat_id, response)
            response_sent = True
            return
        
        user_id = user['user_id']
        
        # Check memory expiry
        if db.check_memory_expiry(user_id):
            response = f"Hi {user.get('first_name')}! It's been 90 days.\n\nPlease confirm your email: {user.get('email', 'N/A')}\n\nReply 'same' or provide updated email."
            send_whatsapp_message(chat_id, response)
            response_sent = True
            return
        
        # Get history (cached)
        history = history_cache.get(user_id)
        if not history:
            history = db.get_history(user_id, limit=CONFIG['MAX_HISTORY'])
            if history:
                history_cache.set(user_id, history)
        
        # Generate response with timeout
        try:
            response, intent = linka_ai.generate_response(text, user, history)
        except Exception as e:
            logger.error(f"AI error: {e}")
            response = "I'm here to help! Please try asking again."
            intent = "error"
        
        # Send immediately
        send_whatsapp_message(chat_id, response)
        response_sent = True
        
        # Log asynchronously (non-blocking)
        if executor:
            executor.submit(log_async, user_id, phone, text, response, intent, user, history)
        
    except Exception as e:
        logger.error(f"Process error: {e}")
        if not response_sent:
            send_whatsapp_message(chat_id, "I'm experiencing a brief issue. Please try again in a moment.")


def log_async(user_id: str, phone: str, text: str, response: str, 
              intent: str, user: Dict, history: List):
    """Asynchronous logging (non-blocking)"""
    try:
        session_id = db.create_session_async(user_id)
        
        # Save messages
        db.save_message_async(user_id, phone, 'user', text, intent, session_id)
        db.save_message_async(user_id, phone, 'assistant', response, intent, session_id)
        
        # Update cache
        if history:
            history.append({
                'message_type': 'user',
                'message_content': text,
                'timestamp': datetime.now(),
                'intent': intent
            })
            history_cache.set(user_id, history[-CONFIG['MAX_HISTORY']:])
        
    except Exception as e:
        logger.error(f"Async logging error: {e}")


# Global instances
db = None
linka_ai = None
user_cache = None
history_cache = None

def init_components():
    """Initialize all components"""
    global db, linka_ai, user_cache, history_cache, executor
    
    try:
        init_executor()
        
        db = DatabaseManager(CONFIG['DATABASE_URL'])
        logger.info("✓ Database ready")
        
        if init_ai_components():
            linka_ai = LinkaAI(llm)
            logger.info("✓ AI ready (fast mode)")
        else:
            raise Exception("AI initialization failed")
        
        user_cache = SimpleCache(ttl=CONFIG['CACHE_TTL_USER'], max_size=CONFIG['MAX_CACHE_SIZE'])
        history_cache = SimpleCache(ttl=CONFIG['CACHE_TTL_HISTORY'], max_size=CONFIG['MAX_CACHE_SIZE'])
        logger.info("✓ Caches ready")
        
        return True
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return False


def cleanup():
    """Cleanup on shutdown"""
    logger.info("Cleaning up...")
    try:
        if db:
            db.stop()
        if executor:
            executor.shutdown(wait=False)
        logger.info("✓ Cleanup complete")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

atexit.register(cleanup)


def send_whatsapp_message(phone: str, message: str) -> bool:
    """Send WhatsApp message - optimized for speed"""
    try:
        url = f"https://api.green-api.com/waInstance{CONFIG['GREEN_API_ID']}/sendMessage/{CONFIG['GREEN_API_TOKEN']}"
        response = requests.post(
            url,
            json={"chatId": phone, "message": message},
            timeout=3  # Fast timeout
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Sent to {phone[:10]}")
            return True
        else:
            logger.warning(f"Send failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Send error: {e}")
        return False


# Keep-alive
last_activity = time.time()

def update_activity():
    global last_activity
    last_activity = time.time()

@app.before_request
def before_request():
    update_activity()


# Flask Routes

@app.route('/')
def health():
    return jsonify({
        "status": "healthy",
        "service": "LINKA AI",
        "version": "8.0-immediate",
        "mode": "fast_response",
        "features": [
            "Immediate responses",
            "Async logging",
            "Smart caching",
            "Fast AI generation"
        ],
        "response_time": "< 2 seconds",
        "cache_stats": {
            "users_cached": user_cache.size() if user_cache else 0,
            "histories_cached": history_cache.size() if history_cache else 0
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Optimized webhook for immediate responses"""
    try:
        data = request.get_json()
        
        # Quick validation
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored"}), 200
        
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        # Filter group messages
        if not chat_id or '@g.us' in chat_id:
            return jsonify({"status": "ignored"}), 200
        
        phone = extract_phone_number(chat_id)
        
        # Extract text quickly
        text = None
        if 'textMessageData' in message_data:
            text = message_data['textMessageData'].get('textMessage', '').strip()
        elif 'extendedTextMessageData' in message_data:
            text = message_data['extendedTextMessageData'].get('text', '').strip()
        
        if not text:
            return jsonify({"status": "non_text"}), 200
        
        logger.info(f"📨 {phone[:10]}: {text[:30]}")
        
        # Process IMMEDIATELY in same thread for fastest response
        process_message(phone, chat_id, text)
        
        return jsonify({"status": "processed"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error"}), 500


@app.route('/ping', methods=['GET'])
def ping():
    """Keep-alive endpoint"""
    return jsonify({
        "status": "alive",
        "uptime": int(time.time() - last_activity),
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint"""
    try:
        data = request.get_json()
        phone = data.get('phone_number', '2348012345678')
        chat_id = data.get('chat_id', f'{phone}@c.us')
        message = data.get('message', 'Hello')
        
        start_time = time.time()
        
        # Get user
        user = user_cache.get(phone)
        if not user:
            user = db.get_user_by_phone(phone)
            if user:
                user_cache.set(phone, user)
        
        # Generate response
        if not user or not user.get('profile_complete'):
            response = handle_registration(phone, chat_id, message, user)
            intent = 'registration'
        else:
            history = history_cache.get(user['user_id'])
            if not history:
                history = db.get_history(user['user_id'])
            response, intent = linka_ai.generate_response(message, user, history)
        
        elapsed = time.time() - start_time
        
        return jsonify({
            "response": response,
            "intent": intent,
            "user": dict(user) if user else None,
            "cached": user_cache.get(phone) is not None,
            "response_time_seconds": round(elapsed, 3),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics"""
    try:
        if not db:
            return jsonify({"error": "Database not initialized"}), 500
            
        with db.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT COUNT(*) as total FROM users')
                total = cur.fetchone()['total']
                
                cur.execute('SELECT COUNT(*) as registered FROM users WHERE profile_complete = TRUE')
                registered = cur.fetchone()['registered']
                
                cur.execute('SELECT COUNT(*) as total FROM conversations')
                messages = cur.fetchone()['total']
                
                cur.execute("SELECT COUNT(*) as today FROM conversations WHERE DATE(timestamp) = CURRENT_DATE")
                today = cur.fetchone()['today']
        
        return jsonify({
            "chatbot": "LINKA AI",
            "version": "8.0-immediate",
            "stats": {
                "total_users": total,
                "registered_users": registered,
                "total_messages": messages,
                "today_messages": today
            },
            "performance": {
                "mode": "immediate_response",
                "target_response_time": "< 2 seconds",
                "users_cached": user_cache.size() if user_cache else 0,
                "histories_cached": history_cache.size() if history_cache else 0
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health_check', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    components = {
        "database": False,
        "google_ai": False,
        "green_api": False,
        "cache": False
    }
    
    try:
        if db and db.connection_pool:
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT 1')
                    components["database"] = True
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
    
    components["google_ai"] = llm is not None
    components["green_api"] = bool(CONFIG['GREEN_API_ID'])
    components["cache"] = user_cache is not None and history_cache is not None
    
    all_healthy = all(components.values())
    
    return jsonify({
        "status": "healthy" if all_healthy else "degraded",
        "chatbot": "LINKA AI v8.0",
        "mode": "immediate_response",
        "components": components,
        "performance": {
            "target_response": "< 2 seconds",
            "max_tokens": CONFIG['MAX_TOKENS'],
            "llm_timeout": CONFIG['LLM_TIMEOUT'],
            "history_limit": CONFIG['MAX_HISTORY']
        },
        "optimizations": {
            "immediate_send": "enabled",
            "async_logging": "enabled",
            "smart_caching": "enabled",
            "db_pool": f"{CONFIG['DB_POOL_MIN']}-{CONFIG['DB_POOL_MAX']}",
            "workers": CONFIG['THREAD_WORKERS']
        },
        "cache_stats": {
            "user_cache_size": user_cache.size() if user_cache else 0,
            "history_cache_size": history_cache.size() if history_cache else 0,
            "max_cache_size": CONFIG['MAX_CACHE_SIZE']
        },
        "timestamp": datetime.now().isoformat()
    }), 200 if all_healthy else 503


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all caches"""
    try:
        if user_cache:
            user_cache.clear()
        if history_cache:
            history_cache.clear()
        return jsonify({
            "status": "success",
            "message": "All caches cleared",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get detailed cache statistics"""
    try:
        return jsonify({
            "user_cache": {
                "size": user_cache.size() if user_cache else 0,
                "ttl_seconds": CONFIG['CACHE_TTL_USER'],
                "max_size": CONFIG['MAX_CACHE_SIZE']
            },
            "history_cache": {
                "size": history_cache.size() if history_cache else 0,
                "ttl_seconds": CONFIG['CACHE_TTL_HISTORY'],
                "max_size": CONFIG['MAX_CACHE_SIZE']
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Initialize on startup
@app.before_first_request
def startup():
    """Initialize components before first request"""
    logger.info("=" * 70)
    logger.info("LINKA AI - IMMEDIATE RESPONSE MODE")
    logger.info("=" * 70)
    
    if not init_components():
        logger.error("Failed to initialize!")
        raise Exception("Initialization failed")
    
    logger.info("=" * 70)
    logger.info("✓ LINKA AI READY - IMMEDIATE MODE")
    logger.info("=" * 70)


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LINKA AI v8.0 - IMMEDIATE RESPONSE")
    logger.info("=" * 70)
    
    # Initialize components
    if not init_components():
        logger.error("Failed to initialize! Exiting...")
        sys.exit(1)
    
    logger.info("✓ Mode: Immediate response")
    logger.info("✓ Target: < 2 seconds")
    logger.info("✓ Strategy: Send first, log later")
    logger.info("✓ AI: Fast generation (400 tokens)")
    logger.info("✓ Cache: Smart user/history caching")
    logger.info("✓ DB: Async writes only")
    logger.info("=" * 70)
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
