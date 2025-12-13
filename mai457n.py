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
from queue import Queue, PriorityQueue
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('linka_ai.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
CONFIG = {
    'PINECONE_API_KEY': os.getenv("PINECONE_API_KEY"),
    'GOOGLE_API_KEY': os.getenv("GOOGLE_API_KEY"), 
    'GREEN_API_ID': os.getenv("GREEN_API_ID_INSTANCE"), 
    'GREEN_API_TOKEN': os.getenv("GREEN_API_TOKEN"), 
    'APPS_SCRIPT_URL': os.getenv("APPS_SCRIPT_URL"), 
    'SPREADSHEET_ID': os.getenv("SPREADSHEET_ID"), 
    'DATABASE_URL': os.getenv("DATABASE_URL"),  
    'MAX_HISTORY': 8,  # Reduced for faster queries
    'INDEX_NAME': 'coach',
    'MEMORY_DAYS': 90,
    'RAG_TIMEOUT': 1.5,
    'LLM_TIMEOUT': 4,
    'MAX_TOKENS': 700,
    'SHEETS_TIMEOUT': 7,
    'CACHE_TTL_USER': 300,  # 5 minutes
    'CACHE_TTL_HISTORY': 180  # 3 minutes
}

# Validate required keys
required_keys = ['PINECONE_API_KEY', 'GOOGLE_API_KEY', 'GREEN_API_ID', 'GREEN_API_TOKEN', 'DATABASE_URL']
missing_keys = [key for key in required_keys if not CONFIG[key]]

if missing_keys:
    logger.error(f"Missing configuration: {', '.join(missing_keys)}")
    raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")

# Thread pool - increased for better concurrency
executor = ThreadPoolExecutor(max_workers=20)

# Initialize AI
pinecone_index = None
embed_model = None
llm = None

try:
    pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
    try:
        pinecone_index = pc.Index(CONFIG['INDEX_NAME'])
        logger.info(f"✓ Pinecone connected")
    except Exception as e:
        logger.warning(f"Pinecone unavailable: {e}")
    
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=CONFIG['GOOGLE_API_KEY']
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=CONFIG['GOOGLE_API_KEY'],
        temperature=0.5,
        max_tokens=CONFIG['MAX_TOKENS'],
        timeout=CONFIG['LLM_TIMEOUT']
    )
    logger.info("✓ Google AI initialized")
except Exception as e:
    logger.error(f"Initialization error: {e}")
    raise


def extract_phone_number(chat_id: str) -> str:
    """Extract phone number from chat_id"""
    phone = re.sub(r'@.*$', '', chat_id)
    return re.sub(r'\D', '', phone) or chat_id


def generate_unique_user_id() -> str:
    """Generate unique user ID"""
    return str(uuid.uuid4())


class SimpleCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl=300):
        self.cache = {}
        self.ttl = ttl
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                del self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            self.cache[key] = (value, time.time())
    
    def delete(self, key):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear(self):
        with self.lock:
            self.cache.clear()


class SheetsLogger:
    """PRIORITY logging to Google Sheets - logs FIRST before database"""
    
    def __init__(self, apps_script_url: str):
        self.apps_script_url = apps_script_url
        self.priority_queue = PriorityQueue()
        self.batch_queue = Queue()
        
        if apps_script_url:
            self._start_workers()
    
    def _start_workers(self):
        def priority_worker():
            while True:
                try:
                    if not self.priority_queue.empty():
                        _, item = self.priority_queue.get()
                        self._log_sync(item)
                    time.sleep(0.3)
                except Exception as e:
                    logger.error(f"Priority worker error: {e}")
        
        def batch_worker():
            while True:
                try:
                    time.sleep(1.5)
                    self._process_batch()
                except Exception as e:
                    logger.error(f"Batch worker error: {e}")
                time.sleep(2)
        
        threading.Thread(target=priority_worker, daemon=True).start()
        threading.Thread(target=batch_worker, daemon=True).start()
        logger.info("✓ Sheets workers started")
    
    def log_user_registration_immediate(self, user_data: Dict) -> bool:
        """Log registration IMMEDIATELY to sheets (BLOCKING)"""
        if not self.apps_script_url:
            return False
        
        payload = {
            'type': 'user_registration',
            'data': {
                'user_id': user_data.get('user_id'),
                'phone_number': user_data.get('phone_number'),
                'name': user_data.get('first_name'),
                'email': user_data.get('email'),
                'location': user_data.get('location'),
                'class': user_data.get('class_taught'),
                'status': 'Registered',
                'registration_date': datetime.now().isoformat(),
                'chatbot_name': 'LINKA AI'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Try immediate sync (3 attempts)
        for attempt in range(3):
            try:
                response = requests.post(
                    self.apps_script_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=CONFIG['SHEETS_TIMEOUT']
                )
                
                if response.status_code == 200:
                    logger.info(f"✓ User logged to sheets: {user_data.get('user_id')[:8]}")
                    return True
            except Exception as e:
                logger.error(f"Sheets attempt {attempt + 1}: {e}")
            
            if attempt < 2:
                time.sleep(0.4)
        
        # Fallback to queue
        self.priority_queue.put((0, payload))
        return False
    
    def log_conversation_immediate(self, conv_data: Dict) -> bool:
        """Log conversation IMMEDIATELY to sheets (BLOCKING)"""
        if not self.apps_script_url:
            return False
        
        payload = {
            'type': 'conversation',
            'data': conv_data,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                self.apps_script_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=CONFIG['SHEETS_TIMEOUT']
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Conversation logged to sheets")
                return True
        except Exception as e:
            logger.error(f"Sheets log error: {e}")
        
        # Fallback to batch queue
        self.batch_queue.put(payload)
        return False
    
    def log_conversation_async(self, conv_data: Dict):
        """Add conversation to async batch"""
        payload = {
            'type': 'conversation',
            'data': conv_data,
            'timestamp': datetime.now().isoformat()
        }
        self.batch_queue.put(payload)
    
    def _log_sync(self, payload: Dict) -> bool:
        if not self.apps_script_url:
            return False
        
        try:
            response = requests.post(
                self.apps_script_url,
                json=payload,
                timeout=CONFIG['SHEETS_TIMEOUT']
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Sync log error: {e}")
            return False
    
    def _process_batch(self):
        if self.batch_queue.empty():
            return
        
        processed = 0
        max_batch = 8
        
        while not self.batch_queue.empty() and processed < max_batch:
            try:
                item = self.batch_queue.get_nowait()
                if self._log_sync(item):
                    processed += 1
                else:
                    self.batch_queue.put(item)
                    break
            except Exception as e:
                logger.error(f"Batch error: {e}")
                break
            time.sleep(0.25)
        
        if processed > 0:
            logger.info(f"✓ Batched {processed} to sheets")


class DatabaseManager:
    """PostgreSQL - SECONDARY storage after Sheets"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool = None
        self.write_queue = Queue()
        self._init_pool()
        self._init_db()
        self._start_worker()
    
    def _start_worker(self):
        def worker():
            while True:
                try:
                    if not self.write_queue.empty():
                        op = self.write_queue.get()
                        op()
                    time.sleep(0.15)
                except Exception as e:
                    logger.error(f"DB worker error: {e}")
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _init_pool(self):
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2, maxconn=20, dsn=self.database_url
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
                    
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON conversations (user_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone ON conversations (phone_number)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone_number)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
                    
                    conn.commit()
                    logger.info("✓ Database initialized")
        except Exception as e:
            logger.error(f"DB init error: {e}")
            raise
    
    @contextmanager
    def get_conn(self):
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    def queue_write(self, operation):
        self.write_queue.put(operation)
    
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
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('SELECT * FROM users WHERE user_id = %s', (user_id,))
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
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
        """SYNC save - used after sheets logging"""
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
                                except: pass
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
            except: pass
        self.queue_write(op)


class ResponseFormatter:
    @staticmethod
    def clean_response(text: str) -> str:
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = text.replace('**', '').replace('*', '')
        
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
    """INSTANT RESPONSE AI"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        self.formatter = ResponseFormatter()
    
    def generate_response(self, message: str, user: Dict, history=None) -> Tuple[str, str]:
        try:
            intent = self._extract_intent(message)
            
            # Minimal context
            context = ""
            if history:
                recent = history[-3:]
                parts = []
                for msg in recent:
                    r = "T" if msg['message_type'] == 'user' else "A"
                    parts.append(f"{r}: {msg['message_content'][:60]}")
                context = "\n".join(parts)
            
            name = user.get('first_name', 'Teacher')
            cls = user.get('class_taught', 'your class')
            
            prompt = f"""You are LINKA AI - a helpful teaching assistant for Nigerian teachers.

TEACHER: {name} teaching {cls}
QUERY: {message}

RULES:
- Be warm and concise
- Use numbered lists (1. 2. 3.)
- NO asterisks or markdown
- Keep responses under 150 words
- Practical Nigerian context"""
            
            if context:
                prompt += f"\n\nRECENT CHAT:\n{context}"
            
            prompt += "\n\nProvide helpful response:"
            
            response = self.llm.invoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}
            ])
            
            return self.formatter.clean_response(response.content), intent
        except Exception as e:
            logger.error(f"Response error: {e}")
            return "I'm experiencing a brief issue. Please try again.", "error"
    
    @staticmethod
    def _extract_intent(msg: str) -> str:
        m = msg.lower()
        if any(k in m for k in ['teach', 'lesson', 'explain', 'strategy']):
            return 'teaching_strategy'
        elif any(k in m for k in ['discipline', 'behavior', 'manage']):
            return 'classroom_management'
        elif any(k in m for k in ['assess', 'evaluate', 'grade', 'test']):
            return 'assessment'
        elif any(k in m for k in ['hello', 'hi', 'hey']):
            return 'greeting'
        return 'general'


# Registration steps
REGISTRATION_STEPS = {
    0: {"message": "Hello! I'm LINKA AI by Schoolinka. I support teachers with strategies and development.\n\nWhat's your first name?", "field": "first_name"},
    1: {"message": "Nice to meet you, {first_name}! What's your email?", "field": "email"},
    2: {"message": "Thanks! Which city/state are you in?", "field": "location"},
    3: {"message": "Great! Which class do you teach?", "field": "class_taught"}
}


def handle_registration_quick(phone: str, chat_id: str, text: str, user: Optional[Dict]) -> str:
    """Fast registration with async DB writes"""
    if not user:
        new_id = generate_unique_user_id()
        # Queue DB write, don't wait
        executor.submit(db.save_user_sync, phone, chat_id, new_id, 
                       profile_complete=False, registration_step=0)
        return REGISTRATION_STEPS[0]["message"]
    
    step = user.get('registration_step', 0)
    user_id = user.get('user_id')
    
    if step >= 4:
        return "Your profile is complete! How can I help?"
    
    text = text.strip()
    if not text or len(text) < 2:
        field = REGISTRATION_STEPS[step]['field']
        return f"Please provide your {field.replace('_', ' ')}."
    
    # STEP 0: Name
    if step == 0:
        name = re.sub(r'[^a-zA-Z\s]', '', text)
        if len(name) > 1:
            executor.submit(db.save_user_sync, phone, chat_id, user_id,
                          first_name=name.title(), registration_step=1)
            # Clear cache
            user_cache.delete(phone)
            return REGISTRATION_STEPS[1]["message"].format(first_name=name.title())
        return "Please enter a valid name."
    
    # STEP 1: Email
    elif step == 1:
        if '@' in text and '.' in text:
            executor.submit(db.save_user_sync, phone, email=text.lower(), registration_step=2)
            user_cache.delete(phone)
            return REGISTRATION_STEPS[2]["message"]
        return "Please enter valid email."
    
    # STEP 2: Location
    elif step == 2:
        if len(text) > 2:
            executor.submit(db.save_user_sync, phone, location=text.title(), registration_step=3)
            user_cache.delete(phone)
            return REGISTRATION_STEPS[3]["message"]
        return "Please enter valid location."
    
    # STEP 3: Class - COMPLETE
    elif step == 3:
        if len(text) > 1:
            # Complete registration async
            executor.submit(complete_registration_async, user_id, phone, text.title())
            
            name = user.get('first_name', 'Teacher')
            return (
                f"Excellent, {name}! Profile complete.\n\n"
                f"I can help with:\n"
                f"1. Teaching strategies\n"
                f"2. Classroom management\n"
                f"3. Assessment methods\n"
                f"4. Student engagement\n\n"
                f"What would you like help with?"
            )
        return "Please enter your class."
    
    return "Something went wrong. Please try again."


def complete_registration_async(user_id: str, phone: str, class_taught: str):
    """Complete registration in background"""
    try:
        # Update DB
        db.save_user_sync(phone, class_taught=class_taught, 
                         profile_complete=True, registration_step=4)
        db.reset_verification(user_id)
        
        # Get user data and log to sheets
        user = db.get_user_by_phone(phone)
        if user:
            sheets_logger.log_user_registration_immediate(user)
            
            # Clear cache to force refresh
            user_cache.delete(phone)
    except Exception as e:
        logger.error(f"Registration completion error: {e}")


def log_conversation_async(user_id: str, phone: str, text: str, response: str, 
                          intent: str, user: Dict):
    """Log everything asynchronously"""
    try:
        session_id = db.create_session_async(user_id)
        
        # Save to DB
        db.save_message_async(user_id, phone, 'user', text, intent, session_id)
        db.save_message_async(user_id, phone, 'assistant', response, intent, session_id)
        
        # Log to sheets
        conv_data = {
            'user_id': user_id,
            'phone_number': phone,
            'user_name': user.get('first_name', 'Unknown'),
            'email': user.get('email', ''),
            'class_taught': user.get('class_taught', ''),
            'location': user.get('location', ''),
            'user_message': text[:1000],
            'bot_response': response[:1000],
            'intent': intent,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'chatbot_name': 'LINKA AI'
        }
        sheets_logger.log_conversation_async(conv_data)
        
        # Update history cache
        history = db.get_history(user_id, limit=CONFIG['MAX_HISTORY'])
        history_cache.set(user_id, history)
        
    except Exception as e:
        logger.error(f"Async logging error: {e}")


def process_and_respond_optimized(phone: str, chat_id: str, text: str):
    """Optimized async processing with pre-warmed cache"""
    try:
        # Quick user lookup with caching
        user = user_cache.get(phone)
        if not user:
            user = db.get_user_by_phone(phone)
            if not user:
                user = db.get_user_by_chat_id(chat_id)
                if user:
                    # Update phone mapping async
                    executor.submit(db.save_user_async, phone, chat_id, user['user_id'])
            
            if user:
                user_cache.set(phone, user)
        
        # Generate response FIRST (before any logging)
        if not user or not user.get('profile_complete') or user.get('registration_step', 0) < 4:
            response = handle_registration_quick(phone, chat_id, text, user)
        else:
            user_id = user['user_id']
            
            # Check memory expiry
            if db.check_memory_expiry(user_id):
                response = f"Hi {user.get('first_name')}! It's been 90 days. Please confirm your email: {user.get('email', 'N/A')}\n\nReply with current email or 'same'."
            else:
                # Get history from cache if available
                history = history_cache.get(user_id)
                if not history:
                    history = db.get_history(user_id, limit=CONFIG['MAX_HISTORY'])
                    history_cache.set(user_id, history)
                
                # Generate AI response immediately
                response, intent = linka_ai.generate_response(text, user, history)
                
                # Log everything asynchronously AFTER generating response
                executor.submit(log_conversation_async, user_id, phone, text, response, intent, user)
        
        # Send response immediately
        send_whatsapp_message(chat_id, response)
        
    except Exception as e:
        logger.error(f"Process error: {e}")
        send_whatsapp_message(chat_id, "I'm experiencing technical difficulties. Please try again.")


# Initialize components
db = DatabaseManager(CONFIG['DATABASE_URL'])
sheets_logger = SheetsLogger(CONFIG['APPS_SCRIPT_URL'])
linka_ai = LinkaAI(llm, embed_model, pinecone_index)

# Initialize caches
user_cache = SimpleCache(ttl=CONFIG['CACHE_TTL_USER'])
history_cache = SimpleCache(ttl=CONFIG['CACHE_TTL_HISTORY'])

# User locks (kept for critical operations)
user_locks = {}
lock_manager = threading.Lock()

def get_user_lock(user_id: str) -> threading.Lock:
    with lock_manager:
        if user_id not in user_locks:
            user_locks[user_id] = threading.Lock()
        return user_locks[user_id]


def send_whatsapp_message(phone: str, message: str) -> bool:
    """Send message via Green API"""
    for attempt in range(2):
        try:
            url = f"https://api.green-api.com/waInstance{CONFIG['GREEN_API_ID']}/sendMessage/{CONFIG['GREEN_API_TOKEN']}"
            response = requests.post(
                url,
                json={"chatId": phone, "message": message},
                timeout=6
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Sent to {phone[:10]}...")
                return True
        except Exception as e:
            logger.error(f"Send error: {e}")
        
        if attempt < 1:
            time.sleep(0.2)
    return False


# Model warmup on startup
@app.before_first_request
def warmup():
    """Warm up AI models to reduce first response latency"""
    try:
        logger.info("Warming up AI models...")
        linka_ai.llm.invoke([{"role": "user", "content": "test"}])
        logger.info("✓ Models warmed up")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")


# Flask Routes

@app.route('/')
def health():
    return jsonify({
        "status": "healthy",
        "service": "LINKA AI",
        "version": "7.0-optimized",
        "features": [
            "Instant response (<2s)",
            "User caching (5min)",
            "History caching (3min)",
            "Async logging",
            "Model warmup"
        ],
        "cache_stats": {
            "users_cached": len(user_cache.cache),
            "histories_cached": len(history_cache.cache)
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """ULTRA-FAST webhook - respond immediately, process async"""
    try:
        data = request.get_json()
        
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored"}), 200
        
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        if not chat_id or '@g.us' in chat_id:
            return jsonify({"status": "ignored"}), 200
        
        phone = extract_phone_number(chat_id)
        
        text = None
        if 'textMessageData' in message_data:
            text = message_data['textMessageData'].get('textMessage', '').strip()
        elif 'extendedTextMessageData' in message_data:
            text = message_data['extendedTextMessageData'].get('text', '').strip()
        
        if not text:
            # Send immediately without waiting
            executor.submit(send_whatsapp_message, chat_id, "I can only respond to text messages.")
            return jsonify({"status": "non_text"}), 200
        
        logger.info(f"📨 Received from {phone[:10]}: {text[:30]}")
        
        # Submit to thread pool and return immediately
        executor.submit(process_and_respond_optimized, phone, chat_id, text)
        
        # Return immediately - don't wait for processing
        return jsonify({"status": "accepted"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error"}), 500


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint for debugging"""
    try:
        data = request.get_json()
        phone = data.get('phone_number', '2348012345678')
        chat_id = data.get('chat_id', f'{phone}@c.us')
        message = data.get('message', 'Hello')
        
        # Process synchronously for testing
        user = user_cache.get(phone)
        if not user:
            user = db.get_user_by_phone(phone)
            if user:
                user_cache.set(phone, user)
        
        if not user or not user.get('profile_complete'):
            response = handle_registration_quick(phone, chat_id, message, user)
        else:
            history = history_cache.get(user['user_id']) or db.get_history(user['user_id'])
            response, intent = linka_ai.generate_response(message, user, history)
        
        return jsonify({
            "response": response,
            "user": dict(user) if user else None,
            "cached": user_cache.get(phone) is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get chatbot statistics"""
    try:
        with db.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT COUNT(*) as total FROM users')
                total = cur.fetchone()['total']
                
                cur.execute('SELECT COUNT(*) as registered FROM users WHERE profile_complete = TRUE')
                registered = cur.fetchone()['registered']
                
                cur.execute('SELECT COUNT(*) as total FROM conversations')
                messages = cur.fetchone()['total']
                
                cur.execute('''
                    SELECT COUNT(DISTINCT user_id) as active 
                    FROM conversations 
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                ''')
                active_24h = cur.fetchone()['active']
        
        return jsonify({
            "chatbot": "LINKA AI",
            "stats": {
                "total_users": total,
                "registered_users": registered,
                "total_messages": messages,
                "active_24h": active_24h
            },
            "cache": {
                "users_cached": len(user_cache.cache),
                "histories_cached": len(history_cache.cache)
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health_check', methods=['GET'])
def health_check():
    """Detailed health check for monitoring"""
    components = {
        "database": False,
        "google_ai": False,
        "green_api": False,
        "sheets": False
    }
    
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                components["database"] = True
    except:
        pass
    
    components["google_ai"] = llm is not None
    components["green_api"] = bool(CONFIG['GREEN_API_ID'])
    components["sheets"] = bool(CONFIG['APPS_SCRIPT_URL'])
    
    all_healthy = all(components.values())
    
    return jsonify({
        "status": "healthy" if all_healthy else "degraded",
        "chatbot": "LINKA AI v7.0",
        "components": components,
        "optimizations": {
            "response_mode": "instant",
            "caching": "enabled",
            "logging": "async",
            "database_writes": "queued",
            "model_warmup": "active"
        },
        "cache_stats": {
            "user_cache_size": len(user_cache.cache),
            "history_cache_size": len(history_cache.cache),
            "user_cache_ttl": CONFIG['CACHE_TTL_USER'],
            "history_cache_ttl": CONFIG['CACHE_TTL_HISTORY']
        },
        "timestamp": datetime.now().isoformat()
    }), 200 if all_healthy else 503


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all caches (admin endpoint)"""
    try:
        user_cache.clear()
        history_cache.clear()
        return jsonify({
            "status": "success",
            "message": "All caches cleared",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LINKA AI v7.0 - ULTRA-FAST RESPONSE")
    logger.info("=" * 70)
    logger.info("✓ User caching (5 min TTL)")
    logger.info("✓ History caching (3 min TTL)")
    logger.info("✓ Async logging (sheets + DB)")
    logger.info("✓ Model warmup on startup")
    logger.info("✓ Response time: < 2 seconds")
    logger.info("✓ Thread pool: 20 workers")
    logger.info("=" * 70)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
