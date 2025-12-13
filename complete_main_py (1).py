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
import signal

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only console logs for free tier
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - OPTIMIZED FOR FREE TIER
CONFIG = {
    'PINECONE_API_KEY': os.getenv("PINECONE_API_KEY"),
    'GOOGLE_API_KEY': os.getenv("GOOGLE_API_KEY"), 
    'GREEN_API_ID': os.getenv("GREEN_API_ID_INSTANCE"), 
    'GREEN_API_TOKEN': os.getenv("GREEN_API_TOKEN"), 
    'APPS_SCRIPT_URL': os.getenv("APPS_SCRIPT_URL"), 
    'SPREADSHEET_ID': os.getenv("SPREADSHEET_ID"), 
    'DATABASE_URL': os.getenv("DATABASE_URL"),  
    'MAX_HISTORY': 6,  # Reduced for memory efficiency
    'INDEX_NAME': 'coach',
    'MEMORY_DAYS': 90,
    'RAG_TIMEOUT': 1.5,
    'LLM_TIMEOUT': 3,  # Faster timeout
    'MAX_TOKENS': 500,  # Reduced for speed
    'SHEETS_TIMEOUT': 5,  # Faster timeout
    'CACHE_TTL_USER': 180,  # 3 minutes (reduced memory)
    'CACHE_TTL_HISTORY': 120,  # 2 minutes
    'MAX_CACHE_SIZE': 50,  # Limit cache size for free tier
    'DB_POOL_MIN': 1,  # Minimal connections
    'DB_POOL_MAX': 5,  # Reduced for free tier
    'THREAD_WORKERS': 8  # Reduced workers for 512MB RAM
}

# Validate required keys
required_keys = ['PINECONE_API_KEY', 'GOOGLE_API_KEY', 'GREEN_API_ID', 'GREEN_API_TOKEN', 'DATABASE_URL']
missing_keys = [key for key in required_keys if not CONFIG[key]]

if missing_keys:
    logger.error(f"Missing configuration: {', '.join(missing_keys)}")
    raise ValueError(f"Missing environment variables: {', '.join(missing_keys)}")

# Thread pool - LIMITED for free tier
executor = ThreadPoolExecutor(max_workers=CONFIG['THREAD_WORKERS'])

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
    """Memory-efficient cache with size limits for free tier"""
    
    def __init__(self, ttl=300, max_size=50):
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
            # Enforce size limit - evict least recently used
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])
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


class SheetsLogger:
    """Lightweight sheets logger for free tier"""
    
    def __init__(self, apps_script_url: str):
        self.apps_script_url = apps_script_url
        self.queue = Queue(maxsize=100)  # Limit queue size
        
        if apps_script_url:
            self._start_worker()
    
    def _start_worker(self):
        def worker():
            while True:
                try:
                    if not self.queue.empty():
                        item = self.queue.get()
                        self._log_sync(item)
                    time.sleep(0.5)  # Slower processing for free tier
                except Exception as e:
                    logger.error(f"Sheets worker error: {e}")
        
        threading.Thread(target=worker, daemon=True).start()
        logger.info("✓ Sheets worker started")
    
    def log_user_registration_async(self, user_data: Dict):
        """Log registration asynchronously"""
        if not self.apps_script_url:
            return
        
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
        
        try:
            self.queue.put_nowait(payload)
        except:
            logger.warning("Sheets queue full, dropping log")
    
    def log_conversation_async(self, conv_data: Dict):
        """Add conversation to async queue"""
        if not self.apps_script_url:
            return
            
        payload = {
            'type': 'conversation',
            'data': conv_data,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.queue.put_nowait(payload)
        except:
            logger.warning("Sheets queue full, dropping log")
    
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
            logger.error(f"Sheets log error: {e}")
            return False


class DatabaseManager:
    """Lightweight database manager for free tier"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool = None
        self.write_queue = Queue(maxsize=200)  # Limit queue size
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
                    time.sleep(0.2)
                except Exception as e:
                    logger.error(f"DB worker error: {e}")
        
        threading.Thread(target=worker, daemon=True).start()
    
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
            logger.warning("DB queue full, dropping write")
    
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
        """SYNC save - used for critical operations"""
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
    """INSTANT RESPONSE AI - optimized for free tier"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        self.formatter = ResponseFormatter()
    
    def generate_response(self, message: str, user: Dict, history=None) -> Tuple[str, str]:
        try:
            intent = self._extract_intent(message)
            
            # Minimal context for speed
            context = ""
            if history:
                recent = history[-2:]  # Only 2 most recent
                parts = []
                for msg in recent:
                    r = "T" if msg['message_type'] == 'user' else "A"
                    parts.append(f"{r}: {msg['message_content'][:50]}")
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
- Keep responses under 120 words
- Practical Nigerian context"""
            
            if context:
                prompt += f"\n\nRECENT:\n{context}"
            
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
    
    if step == 0:
        name = re.sub(r'[^a-zA-Z\s]', '', text)
        if len(name) > 1:
            executor.submit(db.save_user_sync, phone, chat_id, user_id,
                          first_name=name.title(), registration_step=1)
            user_cache.delete(phone)
            return REGISTRATION_STEPS[1]["message"].format(first_name=name.title())
        return "Please enter a valid name."
    
    elif step == 1:
        if '@' in text and '.' in text:
            executor.submit(db.save_user_sync, phone, email=text.lower(), registration_step=2)
            user_cache.delete(phone)
            return REGISTRATION_STEPS[2]["message"]
        return "Please enter valid email."
    
    elif step == 2:
        if len(text) > 2:
            executor.submit(db.save_user_sync, phone, location=text.title(), registration_step=3)
            user_cache.delete(phone)
            return REGISTRATION_STEPS[3]["message"]
        return "Please enter valid location."
    
    elif step == 3:
        if len(text) > 1:
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
        db.save_user_sync(phone, class_taught=class_taught, 
                         profile_complete=True, registration_step=4)
        db.reset_verification(user_id)
        
        user = db.get_user_by_phone(phone)
        if user:
            sheets_logger.log_user_registration_async(user)
            user_cache.delete(phone)
    except Exception as e:
        logger.error(f"Registration completion error: {e}")


def log_conversation_async(user_id: str, phone: str, text: str, response: str, 
                          intent: str, user: Dict):
    """Log everything asynchronously"""
    try:
        session_id = db.create_session_async(user_id)
        
        db.save_message_async(user_id, phone, 'user', text, intent, session_id)
        db.save_message_async(user_id, phone, 'assistant', response, intent, session_id)
        
        conv_data = {
            'user_id': user_id,
            'phone_number': phone,
            'user_name': user.get('first_name', 'Unknown'),
            'email': user.get('email', ''),
            'class_taught': user.get('class_taught', ''),
            'location': user.get('location', ''),
            'user_message': text[:800],
            'bot_response': response[:800],
            'intent': intent,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'chatbot_name': 'LINKA AI'
        }
        sheets_logger.log_conversation_async(conv_data)
        
        history = db.get_history(user_id, limit=CONFIG['MAX_HISTORY'])
        history_cache.set(user_id, history)
        
    except Exception as e:
        logger.error(f"Async logging error: {e}")


def process_and_respond_optimized(phone: str, chat_id: str, text: str):
    """Optimized processing for free tier"""
    try:
        user = user_cache.get(phone)
        if not user:
            user = db.get_user_by_phone(phone)
            if not user:
                user = db.get_user_by_chat_id(chat_id)
                if user:
                    executor.submit(db.save_user_async, phone, chat_id, user['user_id'])
            
            if user:
                user_cache.set(phone, user)
        
        if not user or not user.get('profile_complete') or user.get('registration_step', 0) < 4:
            response = handle_registration_quick(phone, chat_id, text, user)
        else:
            user_id = user['user_id']
            
            if db.check_memory_expiry(user_id):
                response = f"Hi {user.get('first_name')}! It's been 90 days. Please confirm your email: {user.get('email', 'N/A')}\n\nReply with current email or 'same'."
            else:
                history = history_cache.get(user_id)
                if not history:
                    history = db.get_history(user_id, limit=CONFIG['MAX_HISTORY