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
from queue import Queue
import uuid

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
    'MAX_HISTORY': 20,  # Reduced from 25
    'INDEX_NAME': 'coach',
    'MEMORY_DAYS': 90,
    'RAG_TIMEOUT': 3,  # RAG retrieval timeout in seconds
    'LLM_TIMEOUT': 8,  # LLM generation timeout
    'MAX_TOKENS': 1200  # Reduced from 1500 for faster responses
}

# Validate required keys
required_keys = ['PINECONE_API_KEY', 'GOOGLE_API_KEY', 'GREEN_API_ID', 'GREEN_API_TOKEN', 'DATABASE_URL']
missing_keys = [key for key in required_keys if not CONFIG[key]]

if missing_keys:
    logger.error(f"Missing required configuration: {', '.join(missing_keys)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

# Initialize AI services
pinecone_index = None
embed_model = None
llm = None

try:
    logger.info("Initializing Pinecone...")
    pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
    
    try:
        pinecone_index = pc.Index(CONFIG['INDEX_NAME'])
        stats = pinecone_index.describe_index_stats()
        logger.info(f"✓ Connected to Pinecone index '{CONFIG['INDEX_NAME']}' - Vectors: {stats.get('total_vector_count', 0)}")
    except Exception as e:
        logger.error(f"✗ Could not connect to Pinecone index '{CONFIG['INDEX_NAME']}': {e}")
        logger.warning("Continuing without RAG support")
    
    logger.info("Initializing Google AI services...")
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=CONFIG['GOOGLE_API_KEY']
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=CONFIG['GOOGLE_API_KEY'],
        temperature=0.5,
        max_tokens=CONFIG['MAX_TOKENS'],
        timeout=CONFIG['LLM_TIMEOUT']
    )
    logger.info("✓ Google AI services initialized")
    
except Exception as e:
    logger.error(f"✗ Critical initialization error: {e}")
    raise


def extract_phone_number(chat_id: str) -> str:
    """Extract phone number from WhatsApp chat_id"""
    phone = re.sub(r'@.*$', '', chat_id)
    phone = re.sub(r'\D', '', phone)
    return phone or chat_id


def generate_unique_user_id() -> str:
    """Generate a unique user ID"""
    return str(uuid.uuid4())


class DatabaseManager:
    """Handles all PostgreSQL database operations with unique user ID isolation"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool = None
        self._init_connection_pool()
        self._init_db()
    
    def _init_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=3,
                maxconn=25,
                dsn=self.database_url
            )
            logger.info("✓ PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    def _init_db(self):
        """Initialize database tables with proper isolation"""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    # Users table with unique user_id
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
                            logged_to_sheets BOOLEAN DEFAULT FALSE,
                            needs_reverification BOOLEAN DEFAULT FALSE
                        )
                    ''')
                    
                    # Conversations table with user_id foreign key
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS conversations (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(50) NOT NULL,
                            phone_number VARCHAR(20) NOT NULL,
                            message_type VARCHAR(10) CHECK(message_type IN ('user', 'assistant')),
                            message_content TEXT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            intent VARCHAR(50),
                            logged_to_sheets BOOLEAN DEFAULT FALSE,
                            session_id VARCHAR(50),
                            FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                        )
                    ''')
                    
                    # Session tracking table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS sessions (
                            session_id VARCHAR(50) PRIMARY KEY,
                            user_id VARCHAR(50) NOT NULL,
                            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            is_active BOOLEAN DEFAULT TRUE,
                            FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                        )
                    ''')
                    
                    # Create indexes for performance
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON conversations (user_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_number ON conversations (phone_number)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON conversations (session_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone_number)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_chat_id ON users (chat_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sheets_log ON conversations (logged_to_sheets)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_needs_reverification ON users (needs_reverification)')
                    
                    conn.commit()
                    logger.info("✓ PostgreSQL database initialized with user isolation")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    @contextmanager
    def get_conn(self):
        """Context manager for database connections from pool"""
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    def check_memory_expiry(self, user_id: str) -> bool:
        """Check if user's memory has expired (90 days)"""
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('''
                        SELECT last_verification, needs_reverification 
                        FROM users 
                        WHERE user_id = %s
                    ''', (user_id,))
                    result = cursor.fetchone()
                    
                    if not result:
                        return True
                    
                    if result['needs_reverification']:
                        return True
                    
                    last_verification = result['last_verification']
                    if last_verification:
                        days_since = (datetime.now() - last_verification).days
                        if days_since >= CONFIG['MEMORY_DAYS']:
                            cursor.execute('''
                                UPDATE users 
                                SET needs_reverification = TRUE 
                                WHERE user_id = %s
                            ''', (user_id,))
                            conn.commit()
                            return True
                    
                    return False
        except Exception as e:
            logger.error(f"Error checking memory expiry: {e}")
            return False
    
    def reset_verification(self, user_id: str):
        """Reset verification timestamp after successful re-verification"""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        UPDATE users 
                        SET last_verification = %s, needs_reverification = FALSE 
                        WHERE user_id = %s
                    ''', (datetime.now(), user_id))
                    conn.commit()
                    logger.info(f"✓ Verification reset for user {user_id}")
        except Exception as e:
            logger.error(f"Error resetting verification: {e}")
    
    def get_user_by_phone(self, phone_number: str) -> Optional[Dict]:
        """Get user by phone number"""
        phone = re.sub(r'\D', '', phone_number)
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('SELECT * FROM users WHERE phone_number = %s', (phone,))
                    row = cursor.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user by phone: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by unique user_id"""
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('SELECT * FROM users WHERE user_id = %s', (user_id,))
                    row = cursor.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def get_user_by_chat_id(self, chat_id: str) -> Optional[Dict]:
        """Get user by chat_id (backup lookup)"""
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('SELECT * FROM users WHERE chat_id = %s', (chat_id,))
                    row = cursor.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user by chat_id: {e}")
            return None
    
    def save_user(self, phone_number: str, chat_id: str = None, user_id: str = None, **kwargs) -> str:
        """Save or update user - returns user_id"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('SELECT user_id FROM users WHERE phone_number = %s', (phone,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        current_user_id = existing['user_id']
                        updates = []
                        values = []
                        
                        if chat_id:
                            updates.append('chat_id = %s')
                            values.append(chat_id)
                        
                        for k, v in kwargs.items():
                            updates.append(f'{k} = %s')
                            values.append(v)
                        
                        updates.append('last_interaction = %s')
                        updates.append('total_messages = total_messages + 1')
                        values.append(datetime.now())
                        values.append(current_user_id)
                        
                        update_str = ', '.join(updates)
                        cursor.execute(f'UPDATE users SET {update_str} WHERE user_id = %s', values)
                        conn.commit()
                        return current_user_id
                    else:
                        new_user_id = user_id or generate_unique_user_id()
                        
                        fields = ['user_id', 'phone_number']
                        values = [new_user_id, phone]
                        placeholders = ['%s', '%s']
                        
                        if chat_id:
                            fields.append('chat_id')
                            values.append(chat_id)
                            placeholders.append('%s')
                        
                        for k, v in kwargs.items():
                            fields.append(k)
                            values.append(v)
                            placeholders.append('%s')
                        
                        fields.append('total_messages')
                        values.append(1)
                        placeholders.append('%s')
                        
                        cursor.execute(
                            f'INSERT INTO users ({", ".join(fields)}) VALUES ({", ".join(placeholders)})', 
                            values
                        )
                        conn.commit()
                        logger.info(f"✓ New user {new_user_id} created")
                        return new_user_id
        except Exception as e:
            logger.error(f"Error saving user {phone_number}: {e}")
            return None
    
    def mark_user_logged_to_sheets(self, user_id: str):
        """Mark user as logged to Google Sheets"""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('UPDATE users SET logged_to_sheets = TRUE WHERE user_id = %s', (user_id,))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error marking user logged: {e}")
    
    def create_session(self, user_id: str) -> str:
        """Create a new session for user"""
        session_id = str(uuid.uuid4())
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        INSERT INTO sessions (session_id, user_id, started_at, last_activity)
                        VALUES (%s, %s, %s, %s)
                    ''', (session_id, user_id, datetime.now(), datetime.now()))
                    conn.commit()
                    return session_id
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return session_id
    
    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        UPDATE sessions 
                        SET last_activity = %s 
                        WHERE session_id = %s
                    ''', (datetime.now(), session_id))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error updating session: {e}")
    
    def save_message(self, user_id: str, phone_number: str, msg_type: str, 
                    content: str, intent: str = None, session_id: str = None):
        """Save conversation message with user isolation"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        INSERT INTO conversations 
                        (user_id, phone_number, message_type, message_content, intent, session_id, logged_to_sheets)
                        VALUES (%s, %s, %s, %s, %s, %s, FALSE)
                    ''', (user_id, phone, msg_type, content, intent, session_id))
                    conn.commit()
                    self._cleanup_history(user_id)
        except Exception as e:
            logger.error(f"Error saving message for user {user_id}: {e}")
    
    def mark_message_logged_to_sheets(self, message_id: int):
        """Mark message as logged to Google Sheets"""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('UPDATE conversations SET logged_to_sheets = TRUE WHERE id = %s', (message_id,))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error marking message logged: {e}")
    
    def get_unlogged_messages(self, limit: int = 50) -> List[Dict]:
        """Get messages not yet logged to sheets"""
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('''
                        SELECT c.*, u.user_id, u.first_name, u.class_taught, u.location, u.email
                        FROM conversations c
                        LEFT JOIN users u ON c.user_id = u.user_id
                        WHERE c.logged_to_sheets = FALSE
                        ORDER BY c.timestamp ASC
                        LIMIT %s
                    ''', (limit,))
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting unlogged messages: {e}")
            return []
    
    def get_history(self, user_id: str, limit: int = None) -> List[Dict]:
        """Get conversation history for specific user_id"""
        limit = limit or CONFIG['MAX_HISTORY']
        
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('''
                        SELECT message_type, message_content, timestamp, intent
                        FROM conversations 
                        WHERE user_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    ''', (user_id, limit))
                    results = cursor.fetchall()
                    return [dict(row) for row in reversed(results)]
        except Exception as e:
            logger.error(f"Error getting history for user {user_id}: {e}")
            return []
    
    def _cleanup_history(self, user_id: str):
        """Clean up old messages beyond max history for specific user"""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        DELETE FROM conversations 
                        WHERE user_id = %s AND id NOT IN (
                            SELECT id FROM conversations 
                            WHERE user_id = %s 
                            ORDER BY timestamp DESC 
                            LIMIT %s
                        )
                    ''', (user_id, user_id, CONFIG['MAX_HISTORY'] * 2))
                    conn.commit()
        except Exception as e:
            logger.error(f"Cleanup error for user {user_id}: {e}")
    
    def close_pool(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("✓ Database connection pool closed")


class ResponseFormatter:
    """Format responses according to WhatsApp guidelines"""
    
    @staticmethod
    def clean_response(text: str) -> str:
        """Remove markdown and format for WhatsApp"""
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'#{1,6}\s', '', text)
        text = text.replace('**', '').replace('*', '')
        
        lines = text.split('\n')
        formatted_lines = []
        list_counter = 1
        in_list = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                formatted_lines.append('')
                in_list = False
                list_counter = 1
                continue
            
            if re.match(r'^[-•·]\s+', stripped):
                content = re.sub(r'^[-•·]\s+', '', stripped)
                formatted_lines.append(f"{list_counter}. {content}")
                list_counter += 1
                in_list = True
            else:
                formatted_lines.append(stripped)
                if not stripped[0].isdigit():
                    in_list = False
        
        result = '\n'.join(formatted_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()


class SheetsLogger:
    """Handles real-time logging to Google Sheets with retry and queue system"""
    
    def __init__(self, apps_script_url: str, db_manager):
        self.apps_script_url = apps_script_url
        self.db = db_manager
        self.queue = Queue()
        self.is_processing = False
        
        self._start_background_worker()
    
    def _start_background_worker(self):
        """Start background thread to process logging queue"""
        def worker():
            while True:
                try:
                    time.sleep(3)
                    self._process_queue()
                except Exception as e:
                    logger.error(f"Sheets worker error: {e}")
                time.sleep(5)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        logger.info("✓ Sheets logging worker started")
    
    def log_user_registration(self, user_data: Dict) -> bool:
        """Log complete user registration to sheets (BLOCKING - waits for success)"""
        if not self.apps_script_url:
            logger.warning("Apps Script URL not configured - skipping sheets logging")
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
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
                
                response = requests.post(
                    self.apps_script_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.db.mark_user_logged_to_sheets(user_data['user_id'])
                    logger.info(f"✓ User registration logged to sheets: {user_data['user_id']}")
                    return True
                else:
                    logger.warning(f"Sheets log attempt {attempt + 1} failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Sheets log error (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(1)
        
        return False
    
    def log_conversation(self, conversation_data: Dict, message_id: int = None):
        """Add conversation to logging queue (non-blocking)"""
        self.queue.put({
            'type': 'conversation',
            'data': conversation_data,
            'message_id': message_id,
            'timestamp': datetime.now().isoformat()
        })
    
    def _process_queue(self):
        """Process items in the logging queue"""
        if self.is_processing or self.queue.empty():
            return
        
        if not self.apps_script_url:
            return
        
        self.is_processing = True
        processed = 0
        
        try:
            unlogged = self.db.get_unlogged_messages(limit=20)
            
            for msg in unlogged:
                if processed >= 10:
                    break
                
                payload = {
                    'type': 'conversation',
                    'data': {
                        'user_id': msg.get('user_id'),
                        'phone_number': msg.get('phone_number'),
                        'user_name': msg.get('first_name', 'Unknown'),
                        'email': msg.get('email', ''),
                        'class_taught': msg.get('class_taught', ''),
                        'location': msg.get('location', ''),
                        'message_type': msg['message_type'],
                        'message_content': msg['message_content'][:1000],
                        'intent': msg.get('intent', ''),
                        'timestamp': str(msg['timestamp']),
                        'chatbot_name': 'LINKA AI'
                    },
                    'timestamp': str(msg['timestamp'])
                }
                
                try:
                    response = requests.post(
                        self.apps_script_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=8
                    )
                    
                    if response.status_code == 200:
                        self.db.mark_message_logged_to_sheets(msg['id'])
                        processed += 1
                    else:
                        break
                        
                except Exception as e:
                    logger.error(f"Error logging message {msg['id']}: {e}")
                    break
                
                time.sleep(0.3)
            
            while not self.queue.empty() and processed < 10:
                try:
                    item = self.queue.get_nowait()
                    
                    response = requests.post(
                        self.apps_script_url,
                        json=item,
                        headers={'Content-Type': 'application/json'},
                        timeout=8
                    )
                    
                    if response.status_code == 200:
                        if item.get('message_id'):
                            self.db.mark_message_logged_to_sheets(item['message_id'])
                        processed += 1
                    else:
                        self.queue.put(item)
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing queue item: {e}")
                    break
                
                time.sleep(0.3)
            
            if processed > 0:
                logger.info(f"✓ Processed {processed} items to sheets")
                
        finally:
            self.is_processing = False


class LinkaAI:
    """LINKA AI chatbot with RAG and context awareness - OPTIMIZED FOR SPEED"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        self.formatter = ResponseFormatter()
    
    def get_rag_content(self, query: str, intent: str = None) -> Tuple[str, List[str]]:
        """Retrieve relevant content from Pinecone WITH TIMEOUT"""
        if not self.pinecone_index:
            return "", []
        
        try:
            enhanced_query = f"{intent} {query}" if intent else query
            
            # Use threading to enforce timeout
            result_container = {'content': '', 'sources': []}
            
            def retrieve():
                try:
                    query_embed = self.embed_model.embed_query(enhanced_query)
                    results = self.pinecone_index.query(
                        vector=query_embed,
                        top_k=3,  # Reduced from 5 for speed
                        include_metadata=True,
                        include_values=False
                    )
                    
                    contents, sources = [], []
                    for match in results.get('matches', []):
                        score = match.get('score', 0)
                        
                        if score > 0.70:  # Higher threshold for faster, more relevant results
                            text = match.get('metadata', {}).get('text', '')
                            source = match.get('metadata', {}).get('source', 'Knowledge Base')
                            
                            if text:
                                contents.append(text[:400])  # Reduced from 500
                                sources.append(source)
                    
                    result_container['content'] = '\n\n'.join(contents)
                    result_container['sources'] = sources
                    
                except Exception as e:
                    logger.error(f"RAG retrieval error: {e}")
            
            thread = threading.Thread(target=retrieve, daemon=True)
            thread.start()
            thread.join(timeout=CONFIG['RAG_TIMEOUT'])
            
            if thread.is_alive():
                logger.warning("RAG retrieval timed out - continuing without RAG")
                return "", []
            
            if result_container['content']:
                logger.info(f"✓ Retrieved {len(result_container['sources'])} documents")
                return result_container['content'], result_container['sources']
            
            return "", []
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return "", []
    
    def generate_response(self, message: str, user_profile: Dict, 
                         history: List[Dict] = None) -> Tuple[str, str]:
        """Generate formatted AI response - OPTIMIZED"""
        try:
            intent = self._extract_intent(message)
            
            # Only use RAG for complex queries
            rag_content = ""
            sources = []
            if self._needs_rag(message, intent):
                rag_content, sources = self.get_rag_content(message, intent)
            
            # Build concise context from recent history
            context = ""
            if history:
                recent = history[-6:]  # Reduced from 10
                context_parts = []
                for msg in recent:
                    role = "Teacher" if msg['message_type'] == 'user' else "AI"
                    content = msg['message_content'][:100]  # Reduced from 150
                    context_parts.append(f"{role}: {content}")
                context = "\n".join(context_parts)
            
            system_prompt = self._build_system_prompt(
                user_profile, rag_content, context, intent, message
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            response = self.llm.invoke(messages)
            clean_response = self.formatter.clean_response(response.content)
            
            return clean_response, intent
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._get_fallback_response(intent), "error"
    
    def _needs_rag(self, message: str, intent: str) -> bool:
        """Determine if query needs RAG lookup"""
        msg_lower = message.lower()
        
        # Skip RAG for simple queries
        simple_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|thanks|thank you)',
            r'^(how are you|what.*your name)',
            r'^(yes|no|ok|okay)',
        ]
        
        if any(re.match(pattern, msg_lower) for pattern in simple_patterns):
            return False
        
        # Use RAG for teaching-related queries
        needs_rag_intents = ['teaching_strategy', 'curriculum', 'assessment', 'classroom_management']
        return intent in needs_rag_intents or len(message.split()) > 8
    
    def _build_system_prompt(self, user_profile: Dict, rag_content: str, 
                            context: str, intent: str, query: str) -> str:
        """Build optimized system prompt"""
        
        name = user_profile.get('first_name', 'Teacher')
        class_info = user_profile.get('class_taught', 'their class')
        
        prompt = f"""You are LINKA AI by Schoolinka - a helpful teaching assistant for Nigerian teachers.

TEACHER: {name} teaching {class_info}
QUERY: {query}
INTENT: {intent}

RESPONSE RULES:
1. Be warm, professional, and concise
2. Use numbers (1. 2. 3.) for lists only
3. Keep paragraphs short (2-3 sentences)
4. NO asterisks or markdown
5. Provide practical Nigerian-context advice
6. Address large class sizes and limited resources"""
        
        if rag_content:
            prompt += f"\n\nKNOWLEDGE BASE:\n{rag_content}"
        
        if context:
            prompt += f"\n\nRECENT CHAT:\n{context}"
        
        prompt += "\n\nProvide a helpful response."
        
        return prompt
    
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Extract intent from message - OPTIMIZED"""
        msg = message.lower()
        
        # Fast keyword matching
        if any(kw in msg for kw in ['teach', 'lesson', 'explain', 'learn', 'strategy', 'method']):
            return 'teaching_strategy'
        elif any(kw in msg for kw in ['discipline', 'behavior', 'manage', 'control', 'disruptive']):
            return 'classroom_management'
        elif any(kw in msg for kw in ['assess', 'evaluate', 'grade', 'test', 'exam']):
            return 'assessment'
        elif any(kw in msg for kw in ['stress', 'tired', 'overwhelmed', 'burnout']):
            return 'wellbeing'
        elif any(kw in msg for kw in ['curriculum', 'syllabus', 'topic', 'subject']):
            return 'curriculum'
        elif any(kw in msg for kw in ['parent', 'guardian', 'meeting']):
            return 'parent_communication'
        elif any(kw in msg for kw in ['hello', 'hi', 'hey', 'good morning']):
            return 'greeting'
        elif any(kw in msg for kw in ['thank', 'thanks', 'appreciate']):
            return 'gratitude'
        
        return 'general'
    
    @staticmethod
    def _get_fallback_response(intent: str) -> str:
        """Fallback response on error"""
        return "I'm LINKA AI, and I'm experiencing a brief technical issue. Please try your question again, and I'll be happy to help you."


# Registration step templates - OPTIMIZED FOR SPEED
REGISTRATION_STEPS = {
    0: {
        "message": "Hello! I'm LINKA AI by Schoolinka. I'm here to support you with teaching strategies and professional development.\n\nWhat's your first name?",
        "field": "first_name"
    },
    1: {
        "message": "Nice to meet you, {first_name}! What's your email address?",
        "field": "email"
    },
    2: {
        "message": "Thanks! Which city or state are you in?",
        "field": "location"
    },
    3: {
        "message": "Great! Which class do you teach?",
        "field": "class_taught"
    }
}


def handle_registration(user_id: str, phone_number: str, chat_id: str, text: str, 
                       user: Optional[Dict], sheets_logger) -> Tuple[str, bool]:
    """Handle multi-step user registration - COLLECTS ALL DATA THEN UPLOADS ONCE"""
    
    if not user:
        # New user - create with unique ID
        new_user_id = generate_unique_user_id()
        db.save_user(phone_number, chat_id=chat_id, user_id=new_user_id, 
                    profile_complete=False, registration_step=0)
        logger.info(f"✓ New user started registration: {new_user_id}")
        return REGISTRATION_STEPS[0]["message"], False
    
    step = user.get('registration_step', 0)
    current_user_id = user.get('user_id')
    
    # Already registered
    if step >= 4:
        return "Your profile is complete! How can I help you today?", True
    
    text = text.strip()
    
    # Validate input
    if not text or len(text) < 2:
        current_field = REGISTRATION_STEPS[step]['field']
        return f"Please provide your {current_field.replace('_', ' ')} to continue.", False
    
    # STEP 0: Collect First Name
    if step == 0:
        clean_name = re.sub(r'[^a-zA-Z\s]', '', text)
        if len(clean_name) > 1:
            db.save_user(phone_number, chat_id=chat_id, user_id=current_user_id,
                        first_name=clean_name.title(), registration_step=1)
            next_msg = REGISTRATION_STEPS[1]["message"].format(first_name=clean_name.title())
            return next_msg, False
        else:
            return "Please enter a valid name.", False
    
    # STEP 1: Collect Email
    elif step == 1:
        if '@' in text and '.' in text:
            email_lower = text.lower()
            db.save_user(phone_number, email=email_lower, registration_step=2)
            return REGISTRATION_STEPS[2]["message"], False
        else:
            return "Please enter a valid email address (e.g., name@example.com).", False
    
    # STEP 2: Collect Location
    elif step == 2:
        if len(text) > 2:
            db.save_user(phone_number, location=text.title(), registration_step=3)
            return REGISTRATION_STEPS[3]["message"], False
        else:
            return "Please enter a valid location.", False
    
    # STEP 3: Collect Class & COMPLETE REGISTRATION
    elif step == 3:
        if len(text) > 1:
            # Save class and mark profile as complete
            db.save_user(phone_number, class_taught=text.title(), 
                        profile_complete=True, registration_step=4)
            
            # Reset verification timestamp
            db.reset_verification(current_user_id)
            
            # Fetch complete user data
            phone = re.sub(r'\D', '', phone_number)
            updated_user = db.get_user_by_phone(phone)
            
            if not updated_user:
                logger.error(f"Failed to retrieve user after registration: {phone}")
                return "Registration completed! How can I help you today?", True
            
            # Upload complete profile to sheets AND database (BLOCKING)
            success = sheets_logger.log_user_registration(updated_user)
            
            if success:
                logger.info(f"✓ User {current_user_id} fully registered and logged")
            else:
                logger.warning(f"User {current_user_id} registered but sheets logging failed")
            
            welcome_msg = (
                f"Excellent, {updated_user.get('first_name')}! Your profile is complete.\n\n"
                f"I can help you with:\n\n"
                f"1. Teaching strategies and lesson planning\n"
                f"2. Classroom management\n"
                f"3. Assessment methods\n"
                f"4. Student engagement\n"
                f"5. Parent communication\n\n"
                f"What would you like help with today?"
            )
            
            return welcome_msg, True
        else:
            return "Please enter the class you teach.", False
    
    return "Something went wrong. Please try again.", False


def handle_reverification(user_id: str, phone_number: str, text: str, user: Dict, sheets_logger) -> Tuple[str, bool]:
    """Handle 90-day re-verification process"""
    
    step = user.get('registration_step', 4)
    
    # Start re-verification
    if step == 4:
        db.save_user(phone_number, registration_step=5)
        return (
            f"Hi {user.get('first_name', 'Teacher')}! It's been 90 days since we last verified your information.\n\n"
            f"Please confirm your email: {user.get('email', 'N/A')}\n\n"
            f"Reply with your current email or type 'same' if unchanged."
        ), False
    
    elif step == 5:
        # Verify email
        text = text.strip().lower()
        if text == 'same':
            db.save_user(phone_number, registration_step=6)
        elif '@' in text and '.' in text:
            db.save_user(phone_number, email=text, registration_step=6)
        else:
            return "Please enter a valid email or type 'same'.", False
        
        return (
            f"Is your location still {user.get('location', 'N/A')}?\n\n"
            f"Reply with your current location or type 'same'."
        ), False
    
    elif step == 6:
        # Verify location
        text = text.strip()
        if text.lower() == 'same':
            db.save_user(phone_number, registration_step=7)
        else:
            db.save_user(phone_number, location=text.title(), registration_step=7)
        
        return (
            f"Are you still teaching {user.get('class_taught', 'N/A')}?\n\n"
            f"Reply with your current class or type 'same'."
        ), False
    
    elif step == 7:
        # Complete reverification
        text = text.strip()
        if text.lower() == 'same':
            db.save_user(phone_number, registration_step=4)
        else:
            db.save_user(phone_number, class_taught=text.title(), registration_step=4)
        
        # Reset verification timestamp
        db.reset_verification(user_id)
        
        # Log reverification to sheets
        updated_user = db.get_user_by_phone(re.sub(r'\D', '', phone_number))
        sheets_logger.log_user_registration(updated_user)
        
        return (
            f"Thank you, {user.get('first_name')}! Your profile has been verified.\n\n"
            f"How can I help you today?"
        ), True
    
    return "Something went wrong. Please try again.", False


# Initialize components
db = DatabaseManager(CONFIG['DATABASE_URL'])
sheets_logger = SheetsLogger(CONFIG['APPS_SCRIPT_URL'], db)
linka_ai = LinkaAI(llm, embed_model, pinecone_index)

# User session locks to prevent data mixing - CRITICAL FOR CONCURRENT USERS
user_locks = {}
lock_manager_lock = threading.Lock()

def get_user_lock(user_id: str) -> threading.Lock:
    """Get or create a lock for specific user to prevent concurrent processing"""
    with lock_manager_lock:
        if user_id not in user_locks:
            user_locks[user_id] = threading.Lock()
        return user_locks[user_id]


def process_message(phone_number: str, chat_id: str, text_message: str) -> str:
    """Main message processing logic with strict user isolation"""
    phone = extract_phone_number(phone_number)
    
    # Get user by phone number (PRIMARY)
    user = db.get_user_by_phone(phone)
    
    # Fallback: try chat_id if phone lookup fails
    if not user:
        user = db.get_user_by_chat_id(chat_id)
        if user:
            db.save_user(phone, chat_id=chat_id, user_id=user['user_id'])
            logger.info(f"✓ User recovered via chat_id")
    
    # Get or assign user_id
    user_id = user['user_id'] if user else None
    
    # Acquire per-user lock to prevent data mixing
    if user_id:
        user_lock = get_user_lock(user_id)
        with user_lock:
            return _process_message_isolated(user_id, phone, chat_id, text_message, user)
    else:
        # New user registration (no lock needed yet)
        return _process_message_isolated(None, phone, chat_id, text_message, user)


def _process_message_isolated(user_id: Optional[str], phone: str, chat_id: str, 
                              text_message: str, user: Optional[Dict]) -> str:
    """Process message with user isolation guarantee - OPTIMIZED FOR SPEED"""
    try:
        if not text_message or len(text_message.strip()) < 2:
            return "Please send me a message or question. I'm LINKA AI, and I'm here to help!"
        
        text_message = text_message.strip()
        
        # Handle new user or incomplete registration
        if not user or not user.get('profile_complete') or user.get('registration_step', 0) < 4:
            response, completed = handle_registration(user_id, phone, chat_id, text_message, user, sheets_logger)
            return response
        
        user_id = user['user_id']
        
        # Check for 90-day memory expiry
        if db.check_memory_expiry(user_id):
            reverification_msg, completed = handle_reverification(
                user_id, phone, text_message, user, sheets_logger
            )
            if not completed:
                return reverification_msg
            # If completed, continue to normal processing
            user = db.get_user_by_id(user_id)
        
        # Update user activity (non-blocking)
        db.save_user(phone, chat_id=chat_id, user_id=user_id)
        
        logger.info(f"Processing: {user.get('first_name')} (ID: {user_id[:8]}...)")
        
        # Create session
        session_id = db.create_session(user_id)
        
        # Get history for THIS user ONLY
        history = db.get_history(user_id, limit=10)
        
        # Extract intent
        intent = linka_ai._extract_intent(text_message)
        db.save_message(user_id, phone, 'user', text_message, intent, session_id)
        
        # Generate response (with timeout protection)
        ai_response, response_intent = linka_ai.generate_response(
            text_message, user, history
        )
        
        # Save assistant response
        db.save_message(user_id, phone, 'assistant', ai_response, response_intent, session_id)
        
        # Update session activity
        db.update_session_activity(session_id)
        
        # Log to sheets (non-blocking, queued)
        sheets_logger.log_conversation({
            'user_id': user_id,
            'phone_number': phone,
            'user_name': user.get('first_name', 'Unknown'),
            'email': user.get('email', ''),
            'class_taught': user.get('class_taught', ''),
            'location': user.get('location', ''),
            'user_message': text_message[:1000],
            'bot_response': ai_response[:1000],
            'intent': intent,
            'response_length': len(ai_response),
            'session_id': session_id
        })
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Processing error for user {user_id}: {e}", exc_info=True)
        return "I'm LINKA AI, and I'm experiencing technical difficulties. Please try again in a moment."


def send_whatsapp_message(phone_number: str, message: str) -> bool:
    """Send message via Green API with retry logic"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.green-api.com/waInstance{CONFIG['GREEN_API_ID']}/sendMessage/{CONFIG['GREEN_API_TOKEN']}"
            
            response = requests.post(
                url,
                json={"chatId": phone_number, "message": message},
                headers={'Content-Type': 'application/json'},
                timeout=8
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Message sent to {phone_number[:10]}...")
                return True
            else:
                logger.warning(f"Send attempt {attempt + 1} failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Send error (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(0.3)
    
    return False


# Flask Routes

@app.route('/')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "LINKA AI - Schoolinka Initiative",
        "version": "5.1-optimized",
        "timestamp": datetime.now().isoformat(),
        "optimizations": [
            "Faster RAG retrieval with timeout",
            "Reduced context window",
            "Optimized registration flow",
            "Per-user concurrency locks"
        ]
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages with strict user isolation"""
    try:
        data = request.get_json()
        
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored"}), 200
        
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        if not chat_id:
            return jsonify({"status": "no_chat_id"}), 200
        
        # Ignore group messages
        if '@g.us' in chat_id:
            return jsonify({"status": "group_ignored"}), 200
        
        phone_number = extract_phone_number(chat_id)
        
        text_message = None
        
        if 'textMessageData' in message_data:
            text_message = message_data['textMessageData'].get('textMessage', '').strip()
        elif 'extendedTextMessageData' in message_data:
            text_message = message_data['extendedTextMessageData'].get('text', '').strip()
        
        if not text_message:
            send_whatsapp_message(
                chat_id,
                "I can only respond to text messages. Please type your question."
            )
            return jsonify({"status": "non_text"}), 200
        
        logger.info(f"Received from {phone_number[:10]}...: {text_message[:30]}...")
        
        # Process message asynchronously
        def process_and_respond():
            try:
                reply = process_message(phone_number, chat_id, text_message)
                send_whatsapp_message(chat_id, reply)
            except Exception as e:
                logger.error(f"Background processing error: {e}", exc_info=True)
                send_whatsapp_message(
                    chat_id,
                    "I encountered an error. Please try again."
                )
        
        thread = threading.Thread(target=process_and_respond, daemon=True)
        thread.start()
        
        return jsonify({"status": "processing"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error"}), 500


@app.route('/user/<user_id>', methods=['GET'])
def get_user_info(user_id):
    """Get user profile by user_id"""
    try:
        user = db.get_user_by_id(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        history = db.get_history(user_id, limit=10)
        
        return jsonify({
            "user": dict(user),
            "recent_messages": history,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Get user error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number', '2348012345678')
        chat_id = data.get('chat_id', f'{phone_number}@c.us')
        message = data.get('message', 'Hello')
        
        phone = extract_phone_number(phone_number)
        response = process_message(phone, chat_id, message)
        user = db.get_user_by_phone(phone)
        
        return jsonify({
            "response": response,
            "user": dict(user) if user else None,
            "phone_number": phone,
            "chatbot": "LINKA AI",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Test error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """System statistics"""
    try:
        with db.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute('SELECT COUNT(*) as total FROM users')
                total_users = cursor.fetchone()['total']
                
                cursor.execute('SELECT COUNT(*) as registered FROM users WHERE profile_complete = TRUE')
                registered = cursor.fetchone()['registered']
                
                cursor.execute('SELECT COUNT(*) as total FROM conversations')
                total_messages = cursor.fetchone()['total']
                
                cursor.execute('''
                    SELECT COUNT(*) as active FROM users 
                    WHERE last_interaction > NOW() - INTERVAL '7 days'
                ''')
                active_7d = cursor.fetchone()['active']
        
        return jsonify({
            "status": "success",
            "chatbot": "LINKA AI by Schoolinka",
            "stats": {
                "total_users": total_users,
                "registered_users": registered,
                "active_last_7_days": active_7d,
                "total_messages": total_messages
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health_check', methods=['GET'])
def health_check():
    """Detailed health check"""
    components = {
        "database": False,
        "pinecone": False,
        "google_ai": False,
        "green_api": False,
        "sheets_logger": False
    }
    
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT 1')
                components["database"] = True
    except:
        pass
    
    try:
        if pinecone_index:
            components["pinecone"] = True
    except:
        pass
    
    components["google_ai"] = llm is not None and embed_model is not None
    components["green_api"] = bool(CONFIG['GREEN_API_ID'] and CONFIG['GREEN_API_TOKEN'])
    components["sheets_logger"] = sheets_logger is not None
    
    all_ok = components["database"] and components["google_ai"]
    
    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "chatbot": "LINKA AI by Schoolinka",
        "components": components,
        "optimizations": {
            "rag_timeout": f"{CONFIG['RAG_TIMEOUT']}s",
            "llm_timeout": f"{CONFIG['LLM_TIMEOUT']}s",
            "max_tokens": CONFIG['MAX_TOKENS'],
            "max_history": CONFIG['MAX_HISTORY']
        },
        "timestamp": datetime.now().isoformat()
    }), 200 if all_ok else 503


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LINKA AI v5.1 - OPTIMIZED FOR SPEED")
    logger.info("=" * 70)
    logger.info(f"✓ Registration: Batch upload after all details collected")
    logger.info(f"✓ RAG Timeout: {CONFIG['RAG_TIMEOUT']}s")
    logger.info(f"✓ LLM Timeout: {CONFIG['LLM_TIMEOUT']}s")
    logger.info(f"✓ Max Tokens: {CONFIG['MAX_TOKENS']}")
    logger.info(f"✓ User Isolation: Per-user locks prevent data mixing")
    logger.info(f"✓ Sheets Logging: Queued, non-blocking")
    logger.info("=" * 70)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
