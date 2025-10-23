import os
import time
import re
from datetime import datetime
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ai_coach.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
CONFIG = {
    'PINECONE_API_KEY': os.getenv("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg"),
    'GOOGLE_API_KEY': os.getenv("GOOGLE_API_KEY", "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw"),
    'GREEN_API_ID': os.getenv("GREEN_API_ID_INSTANCE", "7105328354"),
    'GREEN_API_TOKEN': os.getenv("GREEN_API_TOKEN", "2a33db828fe64c57a32debcca8f065cac2f901d270d04347a5"),
    'APPS_SCRIPT_URL': os.getenv("APPS_SCRIPT_URL", "https://script.google.com/macros/s/AKfycbwpz7zeTVd0IwEtAIxT80LCQu9_ifQ248_X_uqnqQwVZHQ-j-_9DuiHFAPQA3vO6tqM/exec"),
    'SPREADSHEET_ID': os.getenv("SPREADSHEET_ID", "1br8bPoH39yUrJtFPV3kesTcfnf7eD2TQnsZ0SLI5vlE"),
    'DATABASE_URL': os.getenv("DATABASE_URL"),  # Render provides this automatically
    'MAX_HISTORY': 25,
    'INDEX_NAME': 'coach'
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
        max_tokens=1500,
        timeout=10
    )
    logger.info("✓ Google AI services initialized")
    
except Exception as e:
    logger.error(f"✗ Critical initialization error: {e}")
    raise


def extract_phone_number(chat_id: str) -> str:
    """Extract phone number from WhatsApp chat_id (format: 234XXXXXXXXXX@c.us or similar)"""
    # Remove @c.us and any other suffixes
    phone = re.sub(r'@.*$', '', chat_id)
    # Remove any non-digit characters
    phone = re.sub(r'\D', '', phone)
    return phone or chat_id


class DatabaseManager:
    """Handles all PostgreSQL database operations with phone-number-based user isolation"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool = None
        self._init_connection_pool()
        self._init_db()
    
    def _init_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.database_url
            )
            logger.info("✓ PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    def _init_db(self):
        """Initialize database tables"""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    # Users table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS users (
                            phone_number VARCHAR(20) PRIMARY KEY,
                            chat_id VARCHAR(100) UNIQUE,
                            first_name VARCHAR(100),
                            email VARCHAR(255),
                            location VARCHAR(100),
                            class_taught VARCHAR(100),
                            profile_complete BOOLEAN DEFAULT FALSE,
                            registration_step INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            total_messages INTEGER DEFAULT 0,
                            logged_to_sheets BOOLEAN DEFAULT FALSE
                        )
                    ''')
                    
                    # Conversations table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS conversations (
                            id SERIAL PRIMARY KEY,
                            phone_number VARCHAR(20) NOT NULL,
                            message_type VARCHAR(10) CHECK(message_type IN ('user', 'assistant')),
                            message_content TEXT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            intent VARCHAR(50),
                            logged_to_sheets BOOLEAN DEFAULT FALSE,
                            FOREIGN KEY (phone_number) REFERENCES users (phone_number) ON DELETE CASCADE
                        )
                    ''')
                    
                    # Create indexes
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_number ON conversations (phone_number)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_type ON conversations (phone_number, message_type)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone_number)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sheets_log ON conversations (logged_to_sheets)')
                    
                    conn.commit()
                    logger.info("✓ PostgreSQL database initialized")
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
    
    def get_user_by_phone(self, phone_number: str) -> Optional[Dict]:
        """Get user by phone number (primary lookup method)"""
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
    
    def get_user_by_chat_id(self, chat_id: str) -> Optional[Dict]:
        """Get user by chat_id (secondary lookup for recovery)"""
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('SELECT * FROM users WHERE chat_id = %s', (chat_id,))
                    row = cursor.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user by chat_id: {e}")
            return None
    
    def save_user(self, phone_number: str, chat_id: str = None, **kwargs) -> bool:
        """Save or update user by phone number"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('SELECT phone_number FROM users WHERE phone_number = %s', (phone,))
                    exists = cursor.fetchone()
                    
                    if exists:
                        # User exists, update fields
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
                        values.append(phone)
                        
                        update_str = ', '.join(updates)
                        cursor.execute(f'UPDATE users SET {update_str} WHERE phone_number = %s', values)
                    else:
                        # New user
                        fields = ['phone_number']
                        values = [phone]
                        placeholders = ['%s']
                        
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
                    logger.info(f"✓ User {phone} saved/updated")
                    return True
        except Exception as e:
            logger.error(f"Error saving user {phone_number}: {e}")
            return False
    
    def mark_user_logged_to_sheets(self, phone_number: str):
        """Mark user as logged to Google Sheets"""
        phone = re.sub(r'\D', '', phone_number)
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('UPDATE users SET logged_to_sheets = TRUE WHERE phone_number = %s', (phone,))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error marking user logged: {e}")
    
    def save_message(self, phone_number: str, msg_type: str, content: str, intent: str = None):
        """Save conversation message by phone number"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        INSERT INTO conversations 
                        (phone_number, message_type, message_content, intent, logged_to_sheets)
                        VALUES (%s, %s, %s, %s, FALSE)
                    ''', (phone, msg_type, content, intent))
                    conn.commit()
                    self._cleanup_history(phone)
        except Exception as e:
            logger.error(f"Error saving message for {phone_number}: {e}")
    
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
                        SELECT c.*, u.first_name, u.class_taught, u.location
                        FROM conversations c
                        LEFT JOIN users u ON c.phone_number = u.phone_number
                        WHERE c.logged_to_sheets = FALSE
                        ORDER BY c.timestamp ASC
                        LIMIT %s
                    ''', (limit,))
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting unlogged messages: {e}")
            return []
    
    def get_history(self, phone_number: str, limit: int = None) -> List[Dict]:
        """Get conversation history for specific phone number"""
        limit = limit or CONFIG['MAX_HISTORY']
        phone = re.sub(r'\D', '', phone_number)
        
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('''
                        SELECT message_type, message_content, timestamp, intent
                        FROM conversations 
                        WHERE phone_number = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    ''', (phone, limit))
                    results = cursor.fetchall()
                    return [dict(row) for row in reversed(results)]
        except Exception as e:
            logger.error(f"Error getting history for {phone_number}: {e}")
            return []
    
    def get_last_assistant_messages(self, phone_number: str, num_messages: int = 3) -> List[Dict]:
        """Get last N assistant messages for specific phone number"""
        phone = re.sub(r'\D', '', phone_number)
        
        try:
            with self.get_conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute('''
                        SELECT message_content, timestamp, intent
                        FROM conversations 
                        WHERE phone_number = %s AND message_type = 'assistant'
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    ''', (phone, num_messages))
                    results = cursor.fetchall()
                    return [dict(row) for row in reversed(results)]
        except Exception as e:
            logger.error(f"Error getting last assistant messages for {phone_number}: {e}")
            return []
    
    def _cleanup_history(self, phone_number: str):
        """Clean up old messages beyond max history for specific phone number"""
        phone = re.sub(r'\D', '', phone_number)
        
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        DELETE FROM conversations 
                        WHERE phone_number = %s AND id NOT IN (
                            SELECT id FROM conversations 
                            WHERE phone_number = %s 
                            ORDER BY timestamp DESC 
                            LIMIT %s
                        )
                    ''', (phone, phone, CONFIG['MAX_HISTORY'] * 2))
                    conn.commit()
        except Exception as e:
            logger.error(f"Cleanup error for {phone_number}: {e}")
    
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
    
    def __init__(self, apps_script_url: str, db_manager: DatabaseManager):
        self.apps_script_url = apps_script_url
        self.db = db_manager
        self.queue = Queue()
        self.is_processing = False
        
        # Start background worker
        self._start_background_worker()
    
    def _start_background_worker(self):
        """Start background thread to process logging queue"""
        def worker():
            while True:
                try:
                    time.sleep(2)  # Process every 2 seconds
                    self._process_queue()
                except Exception as e:
                    logger.error(f"Sheets worker error: {e}")
                time.sleep(5)  # Wait before retry on error
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        logger.info("✓ Sheets logging worker started")
    
    def log_user_registration(self, user_data: Dict) -> bool:
        """Log user registration synchronously (blocking) for critical data"""
        if not self.apps_script_url:
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                payload = {
                    'type': 'user_registration',
                    'data': user_data,
                    'timestamp': datetime.now().isoformat()
                }
                
                response = requests.post(
                    self.apps_script_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.db.mark_user_logged_to_sheets(user_data['phone_number'])
                    logger.info(f"✓ User registration logged to sheets: {user_data['phone_number']}")
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
        
        self.is_processing = True
        processed = 0
        
        try:
            # Also process any unlogged messages from database
            unlogged = self.db.get_unlogged_messages(limit=20)
            
            for msg in unlogged:
                if processed >= 10:  # Limit batch size
                    break
                
                payload = {
                    'type': 'conversation',
                    'data': {
                        'phone_number': msg['phone_number'],
                        'user_name': msg.get('first_name', 'Unknown'),
                        'class_taught': msg.get('class_taught', ''),
                        'location': msg.get('location', ''),
                        'message_type': msg['message_type'],
                        'message_content': msg['message_content'][:500],
                        'intent': msg.get('intent', ''),
                        'timestamp': str(msg['timestamp'])
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
                        logger.info(f"✓ Message {msg['id']} logged to sheets")
                    else:
                        logger.warning(f"Failed to log message {msg['id']}: {response.status_code}")
                        break  # Stop batch on failure
                        
                except Exception as e:
                    logger.error(f"Error logging message {msg['id']}: {e}")
                    break  # Stop batch on error
                
                time.sleep(0.5)  # Rate limiting
            
            # Process queue items
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
                        logger.info(f"✓ Queue item logged to sheets")
                    else:
                        # Re-queue on failure
                        self.queue.put(item)
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing queue item: {e}")
                    break
                
                time.sleep(0.5)
            
            if processed > 0:
                logger.info(f"✓ Processed {processed} items to sheets")
                
        finally:
            self.is_processing = False


class AICoach:
    """AI Coach with RAG and context awareness"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        self.formatter = ResponseFormatter()
    
    def detect_followup_intent(self, message: str, history: List[Dict]) -> Tuple[bool, str]:
        """Detect if message is a follow-up to previous response"""
        msg_lower = message.lower()
        
        followup_indicators = [
            r'\b(explain|clarify|elaborate|more|details?)\b',
            r'\b(list|list\s+the|give\s+me|provide|show)\s+(all\s+)?(items?|functions?|examples?|steps?|points?)',
            r'\b(break\s+down|expand\s+on|go\s+deeper)\b',
            r'\b(what\s+about|how\s+about|tell\s+me\s+about|discuss)\b',
            r'\b(previous|last|earlier)\s+(response|answer|point|idea)',
            r'\b(you\s+mentioned|you\s+said)\b',
            r'\b(further|furthermore|additionally|also)\b',
        ]
        
        is_followup = any(re.search(pattern, msg_lower) for pattern in followup_indicators)
        
        followup_type = 'clarification'
        if any(re.search(p, msg_lower) for p in [r'\blist\b', r'\bitems?\b', r'\bfunctions?\b']):
            followup_type = 'list_expansion'
        elif any(re.search(p, msg_lower) for p in [r'\bexample\b', r'\bsteps?\b']):
            followup_type = 'detailed_breakdown'
        
        return is_followup, followup_type
    
    def get_rag_content(self, query: str, intent: str = None) -> Tuple[str, List[str]]:
        """Retrieve relevant content from Pinecone"""
        if not self.pinecone_index:
            return "", []
        
        try:
            enhanced_query = f"{intent} {query}" if intent else query
            query_embed = self.embed_model.embed_query(enhanced_query)
            
            results = self.pinecone_index.query(
                vector=query_embed,
                top_k=5,
                include_metadata=True,
                include_values=False
            )
            
            contents, sources = [], []
            for match in results.get('matches', []):
                score = match.get('score', 0)
                
                if score > 0.65:
                    text = match.get('metadata', {}).get('text', '')
                    source = match.get('metadata', {}).get('source', 'Knowledge Base')
                    
                    if text:
                        contents.append(text[:500])
                        sources.append(source)
            
            if contents:
                logger.info(f"✓ Retrieved {len(contents)} relevant documents")
                return '\n\n'.join(contents), sources
            
            return "", []
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return "", []
    
    def generate_response(self, message: str, user_profile: Dict, 
                         history: List[Dict] = None) -> Tuple[str, str]:
        """Generate formatted AI response"""
        try:
            intent = self._extract_intent(message)
            
            is_followup, followup_type = self.detect_followup_intent(message, history or [])
            
            rag_content = ""
            sources = []
            if not is_followup or followup_type == 'clarification':
                rag_content, sources = self.get_rag_content(message, intent)
            
            context = ""
            assistant_context = ""
            if history:
                recent = history[-10:]
                context_parts = []
                for msg in recent:
                    role = "Teacher" if msg['message_type'] == 'user' else "AI Coach"
                    content = msg['message_content'][:150]
                    context_parts.append(f"{role}: {content}")
                context = "\n".join(context_parts)
            
            if is_followup:
                last_responses = self._get_last_bot_responses(history, num=3)
                if last_responses:
                    assistant_context = f"PREVIOUS BOT RESPONSES:\n{last_responses}\n\n"
            
            system_prompt = self._build_system_prompt(
                user_profile, rag_content, context, intent, message, 
                is_followup, followup_type, assistant_context
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
    
    def _get_last_bot_responses(self, history: List[Dict], num: int = 3) -> str:
        """Extract last N bot responses from history"""
        if not history:
            return ""
        
        bot_responses = [
            msg['message_content'][:200] 
            for msg in history 
            if msg['message_type'] == 'assistant'
        ][-num:]
        
        if bot_responses:
            return "\n\n".join([f"- {resp}" for resp in bot_responses])
        return ""
    
    def _build_system_prompt(self, user_profile: Dict, rag_content: str, 
                            context: str, intent: str, query: str,
                            is_followup: bool = False, followup_type: str = None,
                            assistant_context: str = "") -> str:
        """Build comprehensive system prompt"""
        
        name = user_profile.get('first_name', 'Teacher')
        class_info = user_profile.get('class_taught', 'their class')
        location = user_profile.get('location', 'Nigeria')
        total_msgs = user_profile.get('total_messages', 0)
        
        greeting = ""
        if total_msgs > 3:
            greeting = f"(This is a returning user - greet them warmly!)"
        
        followup_instruction = ""
        if is_followup:
            if followup_type == 'list_expansion':
                followup_instruction = "\nFOLLOW-UP DETECTED: User is asking for lists/items. Provide a comprehensive numbered list with brief descriptions."
            elif followup_type == 'detailed_breakdown':
                followup_instruction = "\nFOLLOW-UP DETECTED: User wants more details. Expand with steps, examples, and detailed explanations."
            else:
                followup_instruction = "\nFOLLOW-UP DETECTED: User seeks clarification. Reference your previous response and clarify."
        
        prompt = f"""You are AI Coach by Schoolinka, a friendly teaching assistant for Nigerian teachers
        TEACHER PROFILE:
        - Name: {name}
        - Teaching: {class_info}
        - Location: {location}
        - Messages exchanged: {total_msgs}
        {greeting}

        CURRENT QUERY: {query}
        DETECTED INTENT: {intent}
        {followup_instruction}

        {assistant_context}

        CORE GUIDELINES:
        AI Coach is by Schoolinka. Schoolinka was founded by Oluwaseun Kayode. It's an integrated platform offering training courses, certifications, and teaching resources for educators, plus a job board for teachers.

        CRITICAL FOLLOW-UP HANDLING:
        - Reference previous responses explicitly when asked
        - Provide ALL items as numbered lists when asked to list
        - Expand 2-3x when asked to elaborate
        - Maintain context from previous exchanges
        - Acknowledge specific items from prior responses

        RESPONSE GUIDELINES:
        1. RESPONSE STYLE:
           - Be warm, conversational, encouraging
           - Don't overuse the user's name
           - Provide practical, Nigeria-specific advice
           - Be detailed and thorough
           - Show empathy and understanding

        2. FORMATTING RULES:
           - Use numbers (1. 2. 3.) for all lists
           - Start each item on new line
           - Keep paragraphs short (2-3 sentences max)
           - Add line breaks between sections
           - NO asterisks, bullets, or markdown

        3. RESPONSE LENGTH:
           - Simple questions: 4-6 sentences
           - How-to: Detailed numbered steps
           - Complex topics: 3-4 paragraphs
           - Follow-ups: More detail than initial response

        4. NIGERIAN CONTEXT:
           - Consider large class sizes (30-60 students)
           - Address limited resources
           - Account for power supply challenges
           - Reference local curriculum
           - Use relevant Nigerian examples"""
        
        if rag_content:
            prompt += f"\n\nRELEVANT KNOWLEDGE BASE:\n{rag_content}\n(Use to enhance accuracy)"
        
        if context:
            prompt += f"\n\nRECENT CONVERSATION:\n{context}\n(Reference naturally when relevant)"
        
        prompt += "\n\nProvide a helpful, well-formatted response addressing the teacher's question."
        
        return prompt
    
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Extract intent from message"""
        msg = message.lower()
        
        intents = {
            'teaching_strategy': ['teach', 'strategy', 'method', 'lesson', 'explain', 'introduce', 'activity', 'engage', 'learn', 'understand'],
            'classroom_management': ['discipline', 'behavior', 'manage', 'control', 'disruptive', 'noise', 'attention', 'class control'],
            'assessment': ['assess', 'evaluate', 'grade', 'test', 'exam', 'mark', 'feedback', 'progress', 'score'],
            'wellbeing': ['stress', 'tired', 'overwhelmed', 'burnout', 'exhausted', 'frustrated', 'difficult', 'help me'],
            'curriculum': ['curriculum', 'syllabus', 'topic', 'subject', 'scheme of work', 'lesson plan'],
            'parent_communication': ['parent', 'guardian', 'meeting', 'report', 'communicate'],
            'resources': ['resource', 'material', 'tool', 'equipment', 'aid', 'need'],
            'compliments': ['Thank', 'Okay', 'Good']
        }
        
        for intent_name, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent_name
        
        return 'general'
    
    @staticmethod
    def _get_fallback_response(intent: str) -> str:
        """Fallback response on error"""
        return "I'm experiencing a brief technical issue. Please try your question again, and I'll help you."


# Registration step templates
REGISTRATION_STEPS = {
    0: {
        "message": "Hello! I'm AI Coach by Schoolinka. I'm here to support you with teaching strategies, classroom management, and professional development.\n\nWhat's your first name?",
        "field": None
    },
    1: {
        "message": "Nice to meet you, {first_name}!\n\nWhat's your email address?",
        "field": "first_name"
    },
    2: {
        "message": "Thanks, {first_name}!\n\nWhich city or state are you in?",
        "field": "email"
    },
    3: {
        "message": "Thanks for sharing!\n\nWhich class do you teach?",
        "field": "location"
    }
}


def handle_registration(phone_number: str, chat_id: str, text: str, user: Optional[Dict], sheets_logger) -> str:
    """Handle multi-step user registration"""
    
    if not user:
        # New user, start registration
        db.save_user(phone_number, chat_id=chat_id, profile_complete=False, registration_step=0)
        return REGISTRATION_STEPS[0]["message"]
    
    step = user.get('registration_step', 0)
    
    if step >= 4:
        return "Your profile is complete! How can I help you today?"
    
    text = text.strip()
    if not text or len(text) < 2:
        current_template = REGISTRATION_STEPS[step]
        return f"Please provide your {current_template['field'] or 'details'} to continue."
    
    # Validate and save based on current step
    if step == 0:
        clean_name = re.sub(r'[^a-zA-Z\s]', '', text)
        if len(clean_name) > 1:
            db.save_user(phone_number, chat_id=chat_id, first_name=clean_name.title(), registration_step=1)
            next_msg = REGISTRATION_STEPS[1]["message"].format(first_name=clean_name.title())
            return next_msg
        else:
            return "Please enter a valid name."
    
    elif step == 1:
        if '@' in text and '.' in text and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}, text.lower()):
            db.save_user(phone_number, email=text.lower(), registration_step=2)
            name = user.get('first_name', 'Teacher')
            next_msg = REGISTRATION_STEPS[2]["message"].format(first_name=name)
            return next_msg
        else:
            return "Please enter a valid email address (e.g., name@example.com)."
    
    elif step == 2:
        if len(text) > 2:
            db.save_user(phone_number, location=text.title(), registration_step=3)
            name = user.get('first_name', 'Teacher')
            next_msg = REGISTRATION_STEPS[3]["message"].format(first_name=name)
            return next_msg
        else:
            return "Please enter a valid location."
    
    elif step == 3:
        if len(text) > 1:
            db.save_user(phone_number, class_taught=text.title(), profile_complete=True, registration_step=4)
            
            # Log to sheets SYNCHRONOUSLY (blocking) for critical registration data
            user_data = {
                'phone_number': phone_number,
                'name': user.get('first_name', ''),
                'email': user.get('email', ''),
                'location': user.get('location', ''),
                'class': text.title(),
                'status': 'Registered',
                'registration_date': datetime.now().isoformat()
            }
            
            sheets_logger.log_user_registration(user_data)
            
            welcome_msg = (
                f"Excellent! Your profile is complete.\n\n"
                f"I can help you with:\n\n"
                f"1. Teaching strategies and lesson planning\n"
                f"2. Classroom management techniques\n"
                f"3. Assessment and evaluation methods\n"
                f"4. Student engagement activities\n"
                f"5. Parent communication strategies\n"
                f"6. Professional development tips\n\n"
                f"What would you like help with today?"
            )
            
            return welcome_msg
        else:
            return "Please enter the class you teach."
    
    return "Something went wrong. Please try again."


# Initialize components
db = DatabaseManager(CONFIG['DATABASE_URL'])
sheets_logger = SheetsLogger(CONFIG['APPS_SCRIPT_URL'], db)
ai_coach = AICoach(llm, embed_model, pinecone_index)


def process_message(phone_number: str, chat_id: str, text_message: str) -> str:
    """Main message processing logic with phone-number isolation"""
    try:
        if not text_message or len(text_message.strip()) < 2:
            return "Please send me a message or question. I'm here to help!"
        
        text_message = text_message.strip()
        phone = extract_phone_number(phone_number)
        
        # Get user by phone number (primary) or create new entry
        user = db.get_user_by_phone(phone)
        
        # If no user found by phone, try chat_id as fallback
        if not user:
            user = db.get_user_by_chat_id(chat_id)
            if user:
                # Update phone number in existing record
                db.save_user(phone, chat_id=chat_id)
                logger.info(f"✓ Recovered user by chat_id, updated phone to {phone}")
        
        # Handle registration - only if user doesn't exist OR profile is incomplete
        if not user:
            # New user, start registration
            return handle_registration(phone, chat_id, text_message, None, sheets_logger)
        elif not user.get('profile_complete') or user.get('registration_step', 0) < 4:
            # Existing user with incomplete profile, continue registration
            return handle_registration(phone, chat_id, text_message, user, sheets_logger)
        
        # User is fully registered - proceed to conversation
        
        # Update user activity and chat_id (in case user messaged from new device)
        db.save_user(phone, chat_id=chat_id)
        
        logger.info(f"✓ Recognized returning user: {user.get('first_name', 'Unknown')} (phone: {phone})")
        
        # Get history for this specific phone number
        history = db.get_history(phone, limit=15)
        
        # Extract intent
        intent = ai_coach._extract_intent(text_message)
        db.save_message(phone, 'user', text_message, intent)
        
        logger.info(f"Processing message from {user.get('first_name', 'Unknown')} (phone: {phone}) - Intent: {intent}")
        
        # Generate response
        ai_response, response_intent = ai_coach.generate_response(
            text_message, user, history
        )
        
        # Save assistant response
        db.save_message(phone, 'assistant', ai_response, response_intent)
        
        # Log to sheets in background (non-blocking via queue)
        sheets_logger.log_conversation({
            'phone_number': phone,
            'user_name': user.get('first_name', 'Unknown'),
            'class_taught': user.get('class_taught', ''),
            'location': user.get('location', ''),
            'user_message': text_message[:500],
            'bot_response': ai_response[:500],
            'intent': intent,
            'response_length': len(ai_response)
        })
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Message processing error for {phone_number}: {e}", exc_info=True)
        return "I'm experiencing technical difficulties. Please try again in a moment."


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
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Message sent to {phone_number}")
                return True
            else:
                logger.warning(f"Send attempt {attempt + 1} failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Send error (attempt {attempt + 1}) for {phone_number}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(0.5)
    
    return False


# Flask Routes

@app.route('/')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI Coach - Schoolinka",
        "version": "4.0-postgresql",
        "timestamp": datetime.now().isoformat(),
        "pinecone": "connected" if pinecone_index else "not available",
        "database": "PostgreSQL (Render)",
        "sheets_logger": "active",
        "user_management": "phone-number-based with PostgreSQL persistence"
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages with phone-number isolation"""
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
                "I can only respond to text messages right now. Please type your question."
            )
            return jsonify({"status": "non_text"}), 200
        
        logger.info(f"Received from {phone_number}: {text_message[:50]}...")
        
        # Process message asynchronously per phone number to prevent cross-user data mixing
        def process_and_respond():
            try:
                reply = process_message(phone_number, chat_id, text_message)
                send_whatsapp_message(chat_id, reply)
            except Exception as e:
                logger.error(f"Background processing error for {phone_number}: {e}", exc_info=True)
                send_whatsapp_message(
                    chat_id,
                    "Sorry, I encountered an error. Please try again."
                )
        
        # Use daemon thread for immediate return to webhook
        thread = threading.Thread(target=process_and_respond, daemon=True)
        thread.start()
        
        return jsonify({"status": "processing"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error"}), 500


@app.route('/user/<phone_number>', methods=['GET'])
def get_user_info(phone_number):
    """Get user profile by phone number - returns data only for requested phone"""
    try:
        phone = extract_phone_number(phone_number)
        user = db.get_user_by_phone(phone)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        history = db.get_history(phone, limit=10)
        
        return jsonify({
            "user": dict(user),
            "recent_messages": history,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Get user error for {phone_number}: {e}")
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
                
                cursor.execute('''
                    SELECT COUNT(*) as unlogged FROM conversations 
                    WHERE logged_to_sheets = FALSE
                ''')
                unlogged = cursor.fetchone()['unlogged']
                
                cursor.execute('''
                    SELECT intent, COUNT(*) as count 
                    FROM conversations 
                    WHERE intent IS NOT NULL 
                    GROUP BY intent 
                    ORDER BY count DESC 
                    LIMIT 5
                ''')
                top_intents = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({
            "status": "success",
            "stats": {
                "total_users": total_users,
                "registered_users": registered,
                "active_last_7_days": active_7d,
                "total_messages": total_messages,
                "unlogged_messages": unlogged,
                "top_intents": top_intents
            },
            "database": "PostgreSQL (Render)",
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
        "apps_script": False,
        "sheets_logger": False
    }
    
    vector_count = 0
    
    try:
        with db.get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute('SELECT 1')
                components["database"] = True
    except:
        pass
    
    try:
        if pinecone_index:
            stats = pinecone_index.describe_index_stats()
            components["pinecone"] = True
            vector_count = stats.get('total_vector_count', 0)
    except:
        pass
    
    components["google_ai"] = llm is not None and embed_model is not None
    components["green_api"] = bool(CONFIG['GREEN_API_ID'] and CONFIG['GREEN_API_TOKEN'])
    components["apps_script"] = bool(CONFIG['APPS_SCRIPT_URL'])
    components["sheets_logger"] = sheets_logger is not None
    
    all_ok = components["database"] and components["google_ai"]
    
    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "components": components,
        "config": {
            "index_name": CONFIG['INDEX_NAME'],
            "max_history": CONFIG['MAX_HISTORY'],
            "vector_count": vector_count if components["pinecone"] else 0,
            "database": "PostgreSQL (Render)",
            "user_identification": "phone-number-based with PostgreSQL persistence"
        },
        "timestamp": datetime.now().isoformat()
    }), 200 if all_ok else 503


@app.route('/test_rag', methods=['POST'])
def test_rag():
    """Test RAG functionality"""
    try:
        data = request.get_json()
        query = data.get('query', 'teaching strategies')
        
        if not pinecone_index:
            return jsonify({
                "error": "Pinecone index not available",
                "status": "failed"
            }), 503
        
        rag_content, sources = ai_coach.get_rag_content(query)
        
        return jsonify({
            "status": "success",
            "query": query,
            "rag_content": rag_content,
            "sources": sources,
            "num_sources": len(sources),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"RAG test error: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


@app.route('/retry_sheets_logging', methods=['POST'])
def retry_sheets_logging():
    """Manually trigger retry of failed sheets logging"""
    try:
        unlogged = db.get_unlogged_messages(limit=50)
        
        return jsonify({
            "status": "triggered",
            "unlogged_count": len(unlogged),
            "message": "Background worker will process these messages",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Retry trigger error: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("AI COACH - SCHOOLINKA v4.0 (POSTGRESQL)")
    logger.info("=" * 70)
    logger.info(f"Database: PostgreSQL (Render)")
    logger.info(f"User Identification: Phone Number (Primary) + Chat ID (Fallback)")
    logger.info(f"Pinecone Index: {CONFIG['INDEX_NAME']} - {'Connected' if pinecone_index else 'Not Available'}")
    logger.info(f"Google AI: {'Initialized' if llm else 'Failed'}")
    logger.info(f"Apps Script: {'Configured' if CONFIG['APPS_SCRIPT_URL'] else 'Not Configured'}")
    logger.info(f"Sheets Logger: {'Active' if sheets_logger else 'Inactive'}")
    logger.info(f"Multi-step Registration: ENABLED")
    logger.info(f"Concurrent User Handling: ENABLED")
    logger.info(f"Data Persistence: ENABLED (PostgreSQL)")
    logger.info(f"Real-time Sheets Logging: ENABLED (Queue-based)")
    logger.info(f"Data Isolation: PHONE-NUMBER BASED")
    
    if pinecone_index:
        try:
            stats = pinecone_index.describe_index_stats()
            logger.info(f"Vector Count: {stats.get('total_vector_count', 0)}")
        except:
            logger.warning("Could not retrieve vector count")
    
    logger.info("=" * 70)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
