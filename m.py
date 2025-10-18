import os
import time
import re
from datetime import datetime, timedelta
import sqlite3
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
    'APPS_SCRIPT_URL': os.getenv("APPS_SCRIPT_URL", "https://script.google.com/macros/s/AKfycbzNmrXYRGTv3FoGsHkBxCYFVnMnop62fih4-T-PZP1jzknlh9oewfU2LHYYGi8r0IE/exec"),
    'DB_PATH': "ai_coach.db",
    'MAX_HISTORY': 25,
    'INDEX_NAME': 'coach',
    'INACTIVITY_TIMEOUT_DAYS': 30  # Re-confirm after 30 days
}

# Validate required keys
required_keys = ['PINECONE_API_KEY', 'GOOGLE_API_KEY', 'GREEN_API_ID', 'GREEN_API_TOKEN']
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
    """Handles all SQLite database operations with phone-number-based user isolation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    phone_number TEXT PRIMARY KEY,
                    chat_id TEXT UNIQUE,
                    first_name TEXT,
                    email TEXT,
                    location TEXT,
                    class_taught TEXT,
                    profile_complete BOOLEAN DEFAULT FALSE,
                    registration_step INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number TEXT NOT NULL,
                    message_type TEXT CHECK(message_type IN ('user', 'assistant')),
                    message_content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    intent TEXT,
                    FOREIGN KEY (phone_number) REFERENCES users (phone_number)
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_number ON conversations (phone_number)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_type ON conversations (phone_number, message_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone_number)')
            
            conn.commit()
            logger.info("✓ Database initialized with phone-number indexing")
    
    @contextmanager
    def get_conn(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_user_by_phone(self, phone_number: str) -> Optional[Dict]:
        """Get user by phone number (primary lookup method)"""
        phone = re.sub(r'\D', '', phone_number)
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE phone_number = ?', (phone,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_user_by_chat_id(self, chat_id: str) -> Optional[Dict]:
        """Get user by chat_id (secondary lookup for recovery)"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE chat_id = ?', (chat_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def save_user(self, phone_number: str, chat_id: str = None, **kwargs) -> bool:
        """Save or update user by phone number"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT phone_number FROM users WHERE phone_number = ?', (phone,))
                exists = cursor.fetchone()
                
                if exists:
                    # User exists, update fields
                    updates = []
                    values = []
                    
                    if chat_id:
                        updates.append('chat_id = ?')
                        values.append(chat_id)
                    
                    for k, v in kwargs.items():
                        updates.append(f'{k} = ?')
                        values.append(v)
                    
                    updates.append('last_interaction = CURRENT_TIMESTAMP')
                    updates.append('total_messages = total_messages + 1')
                    values.append(phone)
                    
                    update_str = ', '.join(updates)
                    cursor.execute(f'UPDATE users SET {update_str} WHERE phone_number = ?', values)
                else:
                    # New user
                    fields = ['phone_number']
                    values = [phone]
                    
                    if chat_id:
                        fields.append('chat_id')
                        values.append(chat_id)
                    
                    for k, v in kwargs.items():
                        fields.append(k)
                        values.append(v)
                    
                    fields.append('total_messages')
                    values.append(1)
                    
                    placeholders = ', '.join(['?' for _ in fields])
                    cursor.execute(
                        f'INSERT INTO users ({", ".join(fields)}) VALUES ({placeholders})', 
                        values
                    )
                
                conn.commit()
                logger.info(f"✓ User {phone} saved/updated")
                return True
        except Exception as e:
            logger.error(f"Error saving user {phone_number}: {e}")
            return False
    
    def save_message(self, phone_number: str, msg_type: str, content: str, intent: str = None):
        """Save conversation message by phone number"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (phone_number, message_type, message_content, intent)
                    VALUES (?, ?, ?, ?)
                ''', (phone, msg_type, content, intent))
                conn.commit()
                self._cleanup_history(phone)
        except Exception as e:
            logger.error(f"Error saving message for {phone_number}: {e}")
    
    def get_history(self, phone_number: str, limit: int = None) -> List[Dict]:
        """Get conversation history for specific phone number"""
        limit = limit or CONFIG['MAX_HISTORY']
        phone = re.sub(r'\D', '', phone_number)
        
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_type, message_content, timestamp, intent
                    FROM conversations 
                    WHERE phone_number = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (phone, limit))
                return [dict(row) for row in reversed(cursor.fetchall())]
        except Exception as e:
            logger.error(f"Error getting history for {phone_number}: {e}")
            return []
    
    def get_last_assistant_messages(self, phone_number: str, num_messages: int = 3) -> List[Dict]:
        """Get last N assistant messages for specific phone number"""
        phone = re.sub(r'\D', '', phone_number)
        
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_content, timestamp, intent
                    FROM conversations 
                    WHERE phone_number = ? AND message_type = 'assistant'
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (phone, num_messages))
                return [dict(row) for row in reversed(cursor.fetchall())]
        except Exception as e:
            logger.error(f"Error getting last assistant messages for {phone_number}: {e}")
            return []
    
    def _cleanup_history(self, phone_number: str):
        """Clean up old messages beyond max history for specific phone number"""
        phone = re.sub(r'\D', '', phone_number)
        
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM conversations 
                    WHERE phone_number = ? AND id NOT IN (
                        SELECT id FROM conversations 
                        WHERE phone_number = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    )
                ''', (phone, phone, CONFIG['MAX_HISTORY'] * 2))
                conn.commit()
        except Exception as e:
            logger.error(f"Cleanup error for {phone_number}: {e}")


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
        
        followup_instruction = ""
        if is_followup:
            if followup_type == 'list_expansion':
                followup_instruction = "\nFOLLOW-UP DETECTED: User is asking for lists/items. Provide a comprehensive numbered list with brief descriptions."
            elif followup_type == 'detailed_breakdown':
                followup_instruction = "\nFOLLOW-UP DETECTED: User wants more details. Expand with steps, examples, and detailed explanations."
else:
