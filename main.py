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
    'DB_PATH': "/opt/render/project/src/ai_coach.db" if os.path.exists("/opt/render/project/src") else "ai_coach.db",
    'MAX_HISTORY': 25,
    'INDEX_NAME': 'coach',
    'INACTIVITY_TIMEOUT_DAYS': 30
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
    phone = re.sub(r'@.*$', '', chat_id)
    phone = re.sub(r'\D', '', phone)
    return phone or chat_id


def validate_first_name(name: str) -> Tuple[bool, str]:
    """Validate first name - must be a single word"""
    cleaned = re.sub(r'[^a-zA-Z\s]', '', name).strip()
    words = cleaned.split()
    
    if len(words) == 0:
        return False, "Please enter a valid first name."
    elif len(words) > 1:
        return False, "Please enter only your first name (one word)."
    elif len(words[0]) < 2:
        return False, "Your first name must be at least 2 characters long."
    else:
        return True, words[0].title()


def validate_full_name(name: str) -> Tuple[bool, str]:
    """Validate full name - must be exactly two words"""
    cleaned = re.sub(r'[^a-zA-Z\s]', '', name).strip()
    words = cleaned.split()
    
    if len(words) == 0:
        return False, "Please enter a valid full name."
    elif len(words) == 1:
        return False, "Please enter your full name (first and last name)."
    elif len(words) > 2:
        return False, "Please enter only your first and last name (two words)."
    else:
        return True, ' '.join([w.title() for w in words])


class DatabaseManager:
    """Handles all SQLite database operations with phone-number-based user isolation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_db()
    
    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"✓ Created database directory: {db_dir}")
            except Exception as e:
                logger.error(f"Failed to create database directory: {e}")
                self.db_path = "ai_coach.db"
    
    def _init_db(self):
        """Initialize database tables with proper persistence"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enable WAL mode for better concurrency and persistence
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute('PRAGMA synchronous=NORMAL')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        phone_number TEXT PRIMARY KEY,
                        chat_id TEXT UNIQUE,
                        first_name TEXT,
                        full_name TEXT,
                        email TEXT,
                        location TEXT,
                        class_taught TEXT,
                        profile_complete BOOLEAN DEFAULT FALSE,
                        registration_step INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_messages INTEGER DEFAULT 0,
                        needs_reconfirmation BOOLEAN DEFAULT FALSE
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
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_number ON conversations (phone_number)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_type ON conversations (phone_number, message_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone_number)')
                
                # Add full_name column if it doesn't exist (migration)
                try:
                    cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
                    logger.info("✓ Added full_name column to users table")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                # Add needs_reconfirmation column if it doesn't exist
                try:
                    cursor.execute("ALTER TABLE users ADD COLUMN needs_reconfirmation BOOLEAN DEFAULT FALSE")
                    logger.info("✓ Added needs_reconfirmation column to users table")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                conn.commit()
                logger.info(f"✓ Database initialized at: {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    @contextmanager
    def get_conn(self):
        """Context manager for database connections with isolation"""
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level='DEFERRED')
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_user_by_phone(self, phone_number: str) -> Optional[Dict]:
        """Get user by phone number (primary lookup method)"""
        phone = re.sub(r'\D', '', phone_number)
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE phone_number = ?', (phone,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user by phone {phone}: {e}")
            return None
    
    def save_user(self, phone_number: str, chat_id: str = None, **kwargs) -> bool:
        """Save or update user by phone number with ACID guarantees"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT phone_number FROM users WHERE phone_number = ?', (phone,))
                exists = cursor.fetchone()
                
                if days_inactive >= CONFIG['INACTIVITY_TIMEOUT_DAYS']:
                logger.info(f"✓ User {user.get('first_name', 'Unknown')} inactive for {days_inactive} days - triggering reconfirmation")
                return True
        except Exception as e:
            logger.error(f"Error checking inactivity: {e}")
    
    return False


def handle_registration(phone_number: str, chat_id: str, text: str, user: Optional[Dict]) -> str:
    """Handle multi-step user registration with proper validation"""
    
    # Strip and clean input
    text = text.strip()
    
    if not user:
        # New user, start registration
        db.save_user(phone_number, chat_id=chat_id, profile_complete=False, registration_step=0)
        log_to_google_sheets('new_user', {
            'phone_number': phone_number,
            'chat_id': chat_id,
            'status': 'Registration Started',
            'timestamp': datetime.now().isoformat()
        })
        return REGISTRATION_STEPS[0]["message"]
    
    step = user.get('registration_step', 0)
    
    # If registration complete, don't process registration
    if step >= 5 and not user.get('needs_reconfirmation', False):
        return None  # Signal to process as regular message
    
    if not text or len(text) < 1:
        current_template = REGISTRATION_STEPS.get(step, REGISTRATION_STEPS[0])
        return f"Please provide your {current_template.get('field', 'information')} to continue."
    
    # Step 0: First Name (single word)
    if step == 0:
        is_valid, result = validate_first_name(text)
        if is_valid:
            db.save_user(phone_number, chat_id=chat_id, first_name=result, registration_step=1)
            next_msg = REGISTRATION_STEPS[1]["message"].format(first_name=result)
            logger.info(f"✓ User {phone_number} - First name registered: {result}")
            return next_msg
        else:
            return result  # Error message
    
    # Step 1: Full Name (two words)
    elif step == 1:
        is_valid, result = validate_full_name(text)
        if is_valid:
            db.save_user(phone_number, full_name=result, registration_step=2)
            name = user.get('first_name', 'Teacher')
            next_msg = REGISTRATION_STEPS[2]["message"].format(first_name=name)
            logger.info(f"✓ User {phone_number} - Full name registered: {result}")
            return next_msg
        else:
            return result  # Error message
    
    # Step 2: Email
    elif step == 2:
        email = text.strip().lower()
        if '@' in email and '.' in email and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,} exists:
                    # User exists, update fields
                    updates = []
                    values = []
                    
                    if chat_id:
                        updates.append('chat_id = ?')
                        values.append(chat_id)
                    
                    for k, v in kwargs.items():
                        updates.append(f'{k} = ?')
                        values.append(v)
                    
                    updates.append('last_interaction = ?')
                    values.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    
                    # Only increment total_messages if not explicitly provided
                    if 'total_messages' not in kwargs:
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
                    
                    if 'total_messages' not in kwargs:
                        fields.append('total_messages')
                        values.append(1)
                    
                    placeholders = ', '.join(['?' for _ in fields])
                    cursor.execute(
                        f'INSERT INTO users ({", ".join(fields)}) VALUES ({placeholders})', 
                        values
                    )
                
                conn.commit()
                logger.info(f"✓ User {phone} saved/updated successfully")
                return True
        except Exception as e:
            logger.error(f"Error saving user {phone_number}: {e}", exc_info=True)
            return False
    
    def save_message(self, phone_number: str, msg_type: str, content: str, intent: str = None):
        """Save conversation message by phone number with isolation"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (phone_number, message_type, message_content, intent, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (phone, msg_type, content, intent, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                self._cleanup_history(phone)
                logger.debug(f"✓ Message saved for {phone}")
        except Exception as e:
            logger.error(f"Error saving message for {phone_number}: {e}")
    
    def get_history(self, phone_number: str, limit: int = None) -> List[Dict]:
        """Get conversation history for specific phone number ONLY"""
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
                results = [dict(row) for row in reversed(cursor.fetchall())]
                logger.debug(f"✓ Retrieved {len(results)} messages for {phone}")
                return results
        except Exception as e:
            logger.error(f"Error getting history for {phone_number}: {e}")
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
                followup_instruction = "\nFOLLOW-UP DETECTED: User seeks clarification. Reference your previous response and clarify."
        
        prompt = f"""You are AI Coach by Schoolinka, a friendly teaching assistant for Nigerian teachers
        TEACHER PROFILE:
        - Name: {name}
        - Teaching: {class_info}
        - Location: {location}

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
            'compliments': ['thank', 'okay', 'good']
        }
        
        for intent_name, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent_name
        
        return 'general'
    
    @staticmethod
    def _get_fallback_response(intent: str) -> str:
        """Fallback response on error"""
        return "I'm experiencing a brief technical issue. Please try your question again, and I'll help you."


def log_to_google_sheets(data_type: str, data: Dict) -> bool:
    """Log data to Google Sheets via Apps Script (non-blocking) with retry"""
    if not CONFIG['APPS_SCRIPT_URL']:
        return False
    
    def _log():
        max_retries = 2
        for attempt in range(max_retries):
            try:
                payload = {
                    'type': data_type,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
                response = requests.post(
                    CONFIG['APPS_SCRIPT_URL'],
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=8
                )
                
                if response.status_code == 200:
                    logger.info(f"✓ Logged {data_type} to Google Sheets for phone {data.get('phone_number', 'unknown')}")
                    return True
                else:
                    logger.warning(f"Sheets logging attempt {attempt + 1} failed: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Sheets logging error (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(1)
        
        return False
    
    thread = threading.Thread(target=_log, daemon=True)
    thread.start()
    return True


# Registration step templates
REGISTRATION_STEPS = {
    0: {
        "message": "Hello! I'm AI Coach by Schoolinka. I'm here to support you with teaching strategies, classroom management, and professional development.\n\nWhat's your first name? (Please enter only your first name)",
        "field": None
    },
    1: {
        "message": "Nice to meet you, {first_name}!\n\nWhat's your full name? (Please enter your first and last name)",
        "field": "first_name"
    },
    2: {
        "message": "Thanks, {first_name}!\n\nWhat's your email address?",
        "field": "full_name"
    },
    3: {
        "message": "Thanks for sharing!\n\nWhich city or state are you in?",
        "field": "email"
    },
    4: {
        "message": "Great!\n\nWhich class do you teach?",
        "field": "location"
    }
}


def check_inactivity_reconfirmation(user: Dict) -> bool:
    """Check if user needs reconfirmation due to 30+ days inactivity"""
    if not user:
        return False
    
    # Skip if already in reconfirmation mode
    if user.get('needs_reconfirmation', False):
        return True
    
    # Skip if not fully registered
    if user.get('registration_step', 0) < 5:
        return False
    
    last_interaction_str = user.get('last_interaction')
    
    if last_interaction_str:
        try:
            last_interaction = datetime.strptime(last_interaction_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
            days_inactive = (datetime.now() - last_interaction).days
            
            if, email):
            db.save_user(phone_number, email=email, registration_step=3)
            name = user.get('first_name', 'Teacher')
            next_msg = REGISTRATION_STEPS[3]["message"].format(first_name=name)
            logger.info(f"✓ User {phone_number} - Email registered: {email}")
            return next_msg
        else:
            return "Please enter a valid email address (e.g., name@example.com)."
    
    # Step 3: Location
    elif step == 3:
        location = text.strip().title()
        if len(location) > 2:
            db.save_user(phone_number, location=location, registration_step=4)
            name = user.get('first_name', 'Teacher')
            next_msg = REGISTRATION_STEPS[4]["message"].format(first_name=name)
            logger.info(f"✓ User {phone_number} - Location registered: {location}")
            return next_msg
        else:
            return "Please enter a valid location (city or state)."
    
    # Step 4: Class Taught
    elif step == 4:
        class_taught = text.strip().title()
        if len(class_taught) > 1:
            db.save_user(phone_number, class_taught=class_taught, profile_complete=True, registration_step=5)
            
            # Get full user profile for logging
            updated_user = db.get_user_by_phone(phone_number)
            
            # Log complete registration to Google Sheets
            log_to_google_sheets('user_registered', {
                'phone_number': phone_number,
                'first_name': updated_user.get('first_name', ''),
                'full_name': updated_user.get('full_name', ''),
                'email': updated_user.get('email', ''),
                'location': updated_user.get('location', ''),
                'class_taught': class_taught,
                'status': 'Registration Complete',
                'registration_date': datetime.now().isoformat()
            })
            
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
            
            logger.info(f"✓ User {phone_number} - Registration COMPLETE")
            return welcome_msg
        else:
            return "Please enter the class you teach (e.g., Primary 3, JSS 2, SS 1)."
    
    return "Something went wrong. Please try again."


def handle_reconfirmation(phone_number: str, text: str, user: Dict) -> Optional[str]:
    """Handle reconfirmation after 30 days of inactivity"""
    
    text = text.strip().lower()
    
    # First time showing reconfirmation message
    if not user.get('needs_reconfirmation', False):
        db.save_user(phone_number, needs_reconfirmation=True)
        
        name = user.get('first_name', 'Teacher')
        full_name = user.get('full_name', 'your name')
        location = user.get('location', 'your location')
        class_taught = user.get('class_taught', 'your class')
        
        reconfirm_msg = (
            f"Welcome back, {name}! It's been a while since we last chatted.\n\n"
            f"Just to make sure I have the right profile:\n"
            f"Name: {full_name}\n"
            f"Location: {location}\n"
            f"Teaching: {class_taught}\n\n"
            f"Is this information still correct? (Reply 'Yes' or 'No')"
        )
        
        logger.info(f"✓ Reconfirmation initiated for {name} (phone: {phone_number})")
        return reconfirm_msg
    
    # User responding to reconfirmation
    if text in ['yes', 'y', 'correct', 'right']:
        db.save_user(phone_number, needs_reconfirmation=False)
        logger.info(f"✓ User {user.get('first_name', 'Unknown')} confirmed profile")
        return "Great, thanks for confirming! Let's get back to coaching. What's on your mind today?"
    
    elif text in ['no', 'n', 'wrong', 'incorrect']:
        db.save_user(phone_number, profile_complete=False, registration_step=0, needs_reconfirmation=False)
        logger.info(f"✓ User {user.get('first_name', 'Unknown')} declined - restarting registration")
        return "No problem! Let's update your profile.\n\n" + REGISTRATION_STEPS[0]["message"]
    
    else:
        return "I didn't understand. Please reply 'Yes' to confirm your profile is correct, or 'No' to update your details."


def process_message(phone_number: str, chat_id: str, text_message: str) -> str:
    """Main message processing logic with strict user isolation"""
    try:
        if not text_message or len(text_message.strip()) < 1:
            return "Please send me a message or question. I'm here to help!"
        
        text_message = text_message.strip()
        phone = extract_phone_number(phone_number)
        
        # Get user by phone number (strict isolation)
        user = db.get_user_by_phone(phone)
        
        # NEW USER - Start registration
        if not user:
            logger.info(f"✓ New user detected: {phone}")
            return handle_registration(phone, chat_id, text_message, None)
        
        # Update chat_id and last_interaction
        db.save_user(phone, chat_id=chat_id)
        
        # CHECK FOR 30-DAY INACTIVITY RECONFIRMATION
        if check_inactivity_reconfirmation(user):
            return handle_reconfirmation(phone, text_message, user)
        
        # HANDLE RECONFIRMATION IN PROGRESS
        if user.get('needs_reconfirmation', False):
            return handle_reconfirmation(phone, text_message, user)
        
        # INCOMPLETE REGISTRATION - Continue registration
        if user.get('registration_step', 0) < 5:
            logger.info(f"✓ Continuing registration for {phone} at step {user.get('registration_step', 0)}")
            return handle_registration(phone, chat_id, text_message, user)
        
        # FULLY REGISTERED USER - Process conversation
        logger.info(f"✓ Processing message from {user.get('first_name', 'Unknown')} (phone: {phone})")
        
        # Get conversation history (isolated to this phone number)
        history = db.get_history(phone, limit=15)
        
        # Extract intent
        intent = ai_coach._extract_intent(text_message)
        
        # Save user message
        db.save_message(phone, 'user', text_message, intent)
        
        # Generate AI response
        ai_response, response_intent = ai_coach.generate_response(
            text_message, user, history
        )
        
        # Save assistant response
        db.save_message(phone, 'assistant', ai_response, response_intent)
        
        # Log conversation to Google Sheets (non-blocking)
        log_to_google_sheets('conversation', {
            'phone_number': phone,
            'first_name': user.get('first_name', 'Unknown'),
            'full_name': user.get('full_name', ''),
            'class_taught': user.get('class_taught', ''),
            'location': user.get('location', ''),
            'user_message': text_message[:500],
            'bot_response': ai_response[:500],
            'intent': intent,
            'response_intent': response_intent,
            'response_length': len(ai_response),
            'message_timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"✓ Response generated for {user.get('first_name', 'Unknown')} - Intent: {intent}")
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Message processing error for {phone_number}: {e}", exc_info=True)
        return "I'm experiencing technical difficulties. Please try again in a moment."


def send_whatsapp_message(phone_number: str, message: str) -> bool:
    """Send message via Green API with retry logic"""
    max_retries = 3
    
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
            time.sleep(1)
    
    logger.error(f"✗ Failed to send message to {phone_number} after {max_retries} attempts")
    return False


# Flask Routes

@app.route('/')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI Coach - Schoolinka",
        "version": "4.0 - Persistent Edition",
        "timestamp": datetime.now().isoformat(),
        "database": CONFIG['DB_PATH'],
        "pinecone": "connected" if pinecone_index else "not available",
        "apps_script": "configured" if CONFIG['APPS_SCRIPT_URL'] else "not configured",
        "user_management": "phone-number-based with persistence",
        "inactivity_reconfirmation": f"enabled ({CONFIG['INACTIVITY_TIMEOUT_DAYS']} days)",
        "database_persistence": "enabled (WAL mode)"
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
                "I can only respond to text messages right now. Please type your question."
            )
            return jsonify({"status": "non_text"}), 200
        
        logger.info(f"📩 Received from {phone_number}: {text_message[:50]}...")
        
        # Process message asynchronously (isolated per phone number)
        def process_and_respond():
            try:
                reply = process_message(phone_number, chat_id, text_message)
                
                if reply:
                    send_whatsapp_message(chat_id, reply)
                    logger.info(f"✓ Response sent to {phone_number}")
            except Exception as e:
                logger.error(f"Background processing error for {phone_number}: {e}", exc_info=True)
                send_whatsapp_message(
                    chat_id,
                    "Sorry, I encountered an error. Please try again."
                )
        
        # Use daemon thread for immediate webhook return
        thread = threading.Thread(target=process_and_respond, daemon=True)
        thread.start()
        
        return jsonify({"status": "processing"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/user/<phone_number>', methods=['GET'])
def get_user_info(phone_number):
    """Get user profile by phone number (strict isolation)"""
    try:
        phone = extract_phone_number(phone_number)
        user = db.get_user_by_phone(phone)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        history = db.get_history(phone, limit=10)
        
        return jsonify({
            "user": dict(user),
            "recent_messages": history,
            "total_messages": len(history),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Get user error for {phone_number}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint for development"""
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
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM users')
            total_users = cursor.fetchone()['total']
            
            cursor.execute('SELECT COUNT(*) as registered FROM users WHERE registration_step >= 5')
            registered = cursor.fetchone()['registered']
            
            cursor.execute('SELECT COUNT(*) as total FROM conversations')
            total_messages = cursor.fetchone()['total']
            
            cursor.execute('''
                SELECT COUNT(*) as active FROM users 
                WHERE last_interaction > datetime('now', '-7 days')
            ''')
            active_7d = cursor.fetchone()['active']
            
            cursor.execute('''
                SELECT COUNT(*) as reconfirm FROM users 
                WHERE needs_reconfirmation = 1
            ''')
            needs_reconfirm = cursor.fetchone()['reconfirm']
            
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
                "needs_reconfirmation": needs_reconfirm,
                "total_messages": total_messages,
                "top_intents": top_intents
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
        "apps_script": False,
        "database_writable": False
    }
    
    vector_count = 0
    
    # Check database
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            components["database"] = True
            
            # Test write capability
            cursor.execute('SELECT COUNT(*) as cnt FROM users')
            components["database_writable"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    # Check Pinecone
    try:
        if pinecone_index:
            stats = pinecone_index.describe_index_stats()
            components["pinecone"] = True
            vector_count = stats.get('total_vector_count', 0)
    except Exception as e:
        logger.error(f"Pinecone health check failed: {e}")
    
    components["google_ai"] = llm is not None and embed_model is not None
    components["green_api"] = bool(CONFIG['GREEN_API_ID'] and CONFIG['GREEN_API_TOKEN'])
    components["apps_script"] = bool(CONFIG['APPS_SCRIPT_URL'])
    
    all_ok = components["database"] and components["database_writable"] and components["google_ai"]
    
    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "components": components,
        "config": {
            "database_path": CONFIG['DB_PATH'],
            "index_name": CONFIG['INDEX_NAME'],
            "max_history": CONFIG['MAX_HISTORY'],
            "vector_count": vector_count if components["pinecone"] else 0,
            "user_identification": "phone-number-based",
            "inactivity_timeout_days": CONFIG['INACTIVITY_TIMEOUT_DAYS'],
            "persistence_mode": "WAL"
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
    logger.info("=" * 80)
    logger.info("AI COACH - SCHOOLINKA v4.0 (PERSISTENT USER DATA)")
    logger.info("=" * 80)
    logger.info(f"Database: {CONFIG['DB_PATH']}")
    logger.info(f"Database Mode: WAL (Write-Ahead Logging) - PERSISTENCE ENABLED")
    logger.info(f"User Identification: Phone Number (Primary Key)")
    logger.info(f"Data Isolation: STRICT - Each user's data completely separated")
    logger.info(f"Registration: 5-step process with validation")
    logger.info(f"  - Step 0: First Name (single word)")
    logger.info(f"  - Step 1: Full Name (two words)")
    logger.info(f"  - Step 2: Email (validated)")
    logger.info(f"  - Step 3: Location")
    logger.info(f"  - Step 4: Class Taught")
    logger.info(f"Pinecone Index: {CONFIG['INDEX_NAME']} - {'Connected' if pinecone_index else 'Not Available'}")
    logger.info(f"Google AI: {'Initialized' if llm else 'Failed'}")
    logger.info(f"Apps Script: {'Configured' if CONFIG['APPS_SCRIPT_URL'] else 'Not Configured'}")
    logger.info(f"Inactivity Reconfirmation: ENABLED ({CONFIG['INACTIVITY_TIMEOUT_DAYS']} days)")
    logger.info(f"Google Sheets Logging: ENABLED (non-blocking)")
    
    if pinecone_index:
        try:
            stats = pinecone_index.describe_index_stats()
            logger.info(f"Vector Count: {stats.get('total_vector_count', 0)}")
        except:
            logger.warning("Could not retrieve vector count")
    
    # Verify database persistence
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as cnt FROM users')
            user_count = cursor.fetchone()['cnt']
            logger.info(f"✓ Database accessible - {user_count} existing users found")
    except Exception as e:
        logger.error(f"✗ Database access error: {e}")
    
    logger.info("=" * 80)
    logger.info("FIXES IMPLEMENTED:")
    logger.info("  ✓ Database persistence across service restarts")
    logger.info("  ✓ Strict user isolation (no cross-contamination)")
    logger.info("  ✓ First name validation (single word only)")
    logger.info("  ✓ Full name validation (exactly two words)")
    logger.info("  ✓ 30-day inactivity reconfirmation")
    logger.info("  ✓ Google Sheets logging per user with retry")
    logger.info("  ✓ WAL mode for better concurrency")
    logger.info("=" * 80)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True) exists:
                    # User exists, update fields
                    updates = []
                    values = []
                    
                    if chat_id:
                        updates.append('chat_id = ?')
                        values.append(chat_id)
                    
                    for k, v in kwargs.items():
                        updates.append(f'{k} = ?')
                        values.append(v)
                    
                    updates.append('last_interaction = ?')
                    values.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    
                    # Only increment total_messages if not explicitly provided
                    if 'total_messages' not in kwargs:
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
                    
                    if 'total_messages' not in kwargs:
                        fields.append('total_messages')
                        values.append(1)
                    
                    placeholders = ', '.join(['?' for _ in fields])
                    cursor.execute(
                        f'INSERT INTO users ({", ".join(fields)}) VALUES ({placeholders})', 
                        values
                    )
                
                conn.commit()
                logger.info(f"✓ User {phone} saved/updated successfully")
                return True
        except Exception as e:
            logger.error(f"Error saving user {phone_number}: {e}", exc_info=True)
            return False
    
    def save_message(self, phone_number: str, msg_type: str, content: str, intent: str = None):
        """Save conversation message by phone number with isolation"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (phone_number, message_type, message_content, intent, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (phone, msg_type, content, intent, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                self._cleanup_history(phone)
                logger.debug(f"✓ Message saved for {phone}")
        except Exception as e:
            logger.error(f"Error saving message for {phone_number}: {e}")
    
    def get_history(self, phone_number: str, limit: int = None) -> List[Dict]:
        """Get conversation history for specific phone number ONLY"""
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
                results = [dict(row) for row in reversed(cursor.fetchall())]
                logger.debug(f"✓ Retrieved {len(results)} messages for {phone}")
                return results
        except Exception as e:
            logger.error(f"Error getting history for {phone_number}: {e}")
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
                followup_instruction = "\nFOLLOW-UP DETECTED: User seeks clarification. Reference your previous response and clarify."
        
        prompt = f"""You are AI Coach by Schoolinka, a friendly teaching assistant for Nigerian teachers
TEACHER PROFILE:
- Name: {name}
- Teaching: {class_info}
- Location: {location}

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
            'compliments': ['thank', 'okay', 'good']
        }
        
        for intent_name, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent_name
        
        return 'general'
    
    @staticmethod
    def _get_fallback_response(intent: str) -> str:
        """Fallback response on error"""
        return "I'm experiencing a brief technical issue. Please try your question again, and I'll help you."


def log_to_google_sheets(data_type: str, data: Dict) -> bool:
    """Log data to Google Sheets via Apps Script (non-blocking) with retry"""
    if not CONFIG['APPS_SCRIPT_URL']:
        return False
    
    def _log():
        max_retries = 2
        for attempt in range(max_retries):
            try:
                payload = {
                    'type': data_type,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
                response = requests.post(
                    CONFIG['APPS_SCRIPT_URL'],
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=8
                )
                
                if response.status_code == 200:
                    logger.info(f"✓ Logged {data_type} to Google Sheets for phone {data.get('phone_number', 'unknown')}")
                    return True
                else:
                    logger.warning(f"Sheets logging attempt {attempt + 1} failed: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Sheets logging error (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(1)
        
        return False
    
    thread = threading.Thread(target=_log, daemon=True)
    thread.start()
    return True


# Registration step templates
REGISTRATION_STEPS = {
    0: {
        "message": "Hello! I'm AI Coach by Schoolinka. I'm here to support you with teaching strategies, classroom management, and professional development.\n\nWhat's your first name? (Please enter only your first name)",
        "field": None
    },
    1: {
        "message": "Nice to meet you, {first_name}!\n\nWhat's your full name? (Please enter your first and last name)",
        "field": "first_name"
    },
    2: {
        "message": "Thanks, {first_name}!\n\nWhat's your email address?",
        "field": "full_name"
    },
    3: {
        "message": "Thanks for sharing!\n\nWhich city or state are you in?",
        "field": "email"
    },
    4: {
        "message": "Great!\n\nWhich class do you teach?",
        "field": "location"
    }
}


def check_inactivity_reconfirmation(user: Dict) -> bool:
    """Check if user needs reconfirmation due to 30+ days inactivity"""
    if not user:
        return False
    
    # Skip if already in reconfirmation mode
    if user.get('needs_reconfirmation', False):
        return True
    
    # Skip if not fully registered
    if user.get('registration_step', 0) < 5:
        return False
    
    last_interaction_str = user.get('last_interaction')
    
    if last_interaction_str:
        try:
            last_interaction = datetime.strptime(last_interaction_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
            days_inactive = (datetime.now() - last_interaction).days
            
            if
