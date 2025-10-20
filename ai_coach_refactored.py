"""
AI Coach by Schoolinka - Production-Ready WhatsApp Chatbot
A professional teaching assistant for Nigerian educators
Author: Refactored for production use
Version: 5.0
"""

import os
import time
import re
import sqlite3
import logging
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple

import requests
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone


# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_coach.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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


# ============================================================================
# VALIDATION & UTILITY FUNCTIONS
# ============================================================================

def extract_phone_number(chat_id: str) -> str:
    """Extract phone number from WhatsApp chat_id (format: 234XXXXXXXXXX@c.us)"""
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


def validate_email(email: str) -> bool:
    """Validate email format"""
    email = email.strip().lower()
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return '@' in email and '.' in email and re.match(pattern, email)


# ============================================================================
# DATABASE MANAGER
# ============================================================================

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
                
                # Enable WAL mode for better concurrency
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute('PRAGMA synchronous=NORMAL')
                
                # Users table
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
                
                # Conversations table
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
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_phone_number ON conversations (phone_number)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_phone ON users (phone_number)')
                
                conn.commit()
                logger.info(f"✓ Database initialized at: {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    @contextmanager
    def get_conn(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level='DEFERRED')
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_user_by_phone(self, phone_number: str) -> Optional[Dict]:
        """Get user by phone number"""
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
        """Save or update user by phone number"""
        try:
            phone = re.sub(r'\D', '', phone_number)
            
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT phone_number FROM users WHERE phone_number = ?', (phone,))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing user
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
                    
                    if 'total_messages' not in kwargs:
                        updates.append('total_messages = total_messages + 1')
                    
                    values.append(phone)
                    update_str = ', '.join(updates)
                    cursor.execute(f'UPDATE users SET {update_str} WHERE phone_number = ?', values)
                else:
                    # Insert new user
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
                return True
        except Exception as e:
            logger.error(f"Error saving user {phone_number}: {e}", exc_info=True)
            return False
    
    def save_message(self, phone_number: str, msg_type: str, content: str, intent: str = None):
        """Save conversation message"""
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
    
    def _cleanup_history(self, phone_number: str):
        """Clean up old messages beyond max history"""
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


# ============================================================================
# RESPONSE FORMATTER
# ============================================================================

class ResponseFormatter:
    """Format responses for WhatsApp"""
    
    @staticmethod
    def clean_response(text: str) -> str:
        """Remove markdown and format for WhatsApp"""
        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'#{1,6}\s', '', text)
        
        # Convert bullets to numbered lists
        lines = text.split('\n')
        formatted_lines = []
        list_counter = 1
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                list_counter = 1
                continue
            
            if re.match(r'^[-•·]\s+', stripped):
                content = re.sub(r'^[-•·]\s+', '', stripped)
                formatted_lines.append(f"{list_counter}. {content}")
                list_counter += 1
            else:
                formatted_lines.append(stripped)
        
        result = '\n'.join(formatted_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()


# ============================================================================
# AI COACH
# ============================================================================

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
            r'\b(list|give\s+me|provide|show)\s+(all\s+)?(items?|functions?|examples?|steps?)',
            r'\b(break\s+down|expand\s+on)\b',
            r'\b(what\s+about|how\s+about|tell\s+me\s+about)\b',
            r'\b(previous|last|earlier)\s+(response|answer)',
            r'\b(you\s+mentioned|you\s+said)\b',
        ]
        
        is_followup = any(re.search(pattern, msg_lower) for pattern in followup_indicators)
        
        followup_type = 'clarification'
        if re.search(r'\blist\b', msg_lower):
            followup_type = 'list_expansion'
        elif re.search(r'\b(example|steps?)\b', msg_lower):
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
                if match.get('score', 0) > 0.65:
                    text = match.get('metadata', {}).get('text', '')
                    source = match.get('metadata', {}).get('source', 'Knowledge Base')
                    if text:
                        contents.append(text[:500])
                        sources.append(source)
            
            if contents:
                logger.info(f"✓ Retrieved {len(contents)} relevant documents")
            
            return '\n\n'.join(contents), sources
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return "", []
    
    def generate_response(self, message: str, user_profile: Dict, 
                         history: List[Dict] = None) -> Tuple[str, str]:
        """Generate formatted AI response"""
        try:
            intent = self._extract_intent(message)
            is_followup, followup_type = self.detect_followup_intent(message, history or [])
            
            # Get RAG content
            rag_content = ""
            if not is_followup or followup_type == 'clarification':
                rag_content, _ = self.get_rag_content(message, intent)
            
            # Build context
            context = self._build_context(history)
            assistant_context = self._get_last_bot_responses(history, 3) if is_followup else ""
            
            # Build system prompt
            system_prompt = self._build_system_prompt(
                user_profile, rag_content, context, intent, message, 
                is_followup, followup_type, assistant_context
            )
            
            # Generate response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            response = self.llm.invoke(messages)
            clean_response = self.formatter.clean_response(response.content)
            
            return clean_response, intent
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm experiencing a brief technical issue. Please try your question again.", "error"
    
    def _build_context(self, history: List[Dict]) -> str:
        """Build conversation context from history"""
        if not history:
            return ""
        
        recent = history[-10:]
        context_parts = []
        for msg in recent:
            role = "Teacher" if msg['message_type'] == 'user' else "AI Coach"
            content = msg['message_content'][:150]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
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
            return "PREVIOUS BOT RESPONSES:\n" + "\n\n".join([f"- {resp}" for resp in bot_responses]) + "\n\n"
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
                followup_instruction = "\nFOLLOW-UP: Provide comprehensive numbered list with descriptions."
            elif followup_type == 'detailed_breakdown':
                followup_instruction = "\nFOLLOW-UP: Expand with detailed steps and examples."
            else:
                followup_instruction = "\nFOLLOW-UP: Reference previous response and clarify."
        
        prompt = f"""You are AI Coach by Schoolinka, a friendly teaching assistant for Nigerian teachers.

TEACHER PROFILE:
- Name: {name}
- Teaching: {class_info}
- Location: {location}

CURRENT QUERY: {query}
DETECTED INTENT: {intent}
{followup_instruction}

{assistant_context}

CORE GUIDELINES:
- AI Coach is by Schoolinka (founded by Oluwaseun Kayode)
- Schoolinka offers training, certifications, teaching resources, and a job board for educators
- Be warm, conversational, and encouraging
- Provide practical, Nigeria-specific advice
- Consider large class sizes (30-60 students) and limited resources

FORMATTING:
- Use numbered lists (1. 2. 3.)
- Keep paragraphs short (2-3 sentences)
- NO asterisks, bullets, or markdown
- Add line breaks between sections

RESPONSE LENGTH:
- Simple questions: 4-6 sentences
- How-to: Detailed numbered steps
- Complex topics: 3-4 paragraphs"""
        
        if rag_content:
            prompt += f"\n\nKNOWLEDGE BASE:\n{rag_content}"
        
        if context:
            prompt += f"\n\nRECENT CONVERSATION:\n{context}"
        
        prompt += "\n\nProvide a helpful, well-formatted response."
        
        return prompt
    
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Extract intent from message"""
        msg = message.lower()
        
        intents = {
            'teaching_strategy': ['teach', 'strategy', 'method', 'lesson', 'explain', 'activity'],
            'classroom_management': ['discipline', 'behavior', 'manage', 'control', 'disruptive'],
            'assessment': ['assess', 'evaluate', 'grade', 'test', 'exam', 'feedback'],
            'wellbeing': ['stress', 'tired', 'overwhelmed', 'burnout', 'exhausted'],
            'curriculum': ['curriculum', 'syllabus', 'topic', 'scheme of work'],
            'parent_communication': ['parent', 'guardian', 'meeting', 'report'],
            'resources': ['resource', 'material', 'tool', 'equipment']
        }
        
        for intent_name, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent_name
        
        return 'general'


# ============================================================================
# REGISTRATION & RECONFIRMATION
# ============================================================================

def check_inactivity_reconfirmation(user: Dict) -> bool:
    """Check if user needs reconfirmation due to inactivity"""
    if not user or user.get('needs_reconfirmation', False):
        return user.get('needs_reconfirmation', False)
    
    if user.get('registration_step', 0) < 5:
        return False
    
    last_interaction_str = user.get('last_interaction')
    if last_interaction_str:
        try:
            last_interaction = datetime.strptime(last_interaction_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
            days_inactive = (datetime.now() - last_interaction).days
            
            if days_inactive >= CONFIG['INACTIVITY_TIMEOUT_DAYS']:
                logger.info(f"User {user.get('first_name')} inactive for {days_inactive} days")
                return True
        except Exception as e:
            logger.error(f"Error checking inactivity: {e}")
    
    return False


def handle_reconfirmation(phone_number: str, text: str, user: Dict, db: DatabaseManager) -> Optional[str]:
    """Handle reconfirmation after inactivity"""
    text = text.strip().lower()
    
    # First time showing reconfirmation
    if not user.get('needs_reconfirmation', False):
        db.save_user(phone_number, needs_reconfirmation=True)
        
        name = user.get('first_name', 'Teacher')
        full_name = user.get('full_name', 'your name')
        location = user.get('location', 'your location')
        class_taught = user.get('class_taught', 'your class')
        
        return (
            f"Welcome back, {name}! It's been a while.\n\n"
            f"Confirm your profile:\n"
            f"Name: {full_name}\n"
            f"Location: {location}\n"
            f"Teaching: {class_taught}\n\n"
            f"Is this correct? (Reply 'Yes' or 'No')"
        )
    
    # User responding
    if text in ['yes', 'y', 'correct']:
        db.save_user(phone_number, needs_reconfirmation=False)
        return "Great! Let's get back to coaching. What's on your mind today?"
    elif text in ['no', 'n', 'wrong']:
        db.save_user(phone_number, profile_complete=False, registration_step=0, needs_reconfirmation=False)
        return "No problem! Let's update your profile.\n\n" + REGISTRATION_STEPS[0]["message"]
    else:
        return "Please reply 'Yes' to confirm or 'No' to update your details."


def handle_registration(phone_number: str, chat_id: str, text: str, user: Optional[Dict], db: DatabaseManager) -> str:
    """Handle multi-step user registration"""
    text = text.strip()
    
    # New user
    if not user:
        db.save_user(phone_number, chat_id=chat_id, profile_complete=False, registration_step=0)
        log_to_google_sheets('new_user', {
            'phone_number': phone_number,
            'status': 'Registration Started',
            'timestamp': datetime.now().isoformat()
        })
        return REGISTRATION_STEPS[0]["message"]
    
    step = user.get('registration_step', 0)
    
    # Registration complete
    if step >= 5 and not user.get('needs_reconfirmation', False):
        return None
    
    if not text:
        current_template = REGISTRATION_STEPS.get(step, REGISTRATION_STEPS[0])
        return f"Please provide your {current_template.get('field', 'information')} to continue."
    
    # Step 0: First Name
    if step == 0:
        is_valid, result = validate_first_name(text)
        if is_valid:
            db.save_user(phone_number, chat_id=chat_id, first_name=result, registration_step=1)
            return REGISTRATION_STEPS[1]["message"].format(first_name=result)
        return result
    
    # Step 1: Full Name
    elif step == 1:
        is_valid, result = validate_full_name(text)
        if is_valid:
            db.save_user(phone_number, full_name=result, registration_step=2)
            name = user.get('first_name', 'Teacher')
            return REGISTRATION_STEPS[2]["message"].format(first_name=name)
        return result
    
    # Step 2: Email
    elif step == 2:
        if validate_email(text):
            db.save_user(phone_number, email=text.strip().lower(), registration_step=3)
            name = user.get('first_name', 'Teacher')
            return REGISTRATION_STEPS[3]["message"].format(first_name=name)
        return "Please enter a valid email address (e.g., name@example.com)."
    
    # Step 3: Location
    elif step == 3:
        location = text.strip().title()
        if len(location) > 2:
            db.save_user(phone_number, location=location, registration_step=4)
            name = user.get('first_name', 'Teacher')
            return REGISTRATION_STEPS[4]["message"].format(first_name=name)
        return "Please enter a valid location (city or state)."
    
    # Step 4: Class Taught
    elif step == 4:
        class_taught = text.strip().title()
        if len(class_taught) > 1:
            db.save_user(phone_number, class_taught=class_taught, profile_complete=True, registration_step=5)
            
            updated_user = db.get_user_by_phone(phone_number)
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
            
            return (
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
        return "Please enter the class you teach (e.g., Primary 3, JSS 2, SS 1)."
    
    return "Something went wrong. Please try again."


# ============================================================================
# EXTERNAL SERVICES
# ============================================================================

def log_to_google_sheets(data_type: str, data: Dict) -> bool:
    """Log data to Google Sheets via Apps Script (non-blocking)"""
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
                    logger.info(f"✓ Logged {data_type} to Google Sheets")
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
            logger.error(f"Send error (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(1)
    
    logger.error(f"✗ Failed to send message to {phone_number} after {max_retries} attempts")
    return False


# ============================================================================
# MESSAGE PROCESSING
# ============================================================================

def process_message(phone_number: str, chat_id: str, text_message: str, 
                   db: DatabaseManager, ai_coach: AICoach) -> str:
    """Main message processing logic with strict user isolation"""
    try:
        if not text_message or len(text_message.strip()) < 1:
            return "Please send me a message or question. I'm here to help!"
        
        text_message = text_message.strip()
        phone = extract_phone_number(phone_number)
        
        # Get user
        user = db.get_user_by_phone(phone)
        
        # New user - start registration
        if not user:
            logger.info(f"✓ New user detected: {phone}")
            return handle_registration(phone, chat_id, text_message, None, db)
        
        # Update chat_id and last_interaction
        db.save_user(phone, chat_id=chat_id)
        
        # Check for inactivity reconfirmation
        if check_inactivity_reconfirmation(user):
            return handle_reconfirmation(phone, text_message, user, db)
        
        # Handle reconfirmation in progress
        if user.get('needs_reconfirmation', False):
            return handle_reconfirmation(phone, text_message, user, db)
        
        # Incomplete registration
        if user.get('registration_step', 0) < 5:
            logger.info(f"✓ Continuing registration for {phone}")
            return handle_registration(phone, chat_id, text_message, user, db)
        
        # Fully registered user - process conversation
        logger.info(f"✓ Processing message from {user.get('first_name', 'Unknown')}")
        
        # Get conversation history
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
        
        # Log conversation to Google Sheets
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
            'message_timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"✓ Response generated - Intent: {intent}")
        return ai_response
        
    except Exception as e:
        logger.error(f"Message processing error: {e}", exc_info=True)
        return "I'm experiencing technical difficulties. Please try again in a moment."


# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

def initialize_services():
    """Initialize AI services and database"""
    # Validate configuration
    required_keys = ['PINECONE_API_KEY', 'GOOGLE_API_KEY', 'GREEN_API_ID', 'GREEN_API_TOKEN']
    missing_keys = [key for key in required_keys if not CONFIG[key]]
    
    if missing_keys:
        logger.error(f"Missing required configuration: {', '.join(missing_keys)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    # Initialize Pinecone
    pinecone_index = None
    try:
        logger.info("Initializing Pinecone...")
        pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
        pinecone_index = pc.Index(CONFIG['INDEX_NAME'])
        stats = pinecone_index.describe_index_stats()
        logger.info(f"✓ Connected to Pinecone - Vectors: {stats.get('total_vector_count', 0)}")
    except Exception as e:
        logger.warning(f"Pinecone initialization failed: {e}. Continuing without RAG support.")
    
    # Initialize Google AI
    try:
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
        logger.error(f"✗ Google AI initialization failed: {e}")
        raise
    
    # Initialize Database
    db = DatabaseManager(CONFIG['DB_PATH'])
    
    # Initialize AI Coach
    ai_coach = AICoach(llm, embed_model, pinecone_index)
    
    return db, ai_coach, pinecone_index


# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)

# Global instances (initialized on startup)
db = None
ai_coach = None
pinecone_index = None


@app.route('/')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI Coach - Schoolinka",
        "version": "5.0 - Production",
        "timestamp": datetime.now().isoformat(),
        "database": CONFIG['DB_PATH'],
        "features": {
            "pinecone": "connected" if pinecone_index else "unavailable",
            "user_management": "phone-based with persistence",
            "inactivity_reconfirmation": f"{CONFIG['INACTIVITY_TIMEOUT_DAYS']} days",
            "registration_steps": 5
        }
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages"""
    try:
        data = request.get_json()
        
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored"}), 200
        
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        if not chat_id or '@g.us' in chat_id:
            return jsonify({"status": "ignored"}), 200
        
        phone_number = extract_phone_number(chat_id)
        
        # Extract text message
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
        
        logger.info(f"📩 Received from {phone_number}: {text_message[:50]}...")
        
        # Process message asynchronously
        def process_and_respond():
            try:
                reply = process_message(phone_number, chat_id, text_message, db, ai_coach)
                if reply:
                    send_whatsapp_message(chat_id, reply)
            except Exception as e:
                logger.error(f"Background processing error: {e}", exc_info=True)
                send_whatsapp_message(chat_id, "Sorry, I encountered an error. Please try again.")
        
        thread = threading.Thread(target=process_and_respond, daemon=True)
        thread.start()
        
        return jsonify({"status": "processing"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/user/<phone_number>', methods=['GET'])
def get_user_info(phone_number):
    """Get user profile by phone number"""
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
        logger.error(f"Get user error: {e}")
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
        response = process_message(phone, chat_id, message, db, ai_coach)
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
        "apps_script": bool(CONFIG['APPS_SCRIPT_URL'])
    }
    
    vector_count = 0
    
    # Check database
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as cnt FROM users')
            components["database"] = True
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
    
    components["google_ai"] = ai_coach is not None
    components["green_api"] = bool(CONFIG['GREEN_API_ID'] and CONFIG['GREEN_API_TOKEN'])
    
    all_ok = components["database"] and components["google_ai"]
    
    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "components": components,
        "config": {
            "database_path": CONFIG['DB_PATH'],
            "index_name": CONFIG['INDEX_NAME'],
            "max_history": CONFIG['MAX_HISTORY'],
            "vector_count": vector_count,
            "inactivity_timeout_days": CONFIG['INACTIVITY_TIMEOUT_DAYS']
        },
        "timestamp": datetime.now().isoformat()
    }), 200 if all_ok else 503


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("AI COACH - SCHOOLINKA v5.0 (PRODUCTION)")
    logger.info("=" * 80)
    
    try:
        # Initialize services
        db, ai_coach, pinecone_index = initialize_services()
        
        logger.info(f"Database: {CONFIG['DB_PATH']}")
        logger.info(f"Pinecone: {'Connected' if pinecone_index else 'Not Available'}")
        logger.info(f"Registration: 5-step process with validation")
        logger.info(f"Inactivity Timeout: {CONFIG['INACTIVITY_TIMEOUT_DAYS']} days")
        logger.info(f"Google Sheets Logging: {'Enabled' if CONFIG['APPS_SCRIPT_URL'] else 'Disabled'}")
        
        # Verify database
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as cnt FROM users')
            user_count = cursor.fetchone()['cnt']
            logger.info(f"✓ Database accessible - {user_count} existing users")
        
        logger.info("=" * 80)
        logger.info("PRODUCTION FEATURES:")
        logger.info("  ✓ Phone-number-based user isolation")
        logger.info("  ✓ WAL mode for database persistence")
        logger.info("  ✓ Multi-step registration with validation")
        logger.info("  ✓ 30-day inactivity reconfirmation")
        logger.info("  ✓ RAG-enhanced responses")
        logger.info("  ✓ Google Sheets logging")
        logger.info("  ✓ Retry logic for external services")
        logger.info("=" * 80)
        
        # Start Flask app
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise