import os
import asyncio
import nest_asyncio
from datetime import datetime, timedelta
import re
import sqlite3
from contextlib import contextmanager
from flask import Flask, request, jsonify
import threading
import json
import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from pinecone import Pinecone
from whatsapp_chatbot_python import GreenAPIBot, Notification
import logging
from typing import List, Dict, Optional, Tuple
import sys
import pysqlite3
import gspread
from google.oauth2.service_account import Credentials
import time

# Fix SQLite import issue
sys.modules["sqlite3"] = pysqlite3

# Apply nest_asyncio patch
nest_asyncio.apply()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teaching_coach.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Get API keys from environment variables with fallbacks
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw")
GREEN_API_ID_INSTANCE = os.getenv("GREEN_API_ID_INSTANCE", "7105328354")
GREEN_API_TOKEN = os.getenv("GREEN_API_TOKEN", "2a33db828fe64c57a32debcca8f065cac2f901d270d04347a5")

# Google Sheets credentials (optional)
GOOGLE_SHEETS_CREDENTIALS_FILE = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "credentials.json")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "116616118324951765726")

# Database configuration
DB_PATH = "teaching_coach.db"
MAX_CONVERSATION_HISTORY = 20

# Validate essential API keys
if not all([PINECONE_API_KEY, GOOGLE_API_KEY, GREEN_API_ID_INSTANCE, GREEN_API_TOKEN]):
    logger.error("Missing required API keys!")
    raise ValueError("Required API keys are missing.")

# Initialize Pinecone and embedding model with error handling
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("coach")
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    logger.info("Pinecone and embeddings initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Pinecone/Embeddings: {e}")
    pinecone_index = None
    embed_model = None

# Initialize Google Generative AI
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_tokens=1000
    )
    logger.info("Google Generative AI initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Google AI: {e}")
    llm = None

class GoogleSheetsManager:
    """Manages Google Sheets operations for real-time data storage"""
    
    def __init__(self, credentials_file: str, spreadsheet_id: str):
        self.credentials_file = credentials_file
        self.spreadsheet_id = spreadsheet_id
        self.gc = None
        self.sheet = None
        self.users_worksheet = None
        self.conversations_worksheet = None
        self.is_enabled = False
        self.initialize_sheets()
    
    def initialize_sheets(self):
        """Initialize Google Sheets connection and worksheets"""
        try:
            # Define the scope for Google Sheets API
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
            
            # Load credentials from service account file
            if os.path.exists(self.credentials_file):
                creds = Credentials.from_service_account_file(
                    self.credentials_file, scopes=scope
                )
                self.gc = gspread.authorize(creds)
                
                # Open the spreadsheet
                self.sheet = self.gc.open_by_key(self.spreadsheet_id)
                
                # Get or create worksheets
                self.users_worksheet = self.get_or_create_worksheet("Users", [
                    "Chat ID", "First Name", "Email", "Phone Number", "Location", 
                    "Teacher Name", "Class Taught", "Profile Complete", "Created At", 
                    "Last Interaction", "Total Messages"
                ])
                
                self.conversations_worksheet = self.get_or_create_worksheet("Conversations", [
                    "Timestamp", "Chat ID", "First Name", "Message Type", "Message Content", 
                    "Intent", "RAG Sources", "Session ID"
                ])
                
                self.is_enabled = True
                logger.info("Google Sheets initialized successfully")
            else:
                logger.warning(f"Google Sheets credentials file not found: {self.credentials_file}")
                
        except Exception as e:
            logger.error(f"Error initializing Google Sheets: {e}")
            self.is_enabled = False
    
    def get_or_create_worksheet(self, title: str, headers: List[str]):
        """Get existing worksheet or create new one with headers"""
        try:
            # Try to get existing worksheet
            worksheet = self.sheet.worksheet(title)
            
            # Check if headers exist, if not add them
            existing_headers = worksheet.row_values(1)
            if not existing_headers or existing_headers != headers:
                worksheet.clear()
                worksheet.append_row(headers)
                logger.info(f"Headers updated for worksheet: {title}")
                
        except gspread.WorksheetNotFound:
            # Create new worksheet
            worksheet = self.sheet.add_worksheet(title=title, rows=1000, cols=len(headers))
            worksheet.append_row(headers)
            logger.info(f"Created new worksheet: {title}")
            
        return worksheet
    
    def save_user_to_sheets(self, user_data: Dict):
        """Save or update user data in Google Sheets"""
        if not self.is_enabled or not self.users_worksheet:
            return False
            
        try:
            # Find existing user row
            cell = None
            try:
                cell = self.users_worksheet.find(user_data['chat_id'])
            except gspread.CellNotFound:
                pass
            
            user_row = [
                user_data['chat_id'],
                user_data.get('first_name', ''),
                user_data.get('email', ''),
                user_data.get('phone_number', ''),
                user_data.get('location', ''),
                user_data.get('teacher_name', ''),
                user_data.get('class_taught', ''),
                str(user_data.get('profile_complete', False)),
                user_data.get('created_at', datetime.now().isoformat()),
                user_data.get('last_interaction', datetime.now().isoformat()),
                str(user_data.get('total_messages', 0))
            ]
            
            if cell:
                # Update existing row
                row_number = cell.row
                self.users_worksheet.update(f'A{row_number}:K{row_number}', [user_row])
                logger.info(f"Updated user data in Google Sheets: {user_data['chat_id']}")
            else:
                # Add new row
                self.users_worksheet.append_row(user_row)
                logger.info(f"Added new user to Google Sheets: {user_data['chat_id']}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving user to Google Sheets: {e}")
            return False
    
    def save_conversation_to_sheets(self, conversation_data: Dict):
        """Save conversation message to Google Sheets"""
        if not self.is_enabled or not self.conversations_worksheet:
            return False
            
        try:
            conversation_row = [
                conversation_data.get('timestamp', datetime.now().isoformat()),
                conversation_data['chat_id'],
                conversation_data.get('first_name', ''),
                conversation_data['message_type'],
                conversation_data['message_content'],
                conversation_data.get('intent', ''),
                conversation_data.get('rag_sources', ''),
                conversation_data.get('session_id', '')
            ]
            
            self.conversations_worksheet.append_row(conversation_row)
            logger.info(f"Saved conversation to Google Sheets: {conversation_data['chat_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation to Google Sheets: {e}")
            return False

class DatabaseManager:
    """Enhanced database manager with better error handling"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with updated schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create users table with new fields
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        chat_id TEXT PRIMARY KEY,
                        first_name TEXT,
                        email TEXT,
                        phone_number TEXT,
                        location TEXT,
                        teacher_name TEXT,
                        class_taught TEXT,
                        profile_complete BOOLEAN DEFAULT FALSE,
                        first_interaction BOOLEAN DEFAULT TRUE,
                        registration_step INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_messages INTEGER DEFAULT 0
                    )
                ''')
                
                # Create conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id TEXT NOT NULL,
                        message_type TEXT NOT NULL CHECK(message_type IN ('user', 'assistant')),
                        message_content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        session_id TEXT,
                        message_intent TEXT,
                        rag_sources TEXT,
                        FOREIGN KEY (chat_id) REFERENCES users (chat_id)
                    )
                ''')
                
                # Create analytics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        action_data TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_chat_id ON conversations (chat_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_chat_id ON analytics (chat_id)')
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_user_profile(self, chat_id: str) -> Optional[Dict]:
        """Get user profile by chat_id"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE chat_id = ?', (chat_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting user profile for {chat_id}: {e}")
            return None
    
    def create_or_update_user(self, chat_id: str, **kwargs) -> bool:
        """Create new user or update existing user profile with flexible parameters"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if user exists
                cursor.execute('SELECT chat_id FROM users WHERE chat_id = ?', (chat_id,))
                exists = cursor.fetchone()
                
                current_time = datetime.now()
                
                if exists:
                    # Update existing user
                    update_fields = []
                    update_values = []
                    
                    for key, value in kwargs.items():
                        if key in ['first_name', 'email', 'phone_number', 'location', 
                                  'teacher_name', 'class_taught', 'registration_step', 
                                  'profile_complete']:
                            update_fields.append(f'{key} = ?')
                            update_values.append(value)
                    
                    # Always update last_interaction and increment total_messages
                    update_fields.extend(['last_interaction = ?', 'total_messages = total_messages + 1'])
                    update_values.extend([current_time, chat_id])
                    
                    if update_fields[:-2]:  # If there are other updates besides timestamp
                        query = f"UPDATE users SET {', '.join(update_fields)} WHERE chat_id = ?"
                        cursor.execute(query, update_values)
                else:
                    # Create new user
                    fields = ['chat_id', 'first_interaction', 'total_messages', 'created_at', 'last_interaction']
                    values = [chat_id, True, 1, current_time, current_time]
                    
                    for key, value in kwargs.items():
                        if key in ['first_name', 'email', 'phone_number', 'location', 
                                  'teacher_name', 'class_taught', 'registration_step', 
                                  'profile_complete']:
                            fields.append(key)
                            values.append(value)
                    
                    placeholders = ', '.join(['?' for _ in values])
                    query = f"INSERT INTO users ({', '.join(fields)}) VALUES ({placeholders})"
                    cursor.execute(query, values)
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error creating/updating user {chat_id}: {e}")
            return False
    
    def save_message(self, chat_id: str, message_type: str, content: str, 
                    session_id: str = None, intent: str = None, rag_sources: str = None) -> bool:
        """Save a conversation message"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations (chat_id, message_type, message_content, 
                                             session_id, message_intent, rag_sources)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (chat_id, message_type, content, session_id, intent, rag_sources))
                conn.commit()
                
                # Clean old messages to maintain history limit
                self._cleanup_old_messages(chat_id)
                return True
                
        except Exception as e:
            logger.error(f"Error saving message for {chat_id}: {e}")
            return False
    
    def get_conversation_history(self, chat_id: str, limit: int = MAX_CONVERSATION_HISTORY) -> List[Dict]:
        """Get conversation history for a user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_type, message_content, timestamp, message_intent, rag_sources
                    FROM conversations 
                    WHERE chat_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (chat_id, limit))
                
                rows = cursor.fetchall()
                return [dict(row) for row in reversed(rows)]
                
        except Exception as e:
            logger.error(f"Error getting conversation history for {chat_id}: {e}")
            return []
    
    def _cleanup_old_messages(self, chat_id: str):
        """Remove old messages beyond the limit"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM conversations 
                    WHERE chat_id = ? AND id NOT IN (
                        SELECT id FROM conversations 
                        WHERE chat_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    )
                ''', (chat_id, chat_id, MAX_CONVERSATION_HISTORY * 2))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up old messages for {chat_id}: {e}")
    
    def log_analytics(self, chat_id: str, action_type: str, action_data: Dict = None):
        """Log analytics data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO analytics (chat_id, action_type, action_data)
                    VALUES (?, ?, ?)
                ''', (chat_id, action_type, json.dumps(action_data) if action_data else None))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging analytics: {e}")
    
    def get_stats(self) -> Dict:
        """Get basic statistics for health check"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) as total_users FROM users')
                total_users = cursor.fetchone()['total_users']
                
                cursor.execute('SELECT COUNT(*) as completed_profiles FROM users WHERE profile_complete = 1')
                completed_profiles = cursor.fetchone()['completed_profiles']
                
                cursor.execute('SELECT COUNT(*) as total_messages FROM conversations')
                total_messages = cursor.fetchone()['total_messages']
                
                return {
                    'total_users': total_users,
                    'completed_profiles': completed_profiles,
                    'total_messages': total_messages
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'total_users': 0, 'completed_profiles': 0, 'total_messages': 0}

class ConversationAnalyzer:
    """Enhanced analyzer with registration step detection"""
    
    @staticmethod
    def extract_intent(message: str) -> str:
        """Extract user intent from message"""
        message_lower = message.lower()
        
        # Registration steps
        if re.search(r'^[a-zA-Z\s]{2,30}$', message.strip()) and len(message.split()) <= 3:
            return 'name_input'
        elif '@' in message and '.' in message:
            return 'email_input'
        elif re.search(r'[\+\d\s\-\(\)]{7,15}', message.strip()):
            return 'phone_input'
        elif any(word in message_lower for word in ['lagos', 'abuja', 'ibadan', 'kano', 'kaduna', 'nigeria']):
            return 'location_input'
        
        # Teaching strategies
        if any(word in message_lower for word in ['strategy', 'method', 'approach', 'technique', 'how to teach']):
            return 'teaching_strategy'
        
        # Classroom management
        if any(word in message_lower for word in ['discipline', 'behavior', 'manage', 'control', 'disruptive']):
            return 'classroom_management'
        
        # Student assessment
        if any(word in message_lower for word in ['assess', 'evaluate', 'grade', 'test', 'exam', 'quiz']):
            return 'assessment'
        
        # Stress and wellbeing
        if any(word in message_lower for word in ['stress', 'tired', 'overwhelmed', 'burnout', 'wellbeing']):
            return 'teacher_wellbeing'
        
        # Parent communication
        if any(word in message_lower for word in ['parent', 'guardian', 'meeting', 'communicate']):
            return 'parent_communication'
        
        # Curriculum and planning
        if any(word in message_lower for word in ['curriculum', 'lesson plan', 'scheme', 'planning']):
            return 'curriculum_planning'
        
        # Profile setup
        if any(phrase in message_lower for phrase in ['i teach', 'my class', 'primary', 'secondary', 'jss', 'sss']):
            return 'teaching_class_input'
        
        return 'general_inquiry'
    
    @staticmethod
    def analyze_conversation_context(history: List[Dict]) -> Dict:
        """Analyze conversation context for better responses"""
        if not history:
            return {'context': 'new_conversation', 'topics': [], 'sentiment': 'neutral'}
        
        recent_intents = []
        
        for entry in history[-10:]:
            intent = entry.get('message_intent', '')
            if intent:
                recent_intents.append(intent)
        
        context = 'ongoing'
        if len(history) <= 2:
            context = 'early_conversation'
        elif 'teacher_wellbeing' in recent_intents:
            context = 'support_needed'
        elif len(set(recent_intents)) == 1:
            context = 'focused_discussion'
        
        return {
            'context': context,
            'recent_intents': recent_intents,
            'message_count': len(history),
            'dominant_intent': max(set(recent_intents), key=recent_intents.count) if recent_intents else None
        }

class EnhancedTeacherAI:
    """Enhanced AI with registration flow and personalization"""
    
    def __init__(self, llm, embed_model, pinecone_index, db_manager):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        self.db_manager = db_manager
        self.analyzer = ConversationAnalyzer()
        
        # Enhanced system prompt with welcome back functionality
        self.system_prompt_template = """
        You are Coach bot, an experienced and empathetic AI teaching assistant for Nigerian teachers.
        
        Your responsibilities:
        1. Provide practical teaching advice tailored to Nigerian classrooms
        2. Consider local challenges: large classes, limited resources, power issues
        3. Maintain cultural sensitivity and knowledge of Nigerian education
        4. Offer emotional support while remaining professional
        5. Use conversation history for personalized, contextual responses
        6. Reference previous conversations naturally with phrases like "welcome back", "as we discussed before"
        
        User Information:
        - First Name: {first_name}
        - Location: {location}
        - Teaching Class: {class_taught}
        - Conversation Context: {conversation_context}
        - Total Previous Messages: {total_messages}
        
        Relevant Information:
        {rag_content}
        
        Recent Conversation Summary:
        {conversation_summary}
        
        Guidelines:
        - Always use the user's first name when appropriate
        - Reference previous conversations when relevant
        - For returning users, use welcoming phrases like "Welcome back [name]!"
        - Keep responses practical and Nigeria-focused
        - Be encouraging and supportive
        - Keep responses concise and actionable
        """
    
    def get_rag_content(self, user_message: str, intent: str = None) -> Tuple[str, List[str]]:
        """Get relevant content from Pinecone with source tracking"""
        if not self.embed_model or not self.pinecone_index:
            return "Teaching knowledge base temporarily unavailable.", []
            
        try:
            enhanced_query = user_message
            if intent:
                enhanced_query = f"{intent} {user_message}"
            
            query_embed = self.embed_model.embed_query(enhanced_query)
            query_embed = [float(val) for val in query_embed]

            results = self.pinecone_index.query(
                vector=query_embed,
                top_k=5,
                include_values=False,
                include_metadata=True
            )

            doc_contents = []
            sources = []
            
            for match in results.get('matches', []):
                if match.get('score', 0) > 0.7:
                    text = match['metadata'].get('text', '')
                    source = match['metadata'].get('source', f"Document {match.get('id', 'Unknown')}")
                    
                    if text:
                        doc_contents.append(f"From {source}: {text}")
                        sources.append(source)

            return "\n\n".join(doc_contents) if doc_contents else "No relevant information found.", sources
        
        except Exception as e:
            logger.error(f"Error getting RAG content: {e}")
            return "Teaching knowledge base temporarily unavailable.", []
    
    def create_conversation_summary(self, history: List[Dict], limit: int = 6) -> str:
        """Create a concise summary of recent conversation"""
        if not history:
            return "This is a new conversation."
        
        recent_messages = history[-limit:]
        summary_parts = []
        
        for entry in recent_messages:
            role = "Teacher" if entry['message_type'] == 'user' else "Coach bot"
            content = entry['message_content'][:100] + "..." if len(entry['message_content']) > 100 else entry['message_content']
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def generate_response(self, user_message: str, user_profile: Dict, 
                         conversation_history: List[Dict]) -> Tuple[str, str, List[str]]:
        """Generate contextual response with personalization"""
        if not self.llm:
            return ("I'm currently unable to process your message. Please try again later."), "error", []
            
        try:
            intent = self.analyzer.extract_intent(user_message)
            context = self.analyzer.analyze_conversation_context(conversation_history)
            rag_content, sources = self.get_rag_content(user_message, intent)
            conversation_summary = self.create_conversation_summary(conversation_history)
            
            # Prepare welcome back message for returning users
            welcome_phrase = ""
            if user_profile.get('total_messages', 0) > 1 and context['context'] != 'early_conversation':
                if user_profile.get('first_name'):
                    welcome_phrase = f"Welcome back, {user_profile['first_name']}! "
                else:
                    welcome_phrase = "Welcome back! "
            
            enhanced_prompt = self.system_prompt_template.format(
                first_name=user_profile.get('first_name', 'Teacher'),
                location=user_profile.get('location', 'Nigeria'),
                class_taught=user_profile.get('class_taught', 'your class'),
                conversation_context=context,
                total_messages=user_profile.get('total_messages', 0),
                rag_content=rag_content,
                conversation_summary=conversation_summary
            )
            
            messages = [
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": f"Current question: {user_message}"}
            ]
            
            response = self.llm.invoke(messages)
            ai_response = response.content.strip()
            
            # Clean response and add welcome phrase if appropriate
            ai_response = ai_response.replace("*", "").strip()
            if welcome_phrase and not ai_response.lower().startswith(('hi', 'hello', 'welcome')):
                ai_response = welcome_phrase + ai_response
            
            # Add contextual elements
            if context['context'] == 'support_needed':
                ai_response += "\n\nRemember, self-care is essential for effective teaching. You're doing important work!"
            
            return ai_response, intent, sources
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return ("I'm having trouble processing your request right now. Please try again in a moment."), "error", []

def validate_email_format(email: str) -> bool:
    """Validate email format"""
    try:
        # Basic email validation regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    except:
        return False

def validate_phone_number(phone: str) -> bool:
    """Basic phone number validation for Nigerian numbers"""
    # Remove common separators and spaces
    clean_phone = re.sub(r'[\s\-\(\)]', '', phone)
    
    # Check for Nigerian phone patterns
    nigerian_patterns = [
        r'^\+234[789]\d{9}$',  # +234 followed by 7,8,9 and 9 digits
        r'^0[789]\d{9}$',      # 0 followed by 7,8,9 and 9 digits
        r'^[789]\d{9}$',       # 7,8,9 followed by 9 digits
        r'^\+234[1-9]\d{8,9}$' # Other Nigerian formats
    ]
    
    return any(re.match(pattern, clean_phone) for pattern in nigerian_patterns)

def extract_profile_info(message: str, step: int) -> Tuple[Optional[str], bool]:
    """Extract profile information based on registration step"""
    message = message.strip()
    
    if step == 1:  # First name
        if re.match(r'^[a-zA-Z\s]{2,30}$', message) and len(message.split()) <= 3:
            return message.title(), True
        return None, False
    
    elif step == 2:  # Email
        if validate_email_format(message):
            return message.lower(), True
        return None, False
    
    elif step == 3:  # Phone number
        if validate_phone_number(message):
            return message, True
        return None, False
    
    elif step == 4:  # Location
        # More flexible location validation
        if len(message) >= 3 and len(message) <= 50:
            return message.title(), True
        return None, False
    
    elif step == 5:  # Teaching class
        # Accept various teaching class formats
        if len(message) >= 2 and len(message) <= 50:
            return message.title(), True
        return None, False
    
    return None, False

# Initialize components
db_manager = DatabaseManager(DB_PATH)
sheets_manager = GoogleSheetsManager(GOOGLE_SHEETS_CREDENTIALS_FILE, SPREADSHEET_ID)
ai_coach = EnhancedTeacherAI(llm, embed_model, pinecone_index, db_manager)

def process_message(chat_id: str, user_message: str) -> str:
    """Enhanced message processing with registration flow and Google Sheets integration"""
    try:
        # Get or create user profile
        user_profile = db_manager.get_user_profile(chat_id)
        
        if not user_profile:
            # Create new user
            db_manager.create_or_update_user(chat_id, registration_step=0)
            user_profile = db_manager.get_user_profile(chat_id)
            db_manager.log_analytics(chat_id, "new_user", {"first_message": user_message})
        
        # Handle registration flow
        if not user_profile.get('profile_complete', False):
            current_step = user_profile.get('registration_step', 0)
            
            if current_step == 0:
                # Welcome message
                db_manager.create_or_update_user(chat_id, registration_step=1)
                return (
                    "Hello! I'm Coach bot, your AI teaching assistant for Nigerian schools.\n\n"
                    "To provide personalized support, I need some information from you.\n\n"
                    "Let's start with your first name:"
                )
            
            elif current_step == 1:
                # Collect first name
                first_name, is_valid = extract_profile_info(user_message, 1)
                if is_valid:
                    db_manager.create_or_update_user(chat_id, first_name=first_name, registration_step=2)
                    return f"Nice to meet you, {first_name}! Now, please share your email address:"
                else:
                    return "Please enter a valid first name (letters only, 2-30 characters):"
            
            elif current_step == 2:
                # Collect email
                email, is_valid = extract_profile_info(user_message, 2)
                if is_valid:
                    db_manager.create_or_update_user(chat_id, email=email, registration_step=3)
                    return "Thank you! Now, please share your phone number:"
                else:
                    return "Please enter a valid email address (e.g., john@example.com):"
            
            elif current_step == 3:
                # Collect phone number
                phone, is_valid = extract_profile_info(user_message, 3)
                if is_valid:
                    db_manager.create_or_update_user(chat_id, phone_number=phone, registration_step=4)
                    return "Great! Now, please tell me your location (e.g., Lagos, Nigeria):"
                else:
                    return "Please enter a valid phone number (Nigerian format, e.g., +2348012345678 or 08012345678):"
            
            elif current_step == 4:
                # Collect location
                location, is_valid = extract_profile_info(user_message, 4)
                if is_valid:
                    db_manager.create_or_update_user(chat_id, location=location, registration_step=5)
                    return "Finally, what class do you teach? (e.g., Primary 3, JSS 1, SS 2, etc.):"
                else:
                    return "Please enter a valid location (e.g., Ibadan, Oyo State or Lagos, Nigeria):"
            
            elif current_step == 5:
                # Collect teaching class and complete registration
                class_taught, is_valid = extract_profile_info(user_message, 5)
                if is_valid:
                    # Complete registration
                    db_manager.create_or_update_user(
                        chat_id, 
                        class_taught=class_taught, 
                        registration_step=6,
                        profile_complete=True
                    )
                    
                    # Get updated profile
                    user_profile = db_manager.get_user_profile(chat_id)
                    
                    # Save registration completion message
                    db_manager.save_message(chat_id, 'user', user_message, intent='registration_completion')
                    
                    # Save to Google Sheets
                    if sheets_manager.is_enabled:
                        sheets_manager.save_user_to_sheets(user_profile)
                    
                    # Generate welcome message
                    first_name = user_profile.get('first_name', '')
                    location = user_profile.get('location', '')
                    
                    welcome_msg = (
                        f"Perfect! Welcome to Coach bot, {first_name}!\n\n"
                        f"I see you're teaching {class_taught} in {location}. I'm here to support you with:\n\n"
                        f"• Teaching strategies and methods\n"
                        f"• Classroom management tips\n"
                        f"• Student assessment guidance\n"
                        f"• Teacher wellbeing and stress management\n"
                        f"• Parent communication advice\n"
                        f"• Curriculum planning support\n\n"
                        f"How can I help you today?"
                    )
                    
                    # Save welcome response
                    db_manager.save_message(chat_id, 'assistant', welcome_msg)
                    
                    # Save to Google Sheets
                    if sheets_manager.is_enabled:
                        sheets_manager.save_conversation_to_sheets({
                            'chat_id': chat_id,
                            'first_name': first_name,
                            'message_type': 'assistant',
                            'message_content': welcome_msg,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Log registration completion
                    db_manager.log_analytics(chat_id, "registration_completed", user_profile)
                    
                    return welcome_msg
                else:
                    return "Please enter the class you teach (e.g., Primary 1, JSS 2, SS 1, Nursery, etc.):"
        
        # Profile is complete - process normal conversation
        # Update user interaction
        db_manager.create_or_update_user(chat_id)
        
        # Get conversation history
        conversation_history = db_manager.get_conversation_history(chat_id)
        
        # Extract intent
        intent = ConversationAnalyzer.extract_intent(user_message)
        
        # Save user message
        db_manager.save_message(chat_id, 'user', user_message, intent=intent)
        
        # Save user message to Google Sheets
        if sheets_manager.is_enabled:
            sheets_manager.save_conversation_to_sheets({
                'chat_id': chat_id,
                'first_name': user_profile.get('first_name', ''),
                'message_type': 'user',
                'message_content': user_message,
                'intent': intent,
                'timestamp': datetime.now().isoformat()
            })
        
        # Generate AI response
        ai_response, response_intent, rag_sources = ai_coach.generate_response(
            user_message, user_profile, conversation_history
        )
        
        # Save AI response
        db_manager.save_message(
            chat_id, 'assistant', ai_response, 
            intent=response_intent, rag_sources=json.dumps(rag_sources)
        )
        
        # Save AI response to Google Sheets
        if sheets_manager.is_enabled:
            sheets_manager.save_conversation_to_sheets({
                'chat_id': chat_id,
                'first_name': user_profile.get('first_name', ''),
                'message_type': 'assistant',
                'message_content': ai_response,
                'intent': response_intent,
                'rag_sources': ', '.join(rag_sources) if rag_sources else '',
                'timestamp': datetime.now().isoformat()
            })
        
        # Log interaction analytics
        db_manager.log_analytics(chat_id, "message_processed", {
            "intent": intent,
            "response_length": len(ai_response),
            "rag_sources_count": len(rag_sources)
        })
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Error processing message for {chat_id}: {e}")
        return (
            "I'm experiencing some technical difficulties right now. "
            "Please try again in a moment. If this continues, please let me know!"
        )

def send_message_via_green_api(chat_id: str, message: str) -> bool:
    """Send message via Green API with improved error handling"""
    try:
        url = f"https://api.green-api.com/waInstance{GREEN_API_ID_INSTANCE}/sendMessage/{GREEN_API_TOKEN}"
        payload = {
            "chatId": chat_id,
            "message": message
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            logger.info(f"Message sent successfully to {chat_id}")
            return True
        else:
            logger.error(f"Failed to send message: {response.status_code} - {response.text}")
            return False
            
    except requests.Timeout:
        logger.error(f"Timeout sending message to {chat_id}")
        return False
    except Exception as e:
        logger.error(f"Error sending message via Green API: {e}")
        return False

# Flask Routes
@app.route('/')
def health_check():
    """Enhanced health check with system status"""
    try:
        stats = db_manager.get_stats()
        sheets_status = "connected" if sheets_manager.is_enabled else "not connected"
        
        return jsonify({
            "status": "healthy",
            "service": "Nigerian Teaching Coach Bot - Enhanced",
            "timestamp": datetime.now().isoformat(),
            "database_status": "connected",
            "google_sheets_status": sheets_status,
            "pinecone_status": "connected" if pinecone_index else "not connected",
            "llm_status": "connected" if llm else "not connected",
            "stats": stats
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Enhanced webhook handler with better error handling"""
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("Webhook received empty data")
            return jsonify({"status": "error", "message": "No data received"}), 400

        # Handle incoming message
        if data.get('typeWebhook') == 'incomingMessageReceived':
            message_data = data.get('messageData', {})
            sender_data = data.get('senderData', {})
            
            chat_id = sender_data.get('chatId', '').strip()
            
            # Handle text messages only
            if 'textMessageData' in message_data and chat_id:
                text_data = message_data.get('textMessageData', {})
                user_message = text_data.get('textMessage', '').strip()
                
                if user_message:
                    logger.info(f"Processing message from {chat_id}: {user_message[:100]}...")
                    
                    # Process message in separate thread to avoid blocking
                    def process_and_respond():
                        try:
                            reply = process_message(chat_id, user_message)
                            if reply:
                                success = send_message_via_green_api(chat_id, reply)
                                if not success:
                                    logger.error(f"Failed to send reply to {chat_id}")
                                    # Retry once after a short delay
                                    time.sleep(2)
                                    send_message_via_green_api(chat_id, "Sorry, I had trouble sending my response. Let me try again.")
                        except Exception as e:
                            logger.error(f"Error in process_and_respond for {chat_id}: {e}")
                            try:
                                error_msg = "I encountered an error processing your message. Please try again!"
                                send_message_via_green_api(chat_id, error_msg)
                            except:
                                logger.error(f"Failed to send error message to {chat_id}")
                    
                    # Run in background thread
                    thread = threading.Thread(target=process_and_respond)
                    thread.daemon = True
                    thread.start()
                    
                    return jsonify({"status": "success", "message": "Message being processed"}), 200
                else:
                    logger.warning(f"Empty message received from {chat_id}")

        return jsonify({"status": "success", "message": "Webhook received"}), 200

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/status')
def bot_status():
    """Enhanced bot status with detailed analytics"""
    try:
        stats = db_manager.get_stats()
        
        return jsonify({
            "bot_status": "running",
            "database_status": "connected",
            "google_sheets_status": "connected" if sheets_manager.is_enabled else "not connected",
            "pinecone_status": "connected" if pinecone_index else "not connected",
            "llm_status": "connected" if llm else "not connected",
            "green_api_instance": GREEN_API_ID_INSTANCE,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        return jsonify({
            "bot_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/test', methods=['POST'])
def test_message():
    """Enhanced test endpoint for development"""
    try:
        data = request.get_json()
        chat_id = data.get('chat_id', 'test_user')
        message = data.get('message', 'Hello')
        
        # Process message
        response = process_message(chat_id, message)
        
        # Get user profile and conversation history
        user_profile = db_manager.get_user_profile(chat_id)
        conversation_history = db_manager.get_conversation_history(chat_id, limit=5)
        
        return jsonify({
            "status": "success",
            "chat_id": chat_id,
            "user_message": message,
            "bot_response": response,
            "user_profile": user_profile,
            "recent_history": conversation_history,
            "registration_step": user_profile.get('registration_step', 0) if user_profile else 0
        })
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/user/<chat_id>', methods=['GET'])
def get_user_info(chat_id):
    """Get detailed user information"""
    try:
        user_profile = db_manager.get_user_profile(chat_id)
        if not user_profile:
            return jsonify({"status": "error", "message": "User not found"}), 404
        
        conversation_history = db_manager.get_conversation_history(chat_id)
        
        return jsonify({
            "status": "success",
            "user_profile": user_profile,
            "conversation_count": len(conversation_history),
            "recent_messages": conversation_history[-5:] if conversation_history else [],
            "registration_complete": user_profile.get('profile_complete', False)
        })
        
    except Exception as e:
        logger.error(f"Error getting user info for {chat_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/user/<chat_id>/reset', methods=['POST'])
def reset_user(chat_id):
    """Reset user profile and conversation history"""
    try:
        user_profile = db_manager.get_user_profile(chat_id)
        if not user_profile:
            return jsonify({"status": "error", "message": "User not found"}), 404
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Reset user profile
            cursor.execute('''
                UPDATE users SET 
                    first_name = NULL,
                    email = NULL,
                    phone_number = NULL,
                    location = NULL,
                    teacher_name = NULL,
                    class_taught = NULL,
                    profile_complete = FALSE,
                    first_interaction = TRUE,
                    registration_step = 0,
                    total_messages = 0
                WHERE chat_id = ?
            ''', (chat_id,))
            
            # Clear conversation history
            cursor.execute('DELETE FROM conversations WHERE chat_id = ?', (chat_id,))
            
            # Log the reset action
            cursor.execute('''
                INSERT INTO analytics (chat_id, action_type, action_data)
                VALUES (?, ?, ?)
            ''', (chat_id, "user_reset", json.dumps({"reset_by": "admin", "timestamp": datetime.now().isoformat()})))
            
            conn.commit()
        
        logger.info(f"User {chat_id} reset successfully")
        return jsonify({
            "status": "success", 
            "message": f"User {chat_id} reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error resetting user {chat_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/sheets/sync', methods=['POST'])
def sync_to_sheets():
    """Manually sync all user data to Google Sheets"""
    try:
        if not sheets_manager.is_enabled:
            return jsonify({"status": "error", "message": "Google Sheets not configured"}), 400
        
        # Get all users
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE profile_complete = 1')
            users = [dict(row) for row in cursor.fetchall()]
        
        success_count = 0
        for user in users:
            if sheets_manager.save_user_to_sheets(user):
                success_count += 1
        
        return jsonify({
            "status": "success",
            "message": f"Synced {success_count} out of {len(users)} users to Google Sheets"
        })
        
    except Exception as e:
        logger.error(f"Error syncing to sheets: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def initialize_database_with_sample_data():
    """Initialize database with sample teaching resources if empty"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM users')
            user_count = cursor.fetchone()['count']
            
        if user_count == 0:
            logger.info("Initializing database with sample data...")
            
            # Create a sample registered user for testing
            sample_chat_id = "sample_user_001"
            db_manager.create_or_update_user(
                sample_chat_id, 
                first_name="Sarah",
                email="sarah@example.com",
                phone_number="+2348012345678",
                location="Lagos, Nigeria",
                class_taught="Primary 5",
                registration_step=6,
                profile_complete=True
            )
            
            # Add sample conversation
            db_manager.save_message(
                sample_chat_id, 'user', 
                "How can I manage a large class of 50 students?", 
                intent='classroom_management'
            )
            
            db_manager.save_message(
                sample_chat_id, 'assistant',
                "Welcome back, Sarah! Managing a large class requires strategic planning. Here are some effective techniques for Nigerian classrooms...",
                intent='classroom_management'
            )
            
            logger.info("Sample data initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")

if __name__ == "__main__":
    logger.info("Starting Enhanced Nigerian Teaching Coach Bot with Registration Flow...")
    logger.info(f"Green API Instance: {GREEN_API_ID_INSTANCE}")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Google Sheets: {'Configured' if sheets_manager.is_enabled else 'Not Configured'}")
    logger.info(f"Pinecone: {'Connected' if pinecone_index else 'Not Connected'}")
    logger.info(f"LLM: {'Connected' if llm else 'Not Connected'}")
    
    # Initialize database with sample data if needed
    initialize_database_with_sample_data()
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting server on port {port}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True) 
