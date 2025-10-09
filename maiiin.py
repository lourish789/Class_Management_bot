import os
import time
import re
from datetime import datetime
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
    'INDEX_NAME': 'coach'
}

#CONFIG = {
  #  'PINECONE_API_KEY': os.getenv("PINECONE_API_KEY", ""),
 #   'GOOGLE_API_KEY': os.getenv("GOOGLE_API_KEY", ""),
  #  'GREEN_API_ID': os.getenv("GREEN_API_ID_INSTANCE", ""),
   # 'GREEN_API_TOKEN': os.getenv("GREEN_API_TOKEN", ""),
  #  'APPS_SCRIPT_URL': os.getenv("APPS_SCRIPT_URL", ""),
  #  'DB_PATH': "ai_coach.db",
  #  'MAX_HISTORY': 20,
  #  'INDEX_NAME': 'coach'
#}

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
    # Initialize Pinecone
    logger.info("Initializing Pinecone...")
    pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
    
    # Connect to existing 'coach' index
    try:
        pinecone_index = pc.Index(CONFIG['INDEX_NAME'])
        stats = pinecone_index.describe_index_stats()
        logger.info(f"✓ Connected to Pinecone index '{CONFIG['INDEX_NAME']}' - Vectors: {stats.get('total_vector_count', 0)}")
    except Exception as e:
        logger.error(f"✗ Could not connect to Pinecone index '{CONFIG['INDEX_NAME']}': {e}")
        logger.warning("Continuing without RAG support")
    
    # Initialize Google AI
    logger.info("Initializing Google AI services...")
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=CONFIG['GOOGLE_API_KEY']
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=CONFIG['GOOGLE_API_KEY'],
        temperature=0.6,
        max_tokens=1000
    )
    logger.info("✓ Google AI services initialized")
    
except Exception as e:
    logger.error(f"✗ Critical initialization error: {e}")
    raise


class DatabaseManager:
    """Handles all SQLite database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    chat_id TEXT PRIMARY KEY,
                    first_name TEXT,
                    email TEXT,
                    phone_number TEXT,
                    location TEXT,
                    class_taught TEXT,
                    profile_complete BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER DEFAULT 0
                )
            ''')
            
            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    message_type TEXT CHECK(message_type IN ('user', 'assistant')),
                    message_content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    intent TEXT,
                    FOREIGN KEY (chat_id) REFERENCES users (chat_id)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_id ON conversations (chat_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
            
            conn.commit()
            logger.info("✓ Database initialized")
    
    @contextmanager
    def get_conn(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_user(self, chat_id: str) -> Optional[Dict]:
        """Get user by chat_id"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE chat_id = ?', (chat_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def save_user(self, chat_id: str, **kwargs) -> bool:
        """Save or update user"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT chat_id FROM users WHERE chat_id = ?', (chat_id,))
                exists = cursor.fetchone()
                
                if exists:
                    if kwargs:
                        updates = ', '.join([f'{k} = ?' for k in kwargs.keys()])
                        updates += ', last_interaction = ?, total_messages = total_messages + 1'
                        values = list(kwargs.values()) + [datetime.now(), chat_id]
                        cursor.execute(f'UPDATE users SET {updates} WHERE chat_id = ?', values)
                    else:
                        cursor.execute(
                            'UPDATE users SET last_interaction = ?, total_messages = total_messages + 1 WHERE chat_id = ?',
                            (datetime.now(), chat_id)
                        )
                else:
                    fields = ['chat_id'] + list(kwargs.keys()) + ['total_messages']
                    placeholders = ', '.join(['?' for _ in fields])
                    values = [chat_id] + list(kwargs.values()) + [1]
                    cursor.execute(
                        f'INSERT INTO users ({", ".join(fields)}) VALUES ({placeholders})', 
                        values
                    )
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving user {chat_id}: {e}")
            return False
    
    def save_message(self, chat_id: str, msg_type: str, content: str, intent: str = None):
        """Save conversation message"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (chat_id, message_type, message_content, intent)
                    VALUES (?, ?, ?, ?)
                ''', (chat_id, msg_type, content, intent))
                conn.commit()
                self._cleanup_history(chat_id)
        except Exception as e:
            logger.error(f"Error saving message: {e}")
    
    def get_history(self, chat_id: str, limit: int = None) -> List[Dict]:
        """Get conversation history"""
        limit = limit or CONFIG['MAX_HISTORY']
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_type, message_content, timestamp, intent
                    FROM conversations 
                    WHERE chat_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (chat_id, limit))
                return [dict(row) for row in reversed(cursor.fetchall())]
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    def _cleanup_history(self, chat_id: str):
        """Clean up old messages beyond max history"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM conversations 
                    WHERE chat_id = ? AND id NOT IN (
                        SELECT id FROM conversations 
                        WHERE chat_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    )
                ''', (chat_id, chat_id, CONFIG['MAX_HISTORY'] * 2))
                conn.commit()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


class ResponseFormatter:
    """Format responses according to WhatsApp guidelines"""
    
    @staticmethod
    def clean_response(text: str) -> str:
        """Remove markdown and format for WhatsApp"""
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code
        text = re.sub(r'#{1,6}\s', '', text)  # Headers
        text = text.replace('**', '').replace('*', '')
        
        # Convert bullets to numbers
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
            
            # Check for bullet points
            if re.match(r'^[\-•·]\s+', stripped):
                content = re.sub(r'^[\-•·]\s+', '', stripped)
                formatted_lines.append(f"{list_counter}. {content}")
                list_counter += 1
                in_list = True
            else:
                formatted_lines.append(stripped)
                if not stripped[0].isdigit():
                    in_list = False
        
        # Join and clean up spacing
        result = '\n'.join(formatted_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()

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
        """
    
    def get_rag_content(self, user_message: str, intent: str = None) -> Tuple[str, List[str]]:
        """Get relevant content from Pinecone with source tracking"""
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
            return "No relevant information found.", []
    
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
            return ("I'm having trouble right now. Please try again in a moment."), "error", []

       
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Extract intent from message"""
        msg = message.lower()
        
        intents = {
            'teaching_strategy': ['teach', 'strategy', 'method', 'lesson', 'explain', 'introduce', 'activity', 'engage'],
            'classroom_management': ['discipline', 'behavior', 'manage', 'control', 'disruptive', 'noise', 'attention'],
            'assessment': ['assess', 'evaluate', 'grade', 'test', 'exam', 'mark', 'feedback', 'progress'],
            'wellbeing': ['stress', 'tired', 'overwhelmed', 'burnout', 'exhausted', 'frustrated', 'difficult'],
            'curriculum': ['curriculum', 'syllabus', 'topic', 'subject', 'scheme of work'],
            'parent_communication': ['parent', 'guardian', 'meeting', 'report'],
            'resources': ['resource', 'material', 'tool', 'equipment', 'aid']
        }
        
        for intent_name, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent_name
        
        return 'general'
                             


class AICoach:
    """AI Coach with RAG and context awareness"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        self.formatter = ResponseFormatter()
    
    def get_rag_content(self, query: str) -> Tuple[str, List[str]]:
        """Retrieve relevant content from Pinecone"""
        if not self.pinecone_index:
            return "", []
        
        try:
            query_embed = self.embed_model.embed_query(query)
            results = self.pinecone_index.query(
                vector=query_embed,
                top_k=3,
                include_metadata=True
            )
            
            contents, sources = [], []
            for match in results.get('matches', []):
                score = match.get('score', 0)
                if score > 0.65:
                    text = match['metadata'].get('text', '')
                    source = match['metadata'].get('source', 'Knowledge Base')
                    if text:
                        contents.append(text[:400])
                        sources.append(source)
            
            if contents:
                logger.info(f"Retrieved {len(contents)} relevant documents")
            
            return '\n\n'.join(contents), sources
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return "", []
    
    def generate_response(self, message: str, user_profile: Dict, 
                         history: List[Dict] = None) -> Tuple[str, str]:
        """Generate formatted AI response"""
        try:
            intent = self._extract_intent(message)
            rag_content, sources = self.get_rag_content(message)
            
            # Build conversation context
            context = ""
            if history:
                recent = history[-10:]
                context_parts = []
                for msg in recent:
                    role = "Teacher" if msg['message_type'] == 'user' else "AI Coach"
                    content = msg['message_content'][:200]
                    context_parts.append(f"{role}: {content}")
                context = "\n".join(context_parts)
            
            # Build system prompt
            system_prompt = self._build_system_prompt(
                user_profile, rag_content, context, intent
            )
            
            # Generate response
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt}
            ])
            
            # Clean and format response
            clean_response = self.formatter.clean_response(response.content)
            
            return clean_response, intent
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._get_fallback_response(intent), "error"
    
    def _build_system_prompt(self, user_profile: Dict, rag_content: str, 
                            context: str, intent: str) -> str:
        """Build comprehensive system prompt"""
        
        name = user_profile.get('first_name', 'Teacher')
        class_info = user_profile.get('class_taught', 'their class')
        location = user_profile.get('location', 'Nigeria')
        
        prompt = f"""You are AI Coach by Schoolinka, helping Nigerian teachers excel in their profession.

    TEACHER PROFILE:
- Name: {name}
- Teaching: {class_info}
- Location: {location}
- Current Intent: {intent}

CORE GUIDELINES:
1. RESPONSE STYLE:
   - Be conversational, warm, and professional
   - Use clear, simple language for text messaging
   - Provide practical advice for Nigerian classrooms
   - Be specific and detailed when explaining

2. FORMATTING RULES (CRITICAL):
   - Use numbers (1. 2. 3.) for lists and steps
   - Start each numbered item on a new line
   - Keep paragraphs short (2-3 sentences max)
   - Add line breaks between sections
   - NO asterisks, bullets, or special characters
   - NO markdown formatting

3. RESPONSE LENGTH:
   - Simple questions: 3-5 sentences
   - How-to questions: Detailed numbered steps
   - Complex topics: 2-3 clear paragraphs
   - Be thorough and helpful

4. CONTENT APPROACH:
   - Address the question directly first
   - Provide context and explanation
   - Give practical Nigerian school examples
   - Consider resource constraints
   - Enumerate steps for processes"""

        if rag_content:
            prompt += f"\n\nKNOWLEDGE BASE:\n{rag_content}"
        
        if context:
            prompt += f"\n\nRECENT CONVERSATION:\n{context}"
        
        prompt += "\n\nProvide a helpful, well-structured response following ALL guidelines."
        
        return prompt
    
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Extract intent from message"""
        msg = message.lower()
        
        intents = {
            'teaching_strategy': ['teach', 'strategy', 'method', 'lesson', 'explain', 'introduce', 'activity', 'engage'],
            'classroom_management': ['discipline', 'behavior', 'manage', 'control', 'disruptive', 'noise', 'attention'],
            'assessment': ['assess', 'evaluate', 'grade', 'test', 'exam', 'mark', 'feedback', 'progress'],
            'wellbeing': ['stress', 'tired', 'overwhelmed', 'burnout', 'exhausted', 'frustrated', 'difficult'],
            'curriculum': ['curriculum', 'syllabus', 'topic', 'subject', 'scheme of work'],
            'parent_communication': ['parent', 'guardian', 'meeting', 'report'],
            'resources': ['resource', 'material', 'tool', 'equipment', 'aid']
        }
        
        for intent_name, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent_name
        
        return 'general'
    
   # @staticmethod
    #def _get_fallback_response(intent: str) -> str:
     #   """Fallback response on error"""
   #     return (
     #       "I'm experiencing a brief technical issue.\n\n"
    #        "Please try asking your question again, and I'll do my best to help you."
   #     )


def log_to_google_sheets(data_type: str, data: Dict) -> bool:
    """Log data to Google Sheets via Apps Script"""
    if not CONFIG['APPS_SCRIPT_URL']:
        logger.warning("Apps Script URL not configured")
        return False
    
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
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Logged {data_type} to Google Sheets")
            return True
        else:
            logger.warning(f"✗ Sheets logging failed: {response.status_code}")
            return False
            
    except requests.Timeout:
        logger.warning("Sheets logging timeout")
        return False
    except Exception as e:
        logger.error(f"Sheets logging error: {e}")
        return False


def parse_registration(text: str) -> Dict:
    """Parse registration details from text"""
    details = {}
    lines = text.strip().split('\n')
    
    for line in lines:
        if ':' not in line:
            continue
        
        try:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if not value or len(value) < 2:
                continue
            
            if 'name' in key:
                clean_name = re.sub(r'[^a-zA-Z\s]', '', value)
                if len(clean_name) > 1:
                    details['first_name'] = clean_name.title()
            
            elif 'email' in key and '@' in value and '.' in value:
                if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value.lower()):
                    details['email'] = value.lower()
            
            elif 'phone' in key:
                phone = re.sub(r'[^\d+]', '', value)
                if len(phone) >= 10:
                    details['phone_number'] = phone
            
            elif 'location' in key and len(value) > 2:
                details['location'] = value.title()
            
            elif 'class' in key and len(value) > 1:
                details['class_taught'] = value.title()
        
        except Exception as e:
            logger.warning(f"Error parsing line '{line}': {e}")
            continue
    
    return details


# Initialize components
db = DatabaseManager(CONFIG['DB_PATH'])
ai_coach = AICoach(llm, embed_model, pinecone_index)


def process_message(chat_id: str, text_message: str) -> str:
    """Main message processing logic"""
    try:
        # Validate input
        if not text_message or len(text_message.strip()) < 2:
            return "Please send me a message or question. I'm here to help!"
        
        text_message = text_message.strip()
        user = db.get_user(chat_id)
        
        # Handle registration flow
        if not user or not user.get('profile_complete'):
            return handle_registration(chat_id, text_message, user)
        
        # Update user activity
        db.save_user(chat_id)
        
        # Get conversation history
        history = db.get_history(chat_id, limit=15)
        
        # Extract intent and save user message
        intent = ai_coach._extract_intent(text_message)
        db.save_message(chat_id, 'user', text_message, intent)
        
        # Generate AI response
        ai_response, response_intent = ai_coach.generate_response(
            text_message, user, history
        )
        
        # Save assistant response
        db.save_message(chat_id, 'assistant', ai_response, response_intent)
        
        # Log to Google Sheets
        log_to_google_sheets('conversation', {
            'chat_id': chat_id,
            'user_name': user.get('first_name', 'Unknown'),
            'class_taught': user.get('class_taught', ''),
            'user_message': text_message[:500],
            'bot_response': ai_response[:500],
            'intent': intent
        })
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Message processing error: {e}", exc_info=True)
        return "I'm experiencing technical difficulties. Please try again in a moment."


def handle_registration(chat_id: str, text: str, user: Optional[Dict]) -> str:
    """Handle user registration flow"""
    
    # First contact - show welcome message
    if not user:
        db.save_user(chat_id, profile_complete=False)
        return (
            "Hello! I'm AI Coach by Schoolinka.\n\n"
            "I'm here to support you with teaching strategies, classroom management, "
            "and professional development.\n\n"
            "To get started, please share your details in one message:\n\n"
            "Name: Your full name\n"
            "Email: Your email address\n"
            "Phone: Your phone number\n"
            "Location: Your city or state\n"
            "Class: The class you teach\n\n"
            "Example:\n"
            "Name: Amina Bello\n"
            "Email: amina@email.com\n"
            "Phone: 08012345678\n"
            "Location: Lagos\n"
            "Class: Primary 4"
        )
    
    # Parse registration attempt
    details = parse_registration(text)
    required = ['first_name', 'email', 'phone_number', 'location', 'class_taught']
    missing = [f for f in required if f not in details]
    
    # Check if registration complete
    if not missing:
        # Save user profile
        db.save_user(chat_id, profile_complete=True, **details)
        user = db.get_user(chat_id)
        
        # Log to Google Sheets
        log_to_google_sheets('user', {
            'chat_id': chat_id,
            'name': details.get('first_name', ''),
            'email': details.get('email', ''),
            'phone': details.get('phone_number', ''),
            'location': details.get('location', ''),
            'class': details.get('class_taught', ''),
            'status': 'Registered'
        })
        
        welcome_msg = (
            f"Welcome, {details['first_name']}!\n\n"
            f"I'm excited to support you with your {details['class_taught']}.\n\n"
            f"I can help you with:\n\n"
            f"1. Teaching strategies and lesson planning\n"
            f"2. Classroom management techniques\n"
            f"3. Assessment and evaluation methods\n"
            f"4. Student engagement activities\n"
            f"5. Parent communication strategies\n"
            f"6. Professional development tips\n\n"
            f"What would you like help with today?"
        )
        
        # Log registration conversation
        log_to_google_sheets('conversation', {
            'chat_id': chat_id,
            'user_name': details['first_name'],
            'class_taught': details['class_taught'],
            'user_message': text[:500],
            'bot_response': welcome_msg[:500],
            'intent': 'registration'
        })
        
        return welcome_msg
    
    # Registration incomplete
    field_names = {
        'first_name': 'Name',
        'email': 'Email',
        'phone_number': 'Phone',
        'location': 'Location',
        'class_taught': 'Class'
    }
    
    missing_names = [field_names[f] for f in missing]
    
    return (
        f"I still need the following details:\n\n"
        f"{', '.join(missing_names)}\n\n"
        f"Please provide all information in this format:\n\n"
        f"Name: Your full name\n"
        f"Email: Your email\n"
        f"Phone: Your phone number\n"
        f"Location: Your city/state\n"
        f"Class: Class you teach"
    )


def send_whatsapp_message(chat_id: str, message: str) -> bool:
    """Send message via Green API"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.green-api.com/waInstance{CONFIG['GREEN_API_ID']}/sendMessage/{CONFIG['GREEN_API_TOKEN']}"
            
            response = requests.post(
                url,
                json={"chatId": chat_id, "message": message},
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Message sent to {chat_id}")
                return True
            else:
                logger.warning(f"Send attempt {attempt + 1} failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Send error (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    return False


# Flask Routes

@app.route('/')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI Coach - Schoolinka",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "pinecone": "connected" if pinecone_index else "not available",
        "apps_script": "configured" if CONFIG['APPS_SCRIPT_URL'] else "not configured"
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages"""
    try:
        data = request.get_json()
        
        # Validate webhook type
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored"}), 200
        
        # Extract message data
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        if not chat_id:
            return jsonify({"status": "no_chat_id"}), 200
        
        # Ignore group messages
        if '@g.us' in chat_id:
            return jsonify({"status": "group_ignored"}), 200
        
        # Extract text message
        text_message = None
        
        if 'textMessageData' in message_data:
            text_message = message_data['textMessageData'].get('textMessage', '').strip()
        elif 'extendedTextMessageData' in message_data:
            text_message = message_data['extendedTextMessageData'].get('text', '').strip()
        
        if not text_message:
            # Non-text message
            send_whatsapp_message(
                chat_id,
                "I can only respond to text messages right now. Please type your question."
            )
            return jsonify({"status": "non_text"}), 200
        
        logger.info(f"Received from {chat_id}: {text_message[:50]}...")
        
        # Process in background
        def process_and_respond():
            try:
                reply = process_message(chat_id, text_message)
                send_whatsapp_message(chat_id, reply)
            except Exception as e:
                logger.error(f"Background processing error: {e}", exc_info=True)
                send_whatsapp_message(
                    chat_id,
                    "Sorry, I encountered an error. Please try again."
                )
        
        thread = threading.Thread(target=process_and_respond)
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "processing"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error"}), 500


@app.route('/user/<chat_id>', methods=['GET'])
def get_user_info(chat_id):
    """Get user profile"""
    try:
        user = db.get_user(chat_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        history = db.get_history(chat_id, limit=10)
        
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
        chat_id = data.get('chat_id', 'test_user')
        message = data.get('message', 'Hello')
        
        response = process_message(chat_id, message)
        user = db.get_user(chat_id)
        
        return jsonify({
            "response": response,
            "user": dict(user) if user else None,
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
            
            cursor.execute('SELECT COUNT(*) as registered FROM users WHERE profile_complete = 1')
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
        "apps_script": False
    }
    
    # Check database
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            components["database"] = True
    except:
        pass
    
    # Check components
    components["pinecone"] = pinecone_index is not None
    components["google_ai"] = llm is not None and embed_model is not None
    components["green_api"] = bool(CONFIG['GREEN_API_ID'] and CONFIG['GREEN_API_TOKEN'])
    components["apps_script"] = bool(CONFIG['APPS_SCRIPT_URL'])
    
    all_ok = components["database"] and components["google_ai"]
    
    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "components": components,
        "config": {
            "index_name": CONFIG['INDEX_NAME'],
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
    logger.info("AI COACH - SCHOOLINKA")
    logger.info("=" * 70)
    logger.info(f"Database: {CONFIG['DB_PATH']}")
    logger.info(f"Pinecone Index: {CONFIG['INDEX_NAME']} - {'Connected' if pinecone_index else 'Not Available'}")
    logger.info(f"Google AI: {'Initialized' if llm else 'Failed'}")
    logger.info(f"Apps Script: {'Configured' if CONFIG['APPS_SCRIPT_URL'] else 'Not Configured'}")
    logger.info("=" * 70)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
