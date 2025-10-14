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


class DatabaseManager:
    """Handles all SQLite database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
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
    
    def get_last_assistant_messages(self, chat_id: str, num_messages: int = 3) -> List[Dict]:
        """Get last N assistant messages (responses from bot)"""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_content, timestamp, intent
                    FROM conversations 
                    WHERE chat_id = ? AND message_type = 'assistant'
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (chat_id, num_messages))
                return [dict(row) for row in reversed(cursor.fetchall())]
        except Exception as e:
            logger.error(f"Error getting last assistant messages: {e}")
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
        
        # Determine followup type
        followup_type = 'clarification'
        if any(re.search(p, msg_lower) for p in [r'\blist\b', r'\bitems?\b', r'\bfunctions?\b']):
            followup_type = 'list_expansion'
        elif any(re.search(p, msg_lower) for p in [r'\bexample\b', r'\bsteps?\b']):
            followup_type = 'detailed_breakdown'
        
        return is_followup, followup_type
    
    def get_rag_content(self, query: str, intent: str = None) -> Tuple[str, List[str]]:
        """Retrieve relevant content from Pinecone"""
        if not self.pinecone_index:
            logger.warning("Pinecone index not available")
            return "", []
        
        try:
            enhanced_query = f"{intent} {query}" if intent else query
            logger.info(f"RAG Query: {enhanced_query[:100]}")
            
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
            else:
                logger.warning("No relevant documents found above threshold")
                return "", []
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}", exc_info=True)
            return "", []
    
    def generate_response(self, message: str, user_profile: Dict, 
                         history: List[Dict] = None) -> Tuple[str, str]:
        """Generate formatted AI response with optimized speed"""
        try:
            intent = self._extract_intent(message)
            logger.info(f"Detected intent: {intent}")
            
            # Check if this is a follow-up question
            is_followup, followup_type = self.detect_followup_intent(message, history or [])
            
            # Get RAG content in parallel for speed (skip if follow-up)
            rag_content = ""
            sources = []
            if not is_followup or followup_type == 'clarification':
                rag_content, sources = self.get_rag_content(message, intent)
            
            # Build conversation context
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
            
            # For follow-ups, include last bot responses
            if is_followup:
                last_responses = self._get_last_bot_responses(history, num=3)
                if last_responses:
                    assistant_context = f"PREVIOUS BOT RESPONSES:\n{last_responses}\n\n"
                    logger.info(f"Follow-up detected ({followup_type}) - Including previous responses")
            
            # Build system prompt
            system_prompt = self._build_system_prompt(
                user_profile, rag_content, context, intent, message, 
                is_followup, followup_type, assistant_context
            )
            
            logger.info("Generating AI response...")
            
            # Generate response with timeout
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            response = self.llm.invoke(messages)
            clean_response = self.formatter.clean_response(response.content)
            
            logger.info(f"✓ Response generated: {len(clean_response)} chars")
            
            return clean_response, intent
            
        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            return self._get_fallback_response(intent), "error"
    
    def _get_last_bot_responses(self, history: List[Dict], num: int = 3) -> str:
        """Extract last N bot responses from history for context"""
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
        """Build comprehensive system prompt with follow-up support"""
        
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
                followup_instruction = "\nFOLLOW-UP DETECTED: User is asking for lists/items related to previous response. Provide a comprehensive numbered list with brief descriptions."
            elif followup_type == 'detailed_breakdown':
                followup_instruction = "\nFOLLOW-UP DETECTED: User wants more details or breakdown. Expand significantly with steps, examples, and detailed explanations."
            else:
                followup_instruction = "\nFOLLOW-UP DETECTED: User is seeking clarification. Reference your previous response and clarify the points they're asking about."
        
        prompt = f"""You are AI Coach by Schoolinka, a friendly and knowledgeable teaching assistant for Nigerian teachers
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
This chatbot is a product of schoolinka. Schoolinka was founded by Oluwaseun Kayode. Schoolinka is an integrated online platform designed to offer training courses, certifications, and teaching resources for educators, while also providing a job board for teachers seeking opportunities in renowned schools worldwide.
Try to explain about the company if a user asks. For more information, refer the user to the Schoolinka web link: https://www.schoolinka.com/

CRITICAL FOLLOW-UP HANDLING:
- When user references "previous response", "last answer", or similar: Reference it explicitly and expand
- When asked to "list" items: Provide ALL items as a comprehensive numbered list with descriptions
- When asked to "elaborate" or "explain more": Provide 2-3x more detail than initial response
- Always maintain context from previous exchanges
- If user asks about specific items from your previous response: Acknowledge and focus on those

RESPONSE GUIDELINES:
1. RESPONSE STYLE:
   - Be warm, conversational, and encouraging
   - Do not call the user's name all the time when responding
   - Do not always say "welcome back" or "it's good to have you back" in all responses
   - Provide practical, Nigeria-specific advice
   - Be detailed and thorough in explanations
   - Show empathy and understanding

2. FORMATTING RULES (CRITICAL):
   - Use numbers (1. 2. 3.) for all lists and steps
   - Start each numbered item on a new line
   - Keep paragraphs short (2-3 sentences max)
   - Add line breaks between sections
   - NO asterisks, bullets (*), or markdown
   - NO special formatting characters

3. RESPONSE LENGTH:
   - Simple questions: 4-6 sentences
   - How-to questions: Detailed numbered steps with explanations
   - Complex topics: 3-4 well-structured paragraphs
   - Follow-ups: Provide more detail than initial response
   - Always be thorough and helpful

4. NIGERIAN CONTEXT:
   - Consider large class sizes (30-60 students)
   - Address limited resources and materials
   - Account for power supply challenges
   - Reference local curriculum standards
   - Use relevant Nigerian examples"""
        
        if rag_content:
            prompt += f"\n\nRELEVANT KNOWLEDGE BASE INFORMATION:\n{rag_content}\n(Use this information to enhance your response with accurate, up-to-date details)"
        
        if context:
            prompt += f"\n\nRECENT CONVERSATION HISTORY:\n{context}\n(Reference previous discussions naturally when relevant)"
        
        prompt += "\n\nProvide a helpful, well-formatted response that directly addresses the teacher's question. Be specific, practical, and encouraging."
        
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
        return (
            "I'm experiencing a brief technical issue.\n\n"
            "Please try asking your question again, and I'll do my best to help you."
        )


def log_to_google_sheets(data_type: str, data: Dict) -> bool:
    """Log data to Google Sheets via Apps Script (non-blocking)"""
    if not CONFIG['APPS_SCRIPT_URL']:
        logger.warning("Apps Script URL not configured")
        return False
    
    def _log():
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
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Logged {data_type} to Google Sheets")
            else:
                logger.warning(f"✗ Sheets logging failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Sheets logging error: {e}")
    
    # Run in background to not slow down response
    thread = threading.Thread(target=_log, daemon=True)
    thread.start()
    return True


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
    """Main message processing logic with optimized speed"""
    try:
        if not text_message or len(text_message.strip()) < 2:
            return "Please send me a message or question. I'm here to help!"
        
        text_message = text_message.strip()
        user = db.get_user(chat_id)
        
        if not user or not user.get('profile_complete'):
            return handle_registration(chat_id, text_message, user)
        
        db.save_user(chat_id)
        
        history = db.get_history(chat_id, limit=15)
        
        intent = ai_coach._extract_intent(text_message)
        db.save_message(chat_id, 'user', text_message, intent)
        
        logger.info(f"Processing message from {user.get('first_name', 'Unknown')} - Intent: {intent}")
        
        ai_response, response_intent = ai_coach.generate_response(
            text_message, user, history
        )
        
        db.save_message(chat_id, 'assistant', ai_response, response_intent)
        
        # Log to sheets in background (non-blocking)
        log_to_google_sheets('conversation', {
            'chat_id': chat_id,
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
        logger.error(f"Message processing error: {e}", exc_info=True)
        return "I'm experiencing technical difficulties. Please try again in a moment."


def handle_registration(chat_id: str, text: str, user: Optional[Dict]) -> str:
    """Handle user registration flow"""
    
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
    
    details = parse_registration(text)
    required = ['first_name', 'email', 'phone_number', 'location', 'class_taught']
    missing = [f for f in required if f not in details]
    
    if not missing:
        db.save_user(chat_id, profile_complete=True, **details)
        user = db.get_user(chat_id)
        
        log_to_google_sheets('user', {
            'chat_id': chat_id,
            'name': details.get('first_name', ''),
            'email': details.get('email', ''),
            'phone': details.get('phone_number', ''),
            'location': details.get('location', ''),
            'class': details.get('class_taught', ''),
            'status': 'Registered',
            'registration_date': datetime.now().isoformat()
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
        
        log_to_google_sheets('conversation', {
            'chat_id': chat_id,
            'user_name': details['first_name'],
            'class_taught': details['class_taught'],
            'user_message': text[:500],
            'bot_response': welcome_msg[:500],
            'intent': 'registration'
        })
        
        return welcome_msg
    
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
    """Send message via Green API with retry logic"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.green-api.com/waInstance{CONFIG['GREEN_API_ID']}/sendMessage/{CONFIG['GREEN_API_TOKEN']}"
            
            response = requests.post(
                url,
                json={"chatId": chat_id, "message": message},
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Message sent to {chat_id}")
                return True
            else:
                logger.warning(f"Send attempt {attempt + 1} failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Send error (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(1)
    
    return False


# Flask Routes

@app.route('/')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI Coach - Schoolinka",
        "version": "2.2",
        "timestamp": datetime.now().isoformat(),
        "pinecone": "connected" if pinecone_index else "not available",
        "apps_script": "configured" if CONFIG['APPS_SCRIPT_URL'] else "not configured"
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages with optimized processing"""
    try:
        data = request.get_json()
        
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored"}), 200
        
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        if not chat_id:
            return jsonify({"status": "no_chat_id"}), 200
        
        if '@g.us' in chat_id:
            return jsonify({"status": "group_ignored"}), 200
        
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
        
        logger.info(f"Received from {chat_id}: {text_message[:50]}...")
        
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
        
        thread = threading.Thread(target=process_and_respond, daemon=True)
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
    
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
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
        vector_count = 0
    
    components["google_ai"] = llm is not None and embed_model is not None
    components["green_api"] = bool(CONFIG['GREEN_API_ID'] and CONFIG['GREEN_API_TOKEN'])
    components["apps_script"] = bool(CONFIG['APPS_SCRIPT_URL'])
    
    all_ok = components["database"] and components["google_ai"]
    
    return jsonify({
        "status": "healthy" if all_ok else "degraded",
        "components": components,
        "config": {
            "index_name": CONFIG['INDEX_NAME'],
            "max_history": CONFIG['MAX_HISTORY'],
            "vector_count": vector_count if components["pinecone"] else 0
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


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("AI COACH - SCHOOLINKA v2.2 (OPTIMIZED)")
    logger.info("=" * 70)
    logger.info(f"Database: {CONFIG['DB_PATH']}")
    logger.info(f"Pinecone Index: {CONFIG['INDEX_NAME']} - {'Connected' if pinecone_index else 'Not Available'}")
    logger.info(f"Google AI: {'Initialized' if llm else 'Failed'}")
    logger.info(f"Apps Script: {'Configured' if CONFIG['APPS_SCRIPT_URL'] else 'Not Configured'}")
    
    if pinecone_index:
        try:
            stats = pinecone_index.describe_index_stats()
            logger.info(f"Vector Count: {stats.get('total_vector_count', 0)}")
        except:
            logger.warning("Could not retrieve vector count")
    
    logger.info("=" * 70)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
