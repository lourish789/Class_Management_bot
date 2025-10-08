import os
import time
import re
import asyncio
import nest_asyncio
from datetime import datetime
import sqlite3
from contextlib import contextmanager
from flask import Flask, request, jsonify
import threading
import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import logging
from typing import List, Dict, Optional, Tuple
import sys
import pysqlite3
import json


# Fix SQLite and apply async patch
sys.modules["sqlite3"] = pysqlite3
nest_asyncio.apply()

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
    'MAX_HISTORY': 20
  #  'INDEX_NAME': 'coach'
}

# Validate keys
if not all([CONFIG['PINECONE_API_KEY'], CONFIG['GOOGLE_API_KEY'], 
            CONFIG['GREEN_API_ID'], CONFIG['GREEN_API_TOKEN']]):
    raise ValueError("Missing required API keys")

if not CONFIG['APPS_SCRIPT_URL']:
    logger.warning("Apps Script URL not configured. Google Sheets logging will be disabled.")

# Initialize AI services
pinecone_index = None
embed_model = None
llm = None

# Initialize Pinecone and embedding model with error handling
try:
    pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
    pinecone_index = pc.Index("coach")
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=CONFIG['GOOGLE_API_KEY'])
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=CONFIG['GOOGLE_API_KEY'],
        temperature=0.7,
        max_tokens=1000
    )
    logger.info("AI services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization error: {e}")
    raise


    
  #  logger.info("Pinecone and embeddings initialized successfully")
    
#except Exception as e:
  #  logger.error(f"Error initializing Pinecone/Embeddings: {e}")
  #  pinecone_index = None
  #  embed_model = None
    
    
#try:
    # Initialize Pinecone
    #pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
    
    # List all indexes to check if 'coach' exists
    #try:
        #existing_indexes = pc.list_indexes()
       # index_names = [index['name'] for index in existing_indexes]
        
       # if CONFIG['INDEX_NAME'] in index_names:
         #   pinecone_index = pc.Index(CONFIG['INDEX_NAME'])
          #  logger.info(f"Pinecone index '{CONFIG['INDEX_NAME']}' connected successfully")
      #  else:
           # logger.warning(f"Pinecone index '{CONFIG['INDEX_NAME']}' not found. RAG features will be disabled.")
          #  logger.info(f"Available indexes: {index_names}")
   # except Exception as e:
      #  logger.error(f"Error checking Pinecone indexes: {e}")
      #  logger.warning("Continuing without Pinecone RAG support")
    
    # Initialize Google AI services
   # embed_model = GoogleGenerativeAIEmbeddings(
     #   model="models/embedding-001", 
    #    google_api_key=CONFIG['GOOGLE_API_KEY']
  #  )
    
 #   llm = ChatGoogleGenerativeAI(
     #   model="gemini-2.0-flash-exp",
      #  google_api_key=CONFIG['GOOGLE_API_KEY'],
      #  temperature=0.5,
     #   max_tokens=800
 #   )
  #  logger.info("AI services initialized successfully")
#except Exception as e:
 #   logger.error(f"Service initialization error: {e}")
  #  raise


class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
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
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat ON conversations (chat_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
            conn.commit()
            logger.info("Database initialized")
    
    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_user(self, chat_id: str) -> Optional[Dict]:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE chat_id = ?', (chat_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def save_user(self, chat_id: str, **kwargs) -> bool:
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


class AICoach:
    """Core AI logic with RAG"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
    
    def get_rag_content(self, query: str) -> Tuple[str, List[str]]:
        """Retrieve relevant content from Pinecone"""
        if not self.pinecone_index:
            logger.info("Pinecone index not available, skipping RAG")
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
                if match.get('score', 0) > 0.7:
                    text = match['metadata'].get('text', '')[:400]
                    source = match['metadata'].get('source', 'Knowledge Base')
                    if text:
                        contents.append(text)
                        sources.append(source)
            
            return '\n\n'.join(contents) if contents else "", sources
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return "", []
    
    def generate_response(self, message: str, user_profile: Dict, history: List[Dict] = None) -> Tuple[str, str]:
        """Generate AI response with proper formatting"""
        try:
            intent = self._extract_intent(message)
            rag_content, sources = self.get_rag_content(message)
            
            # Build context from conversation history
            context = ""
            if history:
                recent_context = history[-6:]
                context = "\n".join([
                    f"{'Teacher' if msg['message_type'] == 'user' else 'AI Coach'}: {msg['message_content'][:150]}"
                    for msg in recent_context
                ])
            
            system_prompt = f"""You are AI Coach by Schoolinka, a knowledgeable and supportive assistant helping Nigerian teachers excel in their profession.

Teacher Profile:
- Name: {user_profile.get('first_name', 'Teacher')}
- Teaching: {user_profile.get('class_taught', 'Not specified')}
- Location: {user_profile.get('location', 'Nigeria')}

Core Guidelines:
1. RESPONSE STYLE:
   - Be conversational, warm, and professional
   - Use clear, simple language suitable for text messaging
   - Provide practical, actionable advice tailored to Nigerian classrooms
   - Be specific and detailed when explaining concepts

2. FORMATTING RULES:
   - Use numbers (1. 2. 3.) when listing steps, strategies, or multiple items
   - Start each numbered item on a new line
   - Use short paragraphs (2-3 sentences max per paragraph)
   - Add line breaks between major sections for readability
   - NO asterisks, bullet points, or special characters
   - NO bold or italic markdown formatting

3. RESPONSE LENGTH:
   - Simple questions: 3-5 sentences
   - How-to questions: Provide detailed numbered steps
   - Complex topics: 2-3 paragraphs with clear explanations
   - Always be thorough enough to be truly helpful

4. CONTENT APPROACH:
   - Address the specific question directly first
   - Provide context and explanation when needed
   - Give practical examples relevant to Nigerian schools
   - Consider resource constraints and local context
   - Enumerate steps or options when appropriate

{f'Recent conversation context:\n{context}\n' if context else ''}

{f'Relevant knowledge from resources:\n{rag_content}\n' if rag_content else ''}

Teacher's question: {message}

Provide a helpful, well-structured response:"""
            
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt}
            ])
            
            # Clean response - remove markdown formatting but keep numbers
            clean_response = response.content.strip()
            
            # Remove asterisks and markdown bold/italic
            clean_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_response)
            clean_response = re.sub(r'\*([^*]+)\*', r'\1', clean_response)
            clean_response = clean_response.replace('**', '').replace('*', '')
            
            # Replace bullet points with numbers if they exist
            lines = clean_response.split('\n')
            processed_lines = []
            bullet_counter = 1
            in_list = False
            
            for line in lines:
                stripped = line.strip()
                # Check if line starts with bullet point
                if stripped.startswith(('•', '-', '—')) and len(stripped) > 2:
                    # Convert to numbered list
                    content = re.sub(r'^[•\-—]\s*', '', stripped)
                    processed_lines.append(f"{bullet_counter}. {content}")
                    bullet_counter += 1
                    in_list = True
                elif stripped and in_list and not stripped[0].isdigit():
                    # Continue previous point
                    processed_lines.append(f"   {stripped}")
                else:
                    if stripped and not in_list:
                        processed_lines.append(stripped)
                    elif stripped:
                        processed_lines.append(stripped)
                    if stripped and stripped[0].isdigit():
                        in_list = True
                    elif not stripped:
                        in_list = False
                        bullet_counter = 1
                        if processed_lines and processed_lines[-1]:
                            processed_lines.append('')
            
            clean_response = '\n'.join(processed_lines)
            
            # Clean up excessive line breaks
            clean_response = re.sub(r'\n{3,}', '\n\n', clean_response)
            clean_response = clean_response.strip()
            
            return clean_response, intent
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble right now. Please try again in a moment.", "error"
    
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Extract intent from message"""
        msg = message.lower()
        intents = {
            'teaching_strategy': ['strategy', 'method', 'technique', 'how to teach', 'lesson plan', 'explain', 'introduce', 'activity', 'engage'],
            'classroom_management': ['discipline', 'behavior', 'manage', 'control', 'disruptive', 'noise', 'attention', 'misbehave', 'order'],
            'assessment': ['assess', 'evaluate', 'grade', 'test', 'exam', 'score', 'mark', 'performance', 'feedback', 'progress'],
            'wellbeing': ['stress', 'tired', 'overwhelmed', 'burnout', 'exhausted', 'frustrated', 'difficult', 'challenge', 'hard'],
            'curriculum': ['curriculum', 'syllabus', 'topic', 'subject', 'content', 'what to teach', 'scheme of work'],
            'parent_communication': ['parent', 'guardian', 'communicate', 'meeting', 'report', 'concern'],
            'resources': ['resource', 'material', 'tool', 'equipment', 'book', 'aid']
        }
        
        for intent, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent
        return 'general'


def log_to_sheets_via_apps_script(data_type: str, data: Dict) -> bool:
    """Log data to Google Sheets via Apps Script"""
    if not CONFIG['APPS_SCRIPT_URL']:
        return False
    
    try:
        payload = {
            'type': data_type,
            'data': data
        }
        
        response = requests.post(
            CONFIG['APPS_SCRIPT_URL'],
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully logged {data_type} to Google Sheets")
            return True
        else:
            logger.error(f"Failed to log to Sheets: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error logging to Google Sheets: {e}")
        return False


def parse_registration(text: str) -> Dict:
    """Parse user registration details"""
    details = {}
    lines = text.strip().split('\n')
    
    for line in lines:
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key, value = key.strip().lower(), value.strip()
        
        if 'name' in key and len(value) > 1:
            details['first_name'] = value.title()
        elif 'email' in key and '@' in value:
            details['email'] = value.lower()
        elif 'phone' in key and len(value) >= 7:
            details['phone_number'] = value
        elif 'location' in key and len(value) > 2:
            details['location'] = value.title()
        elif 'class' in key and len(value) > 1:
            details['class_taught'] = value.title()
    
    return details


# Initialize components
db = DatabaseManager(CONFIG['DB_PATH'])
ai_coach = AICoach(llm, embed_model, pinecone_index)


def process_message(chat_id: str, text_message: str) -> str:
    """Main message processing logic - TEXT ONLY"""
    try:
        user = db.get_user(chat_id)
        
        # Handle registration
        if not user or not user.get('profile_complete'):
            if not user:
                db.save_user(chat_id, profile_complete=False)
                return (
                    "Hello! I'm AI Coach by Schoolinka. 👋\n\n"
                    "I'm here to support you with teaching strategies, classroom management, and professional guidance.\n\n"
                    "To get started, please share your details in one message:\n\n"
                    "Name: [Your full name]\n"
                    "Email: [Your email]\n"
                    "Phone: [Your number]\n"
                    "Location: [City/State]\n"
                    "Class: [Class you teach]\n\n"
                    "Example:\n"
                    "Name: Amina Bello\n"
                    "Email: amina@email.com\n"
                    "Phone: 08012345678\n"
                    "Location: Lagos\n"
                    "Class: Primary 4"
                )
            
            # Parse registration
            details = parse_registration(text_message)
            
            if len(details) >= 4:
                db.save_user(chat_id, profile_complete=True, **details)
                user = db.get_user(chat_id)
                
                # Log user to Google Sheets via Apps Script
                log_to_sheets_via_apps_script('user', {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'chat_id': chat_id,
                    'name': details.get('first_name', ''),
                    'email': details.get('email', ''),
                    'phone': details.get('phone_number', ''),
                    'location': details.get('location', ''),
                    'class': details.get('class_taught', ''),
                    'status': 'Registered'
                })
                
                welcome_msg = (
                    f"Welcome, {details.get('first_name', 'Teacher')}! 🎉\n\n"
                    f"I'm excited to support you with your {details.get('class_taught', 'class')}. "
                    f"I can help you with:\n\n"
                    f"1. Teaching strategies and lesson planning\n"
                    f"2. Classroom management techniques\n"
                    f"3. Assessment and feedback methods\n"
                    f"4. Parent communication\n"
                    f"5. Professional development tips\n\n"
                    f"What would you like help with today?"
                )
                
                # Log registration conversation
                log_to_sheets_via_apps_script('conversation', {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'chat_id': chat_id,
                    'user_name': details.get('first_name', 'Teacher'),
                    'user_message': text_message[:500],
                    'bot_response': welcome_msg[:500],
                    'intent': 'registration'
                })
                
                return welcome_msg
            else:
                return (
                    "I need all 5 details to complete your registration. Please provide:\n\n"
                    "Name: [Your full name]\n"
                    "Email: [Your email]\n"
                    "Phone: [Your number]\n"
                    "Location: [Your city]\n"
                    "Class: [Class you teach]\n\n"
                    "You can copy and fill in the format above."
                )
        
        # Validate text message
        if not text_message or len(text_message.strip()) < 3:
            return "Please send me a question or message about teaching. I'm here to help!"
        
        # Update user activity
        db.save_user(chat_id)
        
        # Get conversation history
        history = db.get_history(chat_id, limit=10)
        
        # Extract intent and save user message
        intent = ai_coach._extract_intent(text_message)
        db.save_message(chat_id, 'user', text_message, intent=intent)
        
        # Generate AI response
        ai_response, response_intent = ai_coach.generate_response(text_message, user, history)
        db.save_message(chat_id, 'assistant', ai_response, intent=response_intent)
        
        # Log to Google Sheets via Apps Script
        log_to_sheets_via_apps_script('conversation', {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chat_id': chat_id,
            'user_name': user.get('first_name', 'Unknown'),
            'user_message': text_message[:500],
            'bot_response': ai_response[:500],
            'intent': intent
        })
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Message processing error for {chat_id}: {e}")
        return "I'm experiencing technical difficulties. Please try again in a moment."


def send_whatsapp_message(chat_id: str, message: str) -> bool:
    """Send message via Green API"""
    try:
        url = f"https://api.green-api.com/waInstance{CONFIG['GREEN_API_ID']}/sendMessage/{CONFIG['GREEN_API_TOKEN']}"
        response = requests.post(
            url, 
            json={"chatId": chat_id, "message": message},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Send message error: {e}")
        return False


# Flask Routes
@app.route('/')
def health():
    return jsonify({
        "status": "healthy",
        "service": "AI Coach - Schoolinka",
        "timestamp": datetime.now().isoformat(),
        "pinecone_connected": pinecone_index is not None,
        "apps_script_configured": bool(CONFIG['APPS_SCRIPT_URL'])
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages - TEXT ONLY"""
    try:
        data = request.get_json()
        
        # Immediately acknowledge webhook
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored"}), 200
        
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        if not chat_id:
            return jsonify({"status": "no_chat_id"}), 200
        
        # Extract TEXT message only
        text_message = None
        
        if 'textMessageData' in message_data:
            text_message = message_data['textMessageData'].get('textMessage', '').strip()
        elif 'extendedTextMessageData' in message_data:
            text_message = message_data['extendedTextMessageData'].get('text', '').strip()
        
        logger.info(f"Received text message from {chat_id}: {text_message[:50] if text_message else 'empty'}")
        
        if not text_message:
            # Ignore non-text messages
            send_whatsapp_message(chat_id, "I can only respond to text messages. Please type your question.")
            return jsonify({"status": "non_text_ignored"}), 200
        
        # Process in background thread
        def process_and_respond():
            try:
                reply = process_message(chat_id, text_message)
                send_whatsapp_message(chat_id, reply)
            except Exception as e:
                logger.error(f"Processing error: {e}")
                send_whatsapp_message(chat_id, "Sorry, I encountered an error. Please try again.")
        
        thread = threading.Thread(target=process_and_respond)
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/user/<chat_id>', methods=['GET'])
def get_user_info(chat_id):
    """Get user profile and recent messages"""
    try:
        user = db.get_user(chat_id)
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404
        
        history = db.get_history(chat_id, limit=10)
        return jsonify({
            "status": "success",
            "user": user,
            "recent_messages": history
        })
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint for development"""
    try:
        data = request.get_json()
        chat_id = data.get('chat_id', 'test_user')
        message = data.get('message', 'Hello')
        
        response = process_message(chat_id, message)
        user = db.get_user(chat_id)
        
        return jsonify({
            "status": "success",
            "response": response,
            "user": user,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total_users FROM users')
            total_users = cursor.fetchone()['total_users']
            
            cursor.execute('SELECT COUNT(*) as total_messages FROM conversations')
            total_messages = cursor.fetchone()['total_messages']
            
            cursor.execute('SELECT COUNT(*) as registered_users FROM users WHERE profile_complete = 1')
            registered_users = cursor.fetchone()['registered_users']
        
        return jsonify({
            "status": "success",
            "stats": {
                "total_users": total_users,
                "registered_users": registered_users,
                "total_messages": total_messages,
                "pinecone_connected": pinecone_index is not None,
                "apps_script_configured": bool(CONFIG['APPS_SCRIPT_URL'])
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/create_index', methods=['POST'])
def create_index():
    """Create Pinecone index if it doesn't exist - for initial setup"""
    try:
        data = request.get_json() or {}
        dimension = data.get('dimension', 768)
        
        pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
        existing_indexes = pc.list_indexes()
        index_names = [index['name'] for index in existing_indexes]
        
        if CONFIG['INDEX_NAME'] in index_names:
            return jsonify({
                "status": "success",
                "message": f"Index '{CONFIG['INDEX_NAME']}' already exists"
            })
        
        pc.create_index(
            name=CONFIG['INDEX_NAME'],
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
        logger.info(f"Successfully created Pinecone index: {CONFIG['INDEX_NAME']}")
        return jsonify({
            "status": "success",
            "message": f"Index '{CONFIG['INDEX_NAME']}' created successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health_check', methods=['GET'])
def health_check():
    """Detailed health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "AI Coach - Schoolinka",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": False,
            "pinecone": False,
            "google_ai": False,
            "green_api": False,
            "apps_script": False
        },
        "config": {
            "max_history": CONFIG['MAX_HISTORY'],
            "index_name": CONFIG['INDEX_NAME']
        }
    }
    
    # Check database
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users')
            health_status["components"]["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    # Check Pinecone
    health_status["components"]["pinecone"] = pinecone_index is not None
    
    # Check Google AI
    health_status["components"]["google_ai"] = llm is not None and embed_model is not None
    
    # Check Green API
    health_status["components"]["green_api"] = bool(CONFIG['GREEN_API_ID'] and CONFIG['GREEN_API_TOKEN'])
    
    # Check Apps Script
    health_status["components"]["apps_script"] = bool(CONFIG['APPS_SCRIPT_URL'])
    
    # Overall status
    all_critical_ok = all([
        health_status["components"]["database"],
        health_status["components"]["google_ai"]
    ])
    
    health_status["status"] = "healthy" if all_critical_ok else "degraded"
    status_code = 200 if all_critical_ok else 503
    
    return jsonify(health_status), status_code


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting AI Coach - Schoolinka")
    logger.info(f"Pinecone Index: {'Connected' if pinecone_index else 'Not Available'}")
    logger.info(f"Apps Script: {'Configured' if CONFIG['APPS_SCRIPT_URL'] else 'Not Configured'}")
    logger.info(f"Database: {CONFIG['DB_PATH']}")
    logger.info("=" * 60)
    
    # Verify critical components
    if not llm or not embed_model:
        logger.error("Critical AI components not initialized. Check Google API key.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
