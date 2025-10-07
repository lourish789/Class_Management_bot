import os
import asyncio
import nest_asyncio
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
import pysqlite3
import gspread
from google.oauth2.service_account import Credentials
import base64
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

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
    'SPREADSHEET_ID': "1H8HKj8sIdD8mPEsR8x32qIprEa_jr6UDfdUU50IjkHk",
    'DB_PATH': "ai_coach.db",
    'MAX_HISTORY': 20
}

# Validate keys
if not all([CONFIG['PINECONE_API_KEY'], CONFIG['GOOGLE_API_KEY'], 
            CONFIG['GREEN_API_ID'], CONFIG['GREEN_API_TOKEN']]):
    raise ValueError("Missing required API keys")

# Initialize Google Sheets
try:
    gc = gspread.service_account(filename="credentials.json")
    sh = gc.open_by_key(CONFIG['SPREADSHEET_ID'])
    worksheet = sh.sheet1
    logger.info("Google Sheets connected successfully")
except Exception as e:
    logger.error(f"Google Sheets initialization error: {e}")
    worksheet = None

# Initialize AI services
try:
    pc = Pinecone(api_key=CONFIG['PINECONE_API_KEY'])
    pinecone_index = pc.Index("coach")
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=CONFIG['GOOGLE_API_KEY']
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=CONFIG['GOOGLE_API_KEY'],
        temperature=0.65,
        max_tokens=1000
    )
    logger.info("AI services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization error: {e}")
    raise


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
                    media_type TEXT,
                    media_url TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    intent TEXT,
                    FOREIGN KEY (chat_id) REFERENCES users (chat_id)
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat ON conversations (chat_id)')
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
                    updates = ', '.join([f'{k} = ?' for k in kwargs.keys()])
                    updates += ', last_interaction = ?, total_messages = total_messages + 1'
                    values = list(kwargs.values()) + [datetime.now(), chat_id]
                    cursor.execute(f'UPDATE users SET {updates} WHERE chat_id = ?', values)
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
    
    def save_message(self, chat_id: str, msg_type: str, content: str, 
                    media_type: str = None, media_url: str = None, intent: str = None):
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (chat_id, message_type, message_content, media_type, media_url, intent)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (chat_id, msg_type, content, media_type, media_url, intent))
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
                    SELECT message_type, message_content, media_type, timestamp, intent
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


class MediaProcessor:
    """Handles voice notes and images"""
    
    @staticmethod
    def download_media(url: str) -> Optional[bytes]:
        """Download media from Green API"""
        try:
            response = requests.get(url, timeout=30)
            return response.content if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"Media download error: {e}")
            return None
    
    @staticmethod
    def transcribe_audio(audio_data: bytes, file_ext: str = 'ogg') -> Optional[str]:
        """Transcribe voice note to text"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_path = temp_audio.name
            
            # Convert to WAV
            audio = AudioSegment.from_file(temp_path)
            wav_path = temp_path.replace(f'.{file_ext}', '.wav')
            audio.export(wav_path, format='wav')
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Transcribe
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_content = recognizer.record(source)
                text = recognizer.recognize_google(audio_content)
            
            os.remove(wav_path)
            logger.info(f"Audio transcribed successfully: {text[:50]}...")
            return text
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return None
    
    @staticmethod
    def analyze_image(image_data: bytes, prompt: str = "") -> str:
        """Analyze image - placeholder for actual implementation"""
        try:
            # Image analysis would require Gemini Vision API
            # For now, return a helpful response
            context = f" related to: {prompt}" if prompt else ""
            return f"I can see you've shared an image{context}. Please describe what you'd like help with regarding this image."
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return "I received your image. Could you describe what you'd like help with?"


class AICoach:
    """Core AI logic with RAG"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
    
    def get_rag_content(self, query: str) -> Tuple[str, List[str]]:
        """Retrieve relevant content from Pinecone"""
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
                    text = match['metadata'].get('text', '')[:200]
                    source = match['metadata'].get('source', 'Unknown')
                    if text:
                        contents.append(text)
                        sources.append(source)
            
            return '\n\n'.join(contents) if contents else "No relevant information found.", sources
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return "Knowledge base temporarily unavailable.", []
    
    def generate_response(self, message: str, user_profile: Dict) -> Tuple[str, str]:
        """Generate AI response"""
        try:
            intent = self._extract_intent(message)
            rag_content, sources = self.get_rag_content(message)
            
            system_prompt = f"""You are AI Coach by Schoolinka. Help Nigerian teachers with:
- Teaching strategies for {user_profile.get('class_taught', 'their class')}
- Classroom management for large classes
- Student assessment techniques
- Teacher wellbeing and stress management

Context from knowledge base: {rag_content}

Provide practical, actionable advice in under 150 words. Be warm, supportive, and professional."""
            
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ])
            
            return response.content.strip(), intent
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble right now. Please try again in a moment.", "error"
    
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Extract intent from message"""
        msg = message.lower()
        intents = {
            'teaching_strategy': ['strategy', 'method', 'technique', 'how to teach', 'lesson plan'],
            'classroom_management': ['discipline', 'behavior', 'manage', 'control', 'disruptive'],
            'assessment': ['assess', 'evaluate', 'grade', 'test', 'exam', 'score'],
            'wellbeing': ['stress', 'tired', 'overwhelmed', 'burnout', 'exhausted']
        }
        
        for intent, keywords in intents.items():
            if any(kw in msg for kw in keywords):
                return intent
        return 'general'


def log_to_sheets(chat_id: str, user_message: str, bot_response: str, timestamp: str = None):
    """Log conversation to Google Sheets"""
    if not worksheet:
        logger.warning("Google Sheets not available")
        return
    
    try:
        timestamp = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_to_add = [timestamp, chat_id, user_message, bot_response]
        worksheet.append_row(data_to_add)
        logger.info(f"Logged to Sheets: {chat_id}")
    except Exception as e:
        logger.error(f"Error logging to Sheets: {e}")


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
media_processor = MediaProcessor()
ai_coach = AICoach(llm, embed_model, pinecone_index)


def process_message(chat_id: str, message_data: Dict) -> str:
    """Main message processing logic"""
    try:
        user = db.get_user(chat_id)
        
        # Handle registration
        if not user or not user.get('profile_complete'):
            if not user:
                db.save_user(chat_id, profile_complete=False)
                return (
                    "Hello! I'm AI Coach by Schoolinka 🎓\n\n"
                    "Please share your details in one message:\n\n"
                    "Name: [Your name]\n"
                    "Email: [Your email]\n"
                    "Phone: [Your number]\n"
                    "Location: [City/State]\n"
                    "Class: [Class you teach]"
                )
            
            # Parse registration
            text = message_data.get('text', '')
            details = parse_registration(text)
            
            if len(details) >= 4:
                db.save_user(chat_id, profile_complete=True, **details)
                user = db.get_user(chat_id)
                
                welcome_msg = (
                    f"Welcome, {details.get('first_name', 'Teacher')}! ✅\n\n"
                    f"I'm here to help with {details.get('class_taught', 'your class')}:\n"
                    f"• Teaching strategies\n"
                    f"• Classroom management\n"
                    f"• Assessment\n"
                    f"• Wellbeing support\n\n"
                    f"How can I help you today?"
                )
                log_to_sheets(chat_id, text, welcome_msg)
                return welcome_msg
            else:
                return "Please provide all details in the format shown above."
        
        # Process different media types
        text_content = ""
        media_type = message_data.get('type')
        media_url = message_data.get('url')
        
        if media_type == 'text':
            text_content = message_data.get('text', '')
        
        elif media_type == 'voice' and media_url:
            logger.info(f"Processing voice message from {chat_id}")
            audio_data = media_processor.download_media(media_url)
            if audio_data:
                transcribed = media_processor.transcribe_audio(audio_data, 'ogg')
                if transcribed:
                    text_content = transcribed
                    db.save_message(chat_id, 'user', text_content, 'voice', media_url)
                else:
                    return "I couldn't transcribe your voice note. Please try typing your message or sending the audio again."
            else:
                return "I couldn't download your voice note. Please try again."
        
        elif media_type == 'image' and media_url:
            logger.info(f"Processing image from {chat_id}")
            image_data = media_processor.download_media(media_url)
            caption = message_data.get('caption', '')
            
            if image_data:
                analysis = media_processor.analyze_image(image_data, caption)
                text_content = f"{analysis}\n\nYour caption: {caption}" if caption else analysis
                db.save_message(chat_id, 'user', text_content, 'image', media_url)
            else:
                return "I couldn't process your image. Please try again."
        
        if not text_content:
            return "I can help with text messages, voice notes, and images. What would you like to know?"
        
        # Update user activity
        db.save_user(chat_id)
        
        # Extract intent and save user message
        intent = ai_coach._extract_intent(text_content)
        if media_type == 'text':
            db.save_message(chat_id, 'user', text_content, intent=intent)
        
        # Generate AI response
        ai_response, response_intent = ai_coach.generate_response(text_content, user)
        db.save_message(chat_id, 'assistant', ai_response, intent=response_intent)
        
        # Log to Google Sheets
        log_to_sheets(chat_id, text_content[:500], ai_response[:500])
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Message processing error for {chat_id}: {e}")
        return "I'm experiencing technical difficulties. Please try again!"


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
        "timestamp": datetime.now().isoformat()
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages"""
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
        
        # Extract message content
        msg_content = {}
        
        if 'textMessageData' in message_data:
            msg_content = {
                'type': 'text',
                'text': message_data['textMessageData'].get('textMessage', '')
            }
        elif 'extendedTextMessageData' in message_data:
            # Voice note
            msg_content = {
                'type': 'voice',
                'url': message_data.get('downloadUrl'),
                'mimeType': message_data.get('mimeType', '')
            }
        elif 'imageMessage' in message_data:
            msg_content = {
                'type': 'image',
                'url': message_data.get('downloadUrl'),
                'caption': message_data.get('caption', '')
            }
        
        logger.info(f"Received {msg_content.get('type', 'unknown')} from {chat_id}")
        
        if msg_content:
            def process_and_respond():
                try:
                    reply = process_message(chat_id, msg_content)
                    send_whatsapp_message(chat_id, reply)
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    send_whatsapp_message(chat_id, "Sorry, I encountered an error. Please try again!")
            
            thread = threading.Thread(target=process_and_respond)
            thread.daemon = True
            thread.start()
        
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/user/<chat_id>', methods=['GET'])
def get_user_info(chat_id):
    """Get user profile"""
    user = db.get_user(chat_id)
    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404
    
    history = db.get_history(chat_id, limit=5)
    return jsonify({
        "status": "success",
        "user": user,
        "recent_messages": history
    })


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint"""
    data = request.get_json()
    chat_id = data.get('chat_id', 'test_user')
    message = data.get('message', 'Hello')
    msg_type = data.get('type', 'text')
    
    response = process_message(chat_id, {'type': msg_type, 'text': message})
    return jsonify({
        "status": "success",
        "response": response,
        "user": db.get_user(chat_id)
    })


if __name__ == "__main__":
    logger.info("Starting AI Coach - Schoolinka")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
