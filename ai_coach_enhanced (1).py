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
import base64
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import re

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
    'MAX_HISTORY': 25
}

# Validate keys
if not all([CONFIG['PINECONE_API_KEY'], CONFIG['GOOGLE_API_KEY'], 
            CONFIG['GREEN_API_ID'], CONFIG['GREEN_API_TOKEN']]):
    raise ValueError("Missing required API keys")

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
        temperature=0.7,
        max_tokens=2000
    )
    logger.info("AI services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization error: {e}")
    raise


class DatabaseManager:
    """Handles all database operations with enhanced tracking"""
    
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
                    school_name TEXT,
                    years_teaching INTEGER,
                    profile_complete BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_messages INTEGER DEFAULT 0,
                    registration_step INTEGER DEFAULT 0
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
                    sentiment TEXT,
                    FOREIGN KEY (chat_id) REFERENCES users (chat_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    chat_id TEXT PRIMARY KEY,
                    preferred_topics TEXT,
                    difficulty_level TEXT,
                    notification_time TEXT,
                    FOREIGN KEY (chat_id) REFERENCES users (chat_id)
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat ON conversations (chat_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_intent ON conversations (intent)')
            conn.commit()
            logger.info("Database initialized with enhanced schema")
    
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
                    media_type: str = None, media_url: str = None, 
                    intent: str = None, sentiment: str = None):
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (chat_id, message_type, message_content, media_type, media_url, intent, sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (chat_id, msg_type, content, media_type, media_url, intent, sentiment))
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
    
    def get_conversation_context(self, chat_id: str, limit: int = 5) -> str:
        """Get recent conversation context for better responses"""
        history = self.get_history(chat_id, limit)
        if not history:
            return ""
        
        context_parts = []
        for msg in history[-5:]:
            role = "Teacher" if msg['message_type'] == 'user' else "AI Coach"
            content = msg['message_content'][:200]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
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
    """Enhanced media processing with better error handling"""
    
    @staticmethod
    def download_media(url: str) -> Optional[bytes]:
        """Download media from Green API with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    return response.content
                logger.warning(f"Download attempt {attempt + 1} failed with status {response.status_code}")
            except Exception as e:
                logger.error(f"Media download error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(2)
        return None
    
    @staticmethod
    def transcribe_audio(audio_data: bytes, file_ext: str = 'ogg') -> Optional[str]:
        """Transcribe voice note to text with enhanced error handling"""
        temp_audio_path = None
        wav_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            # Convert to WAV
            audio = AudioSegment.from_file(temp_audio_path)
            wav_path = temp_audio_path.replace(f'.{file_ext}', '.wav')
            audio.export(wav_path, format='wav')
            
            # Transcribe
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_content = recognizer.record(source)
                text = recognizer.recognize_google(audio_content)
            
            logger.info(f"Audio transcribed successfully: {text[:50]}...")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return None
        finally:
            # Cleanup temp files
            for path in [temp_audio_path, wav_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
    
    @staticmethod
    def analyze_image(image_data: bytes, prompt: str = "") -> str:
        """Analyze image - placeholder for actual Gemini Vision implementation"""
        try:
            context = f" related to: {prompt}" if prompt else ""
            return (
                f"I can see you've shared an image{context}.\n\n"
                f"Please describe what you'd like help with regarding this image, "
                f"and I'll provide specific guidance for your teaching situation."
            )
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return "I received your image. Could you describe what you'd like help with?"


class ResponseFormatter:
    """Formats responses according to WhatsApp guidelines"""
    
    @staticmethod
    def format_response(text: str) -> str:
        """Format response for WhatsApp with proper structure"""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italics
        text = re.sub(r'#{1,6}\s', '', text)  # Remove headers
        text = re.sub(r'`(.*?)`', r'\1', text)  # Remove code formatting
        
        # Replace bullet points with numbers if they exist
        lines = text.split('\n')
        formatted_lines = []
        in_list = False
        list_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                in_list = False
                list_counter = 1
                continue
            
            # Convert bullets to numbers
            if line.startswith(('- ', '• ', '* ')):
                line = f"{list_counter}. {line[2:].strip()}"
                list_counter += 1
                in_list = True
            elif in_list and not line[0].isdigit():
                in_list = False
                list_counter = 1
            
            formatted_lines.append(line)
        
        # Join with proper spacing
        result = '\n'.join(formatted_lines)
        
        # Ensure proper paragraph spacing
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    @staticmethod
    def add_line_breaks(text: str, max_length: int = 65) -> str:
        """Add line breaks for better readability on mobile"""
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            if len(para) <= max_length:
                formatted_paragraphs.append(para)
            else:
                # Split long paragraphs at sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_length * 2:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            formatted_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    formatted_paragraphs.append(current_chunk.strip())
        
        return '\n\n'.join(formatted_paragraphs)


class AICoach:
    """Enhanced AI logic with RAG and context awareness"""
    
    def __init__(self, llm, embed_model, pinecone_index):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        self.formatter = ResponseFormatter()
    
    def get_rag_content(self, query: str, top_k: int = 4) -> Tuple[str, List[str]]:
        """Retrieve relevant content from Pinecone with improved relevance"""
        try:
            query_embed = self.embed_model.embed_query(query)
            results = self.pinecone_index.query(
                vector=query_embed,
                top_k=top_k,
                include_metadata=True
            )
            
            contents, sources = [], []
            for match in results.get('matches', []):
                score = match.get('score', 0)
                if score > 0.65:  # Lower threshold for more context
                    text = match['metadata'].get('text', '')
                    source = match['metadata'].get('source', 'Knowledge Base')
                    if text:
                        contents.append(f"[Relevance: {score:.2f}] {text[:300]}")
                        sources.append(source)
            
            if contents:
                logger.info(f"Retrieved {len(contents)} relevant documents")
                return '\n\n'.join(contents), sources
            
            return "No specific knowledge base content found. Using general expertise.", []
            
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return "Knowledge base temporarily unavailable. Using general expertise.", []
    
    def generate_response(self, message: str, user_profile: Dict, 
                         conversation_context: str = "") -> Tuple[str, str]:
        """Generate AI response with enhanced context and formatting"""
        try:
            intent = self._extract_intent(message)
            sentiment = self._analyze_sentiment(message)
            rag_content, sources = self.get_rag_content(message)
            
            # Build comprehensive system prompt
            system_prompt = self._build_system_prompt(
                user_profile, rag_content, conversation_context, intent
            )
            
            # Generate response
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ])
            
            raw_response = response.content.strip()
            
            # Format response according to guidelines
            formatted_response = self.formatter.format_response(raw_response)
            
            # Add sources if relevant
            if sources and len(sources) > 0:
                source_note = f"\n\nSource: {sources[0]}"
                if len(formatted_response) < 800:  # Only add if space allows
                    formatted_response += source_note
            
            return formatted_response, intent
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._get_fallback_response(intent), "error"
    
    def _build_system_prompt(self, user_profile: Dict, rag_content: str, 
                            context: str, intent: str) -> str:
        """Build comprehensive system prompt with all context"""
        
        class_info = user_profile.get('class_taught', 'their class')
        location = user_profile.get('location', 'Nigeria')
        name = user_profile.get('first_name', 'Teacher')
        
        base_prompt = f"""You are AI Coach by Schoolinka, a specialized assistant helping Nigerian teachers excel in their profession.

TEACHER PROFILE:
- Name: {name}
- Teaching: {class_info}
- Location: {location}
- Intent: {intent}

RESPONSE GUIDELINES:
1. Be conversational, warm, and professional
2. Use simple, clear language suitable for WhatsApp
3. Provide practical, actionable advice for Nigerian classrooms
4. Be specific and detailed when explaining concepts

FORMATTING RULES (CRITICAL):
- Use numbers (1. 2. 3.) for steps, strategies, or lists
- Start each numbered item on a new line
- Keep paragraphs short (2-3 sentences maximum)
- Add line breaks between sections
- NO asterisks, bullets, or special characters
- NO markdown bold or italic formatting
- Write naturally as if texting a colleague

RESPONSE LENGTH:
- Simple questions: 3-5 sentences
- How-to questions: Detailed numbered steps with explanations
- Complex topics: 2-3 clear paragraphs with examples
- Always be thorough enough to truly help

CONTENT APPROACH:
- Address the question directly first
- Provide context and explanation
- Give practical examples for Nigerian schools
- Consider resource constraints
- Be culturally sensitive and contextually appropriate
- Enumerate steps when teaching methods or processes"""

        if rag_content and "temporarily unavailable" not in rag_content:
            base_prompt += f"\n\nKNOWLEDGE BASE CONTEXT:\n{rag_content}"
        
        if context:
            base_prompt += f"\n\nRECENT CONVERSATION:\n{context}"
        
        base_prompt += "\n\nProvide a helpful, well-structured response following ALL formatting guidelines above."
        
        return base_prompt
    
    @staticmethod
    def _extract_intent(message: str) -> str:
        """Enhanced intent extraction"""
        msg = message.lower()
        
        intents = {
            'teaching_strategy': [
                'how to teach', 'teaching method', 'strategy', 'technique',
                'lesson plan', 'explain', 'introduce', 'engage students'
            ],
            'classroom_management': [
                'discipline', 'behavior', 'manage class', 'control', 
                'disruptive', 'attention', 'noise', 'order'
            ],
            'assessment': [
                'assess', 'evaluate', 'grade', 'test', 'exam', 
                'score', 'mark', 'feedback', 'progress'
            ],
            'wellbeing': [
                'stress', 'tired', 'overwhelmed', 'burnout', 
                'exhausted', 'mental health', 'work-life'
            ],
            'curriculum': [
                'curriculum', 'syllabus', 'scheme of work', 
                'topic', 'subject', 'content'
            ],
            'resources': [
                'material', 'resource', 'tool', 'equipment', 
                'textbook', 'worksheet', 'activity'
            ],
            'parent_communication': [
                'parent', 'guardian', 'meeting', 'report', 
                'communicate', 'conference'
            ]
        }
        
        intent_scores = {}
        for intent_name, keywords in intents.items():
            score = sum(1 for kw in keywords if kw in msg)
            if score > 0:
                intent_scores[intent_name] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        return 'general'
    
    @staticmethod
    def _analyze_sentiment(message: str) -> str:
        """Simple sentiment analysis"""
        msg = message.lower()
        
        positive_words = ['happy', 'excited', 'great', 'wonderful', 'thank', 'appreciate']
        negative_words = ['frustrated', 'difficult', 'struggling', 'hard', 'problem', 'help']
        
        pos_count = sum(1 for word in positive_words if word in msg)
        neg_count = sum(1 for word in negative_words if word in msg)
        
        if neg_count > pos_count:
            return 'negative'
        elif pos_count > neg_count:
            return 'positive'
        return 'neutral'
    
    @staticmethod
    def _get_fallback_response(intent: str) -> str:
        """Provide fallback responses based on intent"""
        fallbacks = {
            'teaching_strategy': (
                "I'm having a brief technical issue, but I'm here to help with your teaching strategies.\n\n"
                "Could you please rephrase your question? For example:\n\n"
                "1. What specific topic are you teaching?\n"
                "2. What challenge are you facing?\n"
                "3. What have you tried so far?\n\n"
                "This will help me give you the best advice."
            ),
            'classroom_management': (
                "I'm experiencing a technical hiccup, but I want to help with your classroom management concern.\n\n"
                "Please share more details:\n\n"
                "1. What's the specific behavior issue?\n"
                "2. How many students are involved?\n"
                "3. What have you tried already?\n\n"
                "I'll provide practical strategies for your situation."
            ),
            'default': (
                "I apologize for the technical difficulty. I'm here to support you as a teacher.\n\n"
                "Please try asking your question again, and I'll do my best to provide helpful guidance.\n\n"
                "You can ask me about teaching strategies, classroom management, assessment, or any other teaching-related topic."
            )
        }
        
        return fallbacks.get(intent, fallbacks['default'])


def log_to_sheets(chat_id: str, user_message: str, bot_response: str, 
                 user_profile: Dict = None, intent: str = None):
    """Log conversation to Google Sheets via Apps Script"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        payload = {
            'action': 'log_conversation',
            'timestamp': timestamp,
            'chat_id': chat_id,
            'user_message': user_message[:1000],  # Limit length
            'bot_response': bot_response[:1000],
            'intent': intent or 'unknown',
            'user_name': user_profile.get('first_name', '') if user_profile else '',
            'class_taught': user_profile.get('class_taught', '') if user_profile else ''
        }
        
        response = requests.post(
            CONFIG['APPS_SCRIPT_URL'],
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Logged to Sheets: {chat_id}")
        else:
            logger.warning(f"Sheets logging failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error logging to Sheets: {e}")


def update_user_in_sheets(chat_id: str, user_data: Dict):
    """Update user profile in Google Sheets"""
    try:
        payload = {
            'action': 'update_user',
            'chat_id': chat_id,
            **user_data
        }
        
        response = requests.post(
            CONFIG['APPS_SCRIPT_URL'],
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Updated user in Sheets: {chat_id}")
        else:
            logger.warning(f"Sheets user update failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error updating user in Sheets: {e}")


def parse_registration(text: str) -> Dict:
    """Enhanced registration parser with better validation"""
    details = {}
    lines = text.strip().split('\n')
    
    for line in lines:
        if ':' not in line:
            continue
            
        key, value = line.split(':', 1)
        key, value = key.strip().lower(), value.strip()
        
        # Name validation
        if 'name' in key and len(value) > 1:
            # Remove numbers and special chars from name
            clean_name = re.sub(r'[^a-zA-Z\s]', '', value)
            if len(clean_name) > 1:
                details['first_name'] = clean_name.title()
        
        # Email validation
        elif 'email' in key and '@' in value and '.' in value:
            email = value.lower().strip()
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                details['email'] = email
        
        # Phone validation
        elif 'phone' in key:
            # Extract digits
            phone = re.sub(r'[^\d+]', '', value)
            if len(phone) >= 10:
                details['phone_number'] = phone
        
        # Location validation
        elif 'location' in key and len(value) > 2:
            details['location'] = value.title()
        
        # Class validation
        elif 'class' in key and len(value) > 1:
            details['class_taught'] = value.title()
        
        # School name (optional)
        elif 'school' in key and len(value) > 2:
            details['school_name'] = value.title()
    
    return details


# Initialize components
db = DatabaseManager(CONFIG['DB_PATH'])
media_processor = MediaProcessor()
ai_coach = AICoach(llm, embed_model, pinecone_index)


def process_message(chat_id: str, message_data: Dict) -> str:
    """Enhanced message processing with better flow"""
    try:
        user = db.get_user(chat_id)
        
        # Handle new users and registration
        if not user or not user.get('profile_complete'):
            if not user:
                db.save_user(chat_id, profile_complete=False, registration_step=1)
                welcome_message = (
                    "Hello! I'm AI Coach by Schoolinka.\n\n"
                    "I'm here to support you with teaching strategies, classroom management, "
                    "assessment techniques, and more.\n\n"
                    "Let's get started. Please share your details in one message:\n\n"
                    "Name: Your full name\n"
                    "Email: Your email address\n"
                    "Phone: Your phone number\n"
                    "Location: Your city or state\n"
                    "Class: The class you teach (e.g., JSS 2, Primary 4)\n\n"
                    "You can also add:\n"
                    "School: Your school name (optional)"
                )
                return welcome_message
            
            # Parse registration attempt
            text = message_data.get('text', '')
            details = parse_registration(text)
            
            # Check if we have minimum required fields
            required_fields = ['first_name', 'email', 'phone_number', 'location', 'class_taught']
            missing_fields = [f for f in required_fields if f not in details]
            
            if len(missing_fields) == 0:
                # Complete registration
                db.save_user(chat_id, profile_complete=True, registration_step=5, **details)
                user = db.get_user(chat_id)
                
                # Update in Google Sheets
                update_user_in_sheets(chat_id, details)
                
                welcome_msg = (
                    f"Welcome aboard, {details.get('first_name')}!\n\n"
                    f"I'm excited to support you with {details.get('class_taught')}.\n\n"
                    f"I can help you with:\n\n"
                    f"1. Teaching strategies and lesson planning\n"
                    f"2. Classroom management techniques\n"
                    f"3. Assessment and evaluation methods\n"
                    f"4. Resources and teaching materials\n"
                    f"5. Teacher wellbeing and stress management\n"
                    f"6. Parent communication strategies\n\n"
                    f"What would you like help with today?"
                )
                
                log_to_sheets(chat_id, text, welcome_msg, user, 'registration')
                return welcome_msg
            
            else:
                # Provide helpful feedback about missing fields
                field_names = {
                    'first_name': 'Name',
                    'email': 'Email',
                    'phone_number': 'Phone',
                    'location': 'Location',
                    'class_taught': 'Class'
                }
                
                missing_list = [field_names.get(f, f) for f in missing_fields]
                
                return (
                    f"Thanks for providing some details. I still need:\n\n"
                    f"{', '.join(missing_list)}\n\n"
                    f"Please provide all information in this format:\n\n"
                    f"Name: Your full name\n"
                    f"Email: Your email\n"
                    f"Phone: Your phone number\n"
                    f"Location: Your city/state\n"
                    f"Class: Class you teach"
                )
        
        # Process different media types
        text_content = ""
        media_type = message_data.get('type')
        media_url = message_data.get('url')
        
        if media_type == 'text':
            text_content = message_data.get('text', '').strip()
        
        elif media_type == 'voice' and media_url:
            logger.info(f"Processing voice message from {chat_id}")
            
            audio_data = media_processor.download_media(media_url)
            if audio_data:
                transcribed = media_processor.transcribe_audio(audio_data, 'ogg')
                if transcribed:
                    text_content = transcribed
                    db.save_message(chat_id, 'user', text_content, 'voice', media_url)
                    
                    # Acknowledge voice note
                    ack_message = f"I heard: \"{text_content[:100]}...\"\n\nLet me help you with that."
                    if len(text_content) < 100:
                        ack_message = f"I heard: \"{text_content}\"\n\nLet me help you with that."
                else:
                    return (
                        "I couldn't clearly transcribe your voice note. This could be due to:\n\n"
                        "1. Background noise\n"
                        "2. Audio quality\n"
                        "3. Speaking speed\n\n"
                        "Please try recording again in a quiet space, or type your message instead."
                    )
            else:
                return "I couldn't download your voice note. Please check your connection and try again."
        
        elif media_type == 'image' and media_url:
            logger.info(f"Processing image from {chat_id}")
            
            image_data = media_processor.download_media(media_url)
            caption = message_data.get('caption', '').strip()
            
            if image_data:
                analysis = media_processor.analyze_image(image_data, caption)
                text_content = f"{caption}\n\n{analysis}" if caption else analysis
                db.save_message(chat_id, 'user', text_content, 'image', media_url)
            else:
                return "I couldn't process your image. Please try sending it again or describe what you need help with."
        
        # Validate we have content to process
        if not text_content:
            return (
                "I can help you through:\n\n"
                "1. Text messages - Type your questions\n"
                "2. Voice notes - Record your questions\n"
                "3. Images - Share teaching materials or classroom situations\n\n"
                "What would you like to know?"
            )
        
        # Check for special commands
        command_response = handle_special_commands(text_content, user)
        if command_response:
            return command_response
        
        # Update user activity
        db.save_user(chat_id)
        
        # Get conversation context for better responses
        conversation_context = db.get_conversation_context(chat_id, limit=5)
        
        # Extract intent and sentiment
        intent = ai_coach._extract_intent(text_content)
        sentiment = ai_coach._analyze_sentiment(text_content)
        
        # Save user message if text type
        if media_type == 'text':
            db.save_message(chat_id, 'user', text_content, intent=intent, sentiment=sentiment)
        
        # Generate AI response with full context
        ai_response, response_intent = ai_coach.generate_response(
            text_content, 
            user, 
            conversation_context
        )
        
        # Save assistant response
        db.save_message(chat_id, 'assistant', ai_response, intent=response_intent)
        
        # Log to Google Sheets
        log_to_sheets(chat_id, text_content[:500], ai_response[:500], user, intent)
        
        return ai_response
        
    except Exception as e:
        logger.error(f"Message processing error for {chat_id}: {e}", exc_info=True)
        return (
            "I'm experiencing a technical issue right now.\n\n"
            "Please try again in a moment. If the problem persists, "
            "contact support@schoolinka.com"
        )


def handle_special_commands(text: str, user: Dict) -> Optional[str]:
    """Handle special commands like help, profile, etc."""
    text_lower = text.lower().strip()
    
    # Help command
    if text_lower in ['help', 'menu', 'commands', 'what can you do']:
        return (
            "Here's how I can help you:\n\n"
            "1. Teaching Strategies - Ask about effective teaching methods for any topic\n\n"
            "2. Classroom Management - Get tips for discipline, engagement, and large classes\n\n"
            "3. Assessment - Learn about evaluation techniques and feedback methods\n\n"
            "4. Lesson Planning - Create engaging and effective lesson plans\n\n"
            "5. Resources - Find or create teaching materials\n\n"
            "6. Wellbeing - Get support for stress and work-life balance\n\n"
            "7. Parent Communication - Strategies for effective parent engagement\n\n"
            "Just ask me anything related to teaching, and I'll provide practical advice!"
        )
    
    # Profile command
    elif text_lower in ['profile', 'my profile', 'my info', 'my details']:
        name = user.get('first_name', 'Teacher')
        email = user.get('email', 'Not provided')
        location = user.get('location', 'Not provided')
        class_taught = user.get('class_taught', 'Not provided')
        school = user.get('school_name', 'Not provided')
        total_msgs = user.get('total_messages', 0)
        
        return (
            f"Your Profile:\n\n"
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Location: {location}\n"
            f"Teaching: {class_taught}\n"
            f"School: {school}\n"
            f"Total Messages: {total_msgs}\n\n"
            f"To update your profile, type 'update profile'"
        )
    
    # Update profile command
    elif 'update profile' in text_lower:
        return (
            "To update your profile, please provide the new information:\n\n"
            "Name: Your new name\n"
            "Email: Your new email\n"
            "Phone: Your new phone\n"
            "Location: Your new location\n"
            "Class: Your new class\n"
            "School: Your school name\n\n"
            "Only include the fields you want to update."
        )
    
    # Quick tips command
    elif text_lower in ['tips', 'quick tips', 'daily tip']:
        tips = [
            "Start each lesson with a clear learning objective so students know what to expect.",
            "Use the first 5 minutes to review previous lessons and connect to today's topic.",
            "Mix teaching methods - combine lecture, discussion, and hands-on activities.",
            "Give students wait time after asking questions - at least 3-5 seconds.",
            "Use positive reinforcement more than correction - catch students doing well.",
            "Create clear classroom routines for common activities to save time.",
            "Take 5-minute breaks during long lessons to keep students engaged.",
            "Use local examples and stories to make abstract concepts relatable.",
        ]
        import random
        return f"Quick Teaching Tip:\n\n{random.choice(tips)}\n\nWould you like more tips on a specific topic?"
    
    return None


def send_whatsapp_message(chat_id: str, message: str) -> bool:
    """Send message via Green API with retry logic"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            url = f"https://api.green-api.com/waInstance{CONFIG['GREEN_API_ID']}/sendMessage/{CONFIG['GREEN_API_TOKEN']}"
            
            payload = {
                "chatId": chat_id,
                "message": message
            }
            
            response = requests.post(
                url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {chat_id}")
                return True
            else:
                logger.warning(f"Send attempt {attempt + 1} failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Send message error (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to send message to {chat_id} after {max_retries} attempts")
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
        "features": [
            "Text messaging",
            "Voice note transcription",
            "Image processing",
            "RAG-enhanced responses",
            "Context-aware conversations"
        ]
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages"""
    try:
        data = request.get_json()
        
        # Validate webhook type
        if not data or data.get('typeWebhook') != 'incomingMessageReceived':
            return jsonify({"status": "ignored", "reason": "not_incoming_message"}), 200
        
        # Extract message details
        message_data = data.get('messageData', {})
        sender_data = data.get('senderData', {})
        chat_id = sender_data.get('chatId', '').strip()
        
        if not chat_id:
            logger.warning("Received webhook without chat_id")
            return jsonify({"status": "error", "reason": "no_chat_id"}), 200
        
        # Don't process group messages
        if '@g.us' in chat_id:
            logger.info(f"Ignoring group message from {chat_id}")
            return jsonify({"status": "ignored", "reason": "group_message"}), 200
        
        # Extract message content based on type
        msg_content = {}
        
        if 'textMessageData' in message_data:
            # Regular text message
            msg_content = {
                'type': 'text',
                'text': message_data['textMessageData'].get('textMessage', '')
            }
            
        elif 'extendedTextMessageData' in message_data:
            # Could be voice or other extended content
            if message_data.get('typeMessage') == 'audioMessage':
                msg_content = {
                    'type': 'voice',
                    'url': message_data.get('downloadUrl'),
                    'mimeType': message_data.get('mimeType', '')
                }
            else:
                msg_content = {
                    'type': 'text',
                    'text': message_data['extendedTextMessageData'].get('text', '')
                }
                
        elif 'imageMessage' in message_data:
            # Image message
            msg_content = {
                'type': 'image',
                'url': message_data.get('downloadUrl'),
                'caption': message_data['imageMessage'].get('caption', '')
            }
        
        elif message_data.get('typeMessage') == 'audioMessage':
            # Voice note
            msg_content = {
                'type': 'voice',
                'url': message_data.get('downloadUrl'),
                'mimeType': message_data.get('mimeType', 'audio/ogg')
            }
        
        logger.info(f"Received {msg_content.get('type', 'unknown')} message from {chat_id}")
        
        if msg_content and msg_content.get('type'):
            # Process message in background thread
            def process_and_respond():
                try:
                    reply = process_message(chat_id, msg_content)
                    success = send_whatsapp_message(chat_id, reply)
                    
                    if not success:
                        logger.error(f"Failed to send response to {chat_id}")
                        # Retry once after 3 seconds
                        asyncio.sleep(3)
                        send_whatsapp_message(chat_id, reply)
                        
                except Exception as e:
                    logger.error(f"Error in process_and_respond: {e}", exc_info=True)
                    error_message = (
                        "I apologize for the technical difficulty. "
                        "Please try your question again or contact support@schoolinka.com"
                    )
                    send_whatsapp_message(chat_id, error_message)
            
            # Start processing thread
            thread = threading.Thread(target=process_and_respond)
            thread.daemon = True
            thread.start()
            
            return jsonify({"status": "success", "message": "processing"}), 200
        else:
            logger.warning(f"Unsupported message type from {chat_id}")
            return jsonify({"status": "ignored", "reason": "unsupported_type"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/user/<chat_id>', methods=['GET'])
def get_user_info(chat_id):
    """Get user profile and recent conversation history"""
    try:
        user = db.get_user(chat_id)
        if not user:
            return jsonify({
                "status": "error",
                "message": "User not found"
            }), 404
        
        history = db.get_history(chat_id, limit=10)
        
        # Remove sensitive info
        safe_user = {k: v for k, v in user.items() if k not in ['phone_number']}
        
        return jsonify({
            "status": "success",
            "user": safe_user,
            "recent_messages": history,
            "total_conversations": len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@app.route('/test', methods=['POST'])
def test():
    """Test endpoint for development"""
    try:
        data = request.get_json()
        chat_id = data.get('chat_id', 'test_user_12345')
        message = data.get('message', 'Hello, how can you help me?')
        msg_type = data.get('type', 'text')
        
        response = process_message(chat_id, {
            'type': msg_type,
            'text': message
        })
        
        user = db.get_user(chat_id)
        
        return jsonify({
            "status": "success",
            "response": response,
            "user": user,
            "message_length": len(response),
            "response_formatted": True
        })
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        with db.get_conn() as conn:
            cursor = conn.cursor()
            
            # Total users
            cursor.execute('SELECT COUNT(*) as total FROM users')
            total_users = cursor.fetchone()['total']
            
            # Active users (last 7 days)
            cursor.execute('''
                SELECT COUNT(*) as active 
                FROM users 
                WHERE last_interaction > datetime('now', '-7 days')
            ''')
            active_users = cursor.fetchone()['active']
            
            # Total conversations
            cursor.execute('SELECT COUNT(*) as total FROM conversations')
            total_messages = cursor.fetchone()['total']
            
            # Intent distribution
            cursor.execute('''
                SELECT intent, COUNT(*) as count 
                FROM conversations 
                WHERE intent IS NOT NULL 
                GROUP BY intent 
                ORDER BY count DESC
            ''')
            intent_stats = [dict(row) for row in cursor.fetchall()]
            
        return jsonify({
            "status": "success",
            "statistics": {
                "total_users": total_users,
                "active_users_7d": active_users,
                "total_messages": total_messages,
                "intent_distribution": intent_stats
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/broadcast', methods=['POST'])
def broadcast_message():
    """Broadcast message to all users (admin only)"""
    try:
        data = request.get_json()
        admin_key = data.get('admin_key')
        message = data.get('message')
        
        # Simple admin authentication
        if admin_key != os.getenv('ADMIN_KEY', 'your_secure_admin_key_here'):
            return jsonify({
                "status": "error",
                "message": "Unauthorized"
            }), 401
        
        if not message:
            return jsonify({
                "status": "error",
                "message": "Message is required"
            }), 400
        
        # Get all users
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT chat_id FROM users WHERE profile_complete = 1')
            users = cursor.fetchall()
        
        # Send to all users
        success_count = 0
        for user in users:
            chat_id = user['chat_id']
            if send_whatsapp_message(chat_id, message):
                success_count += 1
            asyncio.sleep(1)  # Rate limiting
        
        return jsonify({
            "status": "success",
            "sent": success_count,
            "total": len(users)
        })
        
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting AI Coach - Schoolinka Enhanced Version 2.0")
    logger.info("=" * 60)
    logger.info(f"Database: {CONFIG['DB_PATH']}")
    logger.info(f"Pinecone Index: coach")
    logger.info(f"Model: gemini-2.0-flash-exp")
    logger.info(f"Apps Script URL: {CONFIG['APPS_SCRIPT_URL'][:50]}...")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)