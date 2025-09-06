import os
import asyncio
import nest_asyncio
from datetime import datetime, timedelta
import re
#import sqlite3
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

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw")
GREEN_API_ID_INSTANCE = os.getenv("GREEN_API_ID_INSTANCE", "7105287498")
GREEN_API_TOKEN = os.getenv("GREEN_API_TOKEN", "0017430b3b204cf28ac14a41cc5ede0ce8e5a68d91134d5fbe")

# Database configuration
DB_PATH = "teaching_coach.db"
MAX_CONVERSATION_HISTORY = 20  # Keep last 20 exchanges per user

# Validate API keys
if not all([PINECONE_API_KEY, GOOGLE_API_KEY, GREEN_API_ID_INSTANCE, GREEN_API_TOKEN]):
    logger.error("Missing required API keys!")
    raise ValueError("Required API keys are missing.")

# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("coach")
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Initialize Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_tokens=1000
)

class DatabaseManager:
    """Manages SQLite database operations for conversation history and user profiles"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    chat_id TEXT PRIMARY KEY,
                    teacher_name TEXT,
                    class_taught TEXT,
                    phone_number TEXT,
                    profile_complete BOOLEAN DEFAULT FALSE,
                    first_interaction BOOLEAN DEFAULT TRUE,
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
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_chat_id ON conversations (chat_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_chat_id ON analytics (chat_id)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        try:
            yield conn
        finally:
            conn.close()
    
    def get_user_profile(self, chat_id: str) -> Optional[Dict]:
        """Get user profile by chat_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE chat_id = ?', (chat_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def create_or_update_user(self, chat_id: str, teacher_name: str = None, 
                            class_taught: str = None, phone_number: str = None) -> bool:
        """Create new user or update existing user profile"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if user exists
                cursor.execute('SELECT chat_id FROM users WHERE chat_id = ?', (chat_id,))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing user
                    update_fields = []
                    update_values = []
                    
                    if teacher_name:
                        update_fields.append('teacher_name = ?')
                        update_values.append(teacher_name)
                    if class_taught:
                        update_fields.append('class_taught = ?')
                        update_values.append(class_taught)
                    if phone_number:
                        update_fields.append('phone_number = ?')
                        update_values.append(phone_number)
                    if teacher_name and class_taught:
                        update_fields.append('profile_complete = ?')
                        update_values.append(True)
                    
                    update_fields.append('last_interaction = ?')
                    update_values.append(datetime.now())
                    update_fields.append('total_messages = total_messages + 1')
                    
                    update_values.append(chat_id)
                    
                    if update_fields:
                        query = f"UPDATE users SET {', '.join(update_fields)} WHERE chat_id = ?"
                        cursor.execute(query, update_values)
                else:
                    # Create new user
                    cursor.execute('''
                        INSERT INTO users (chat_id, teacher_name, class_taught, phone_number, 
                                         profile_complete, first_interaction, total_messages)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (chat_id, teacher_name, class_taught, phone_number,
                          bool(teacher_name and class_taught), True, 1))
                
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
                # Reverse to get chronological order
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
                ''', (chat_id, chat_id, MAX_CONVERSATION_HISTORY * 2))  # Keep user+assistant pairs
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
        """Get bot usage statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total users
                cursor.execute('SELECT COUNT(*) as total_users FROM users')
                total_users = cursor.fetchone()['total_users']
                
                # Complete profiles
                cursor.execute('SELECT COUNT(*) as complete_profiles FROM users WHERE profile_complete = 1')
                complete_profiles = cursor.fetchone()['complete_profiles']
                
                # Total messages
                cursor.execute('SELECT COUNT(*) as total_messages FROM conversations')
                total_messages = cursor.fetchone()['total_messages']
                
                # Active users (interacted in last 7 days)
                week_ago = datetime.now() - timedelta(days=7)
                cursor.execute('SELECT COUNT(*) as active_users FROM users WHERE last_interaction > ?', (week_ago,))
                active_users = cursor.fetchone()['active_users']
                
                return {
                    'total_users': total_users,
                    'complete_profiles': complete_profiles,
                    'total_messages': total_messages,
                    'active_users': active_users,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

class ConversationAnalyzer:
    """Analyzes conversation patterns and user intent"""
    
    @staticmethod
    def extract_intent(message: str) -> str:
        """Extract user intent from message"""
        message_lower = message.lower()
        
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
        if any(phrase in message_lower for phrase in ['my name is', 'i am', 'i teach']):
            return 'profile_setup'
        
        return 'general_inquiry'
    
    @staticmethod
    def analyze_conversation_context(history: List[Dict]) -> Dict:
        """Analyze conversation context for better responses"""
        if not history:
            return {'context': 'new_conversation', 'topics': [], 'sentiment': 'neutral'}
        
        topics = []
        recent_intents = []
        
        # Analyze last 5 exchanges
        for entry in history[-10:]:
            intent = entry.get('message_intent', '')
            if intent:
                recent_intents.append(intent)
        
        # Determine conversation context
        context = 'ongoing'
        if len(history) <= 2:
            context = 'early_conversation'
        elif 'teacher_wellbeing' in recent_intents:
            context = 'support_needed'
        elif len(set(recent_intents)) == 1:  # Same intent repeated
            context = 'focused_discussion'
        
        return {
            'context': context,
            'recent_intents': recent_intents,
            'message_count': len(history),
            'dominant_intent': max(set(recent_intents), key=recent_intents.count) if recent_intents else None
        }

class EnhancedTeacherAI:
    """Enhanced AI with better context awareness and response generation"""
    
    def __init__(self, llm, embed_model, pinecone_index, db_manager):
        self.llm = llm
        self.embed_model = embed_model
        self.pinecone_index = pinecone_index
        self.db_manager = db_manager
        self.analyzer = ConversationAnalyzer()
        
        # Enhanced system prompt
        self.system_prompt_template = """
        You are Coach bot, a highly experienced and empathetic AI teaching assistant specifically designed for Nigerian teachers.
        
        Your core responsibilities:
        1. Provide practical, actionable teaching advice tailored to Nigerian classrooms
        2. Consider challenges like large class sizes, limited resources, power outages, and diverse student backgrounds
        3. Maintain cultural sensitivity and understanding of the Nigerian educational system
        4. Offer emotional support while remaining professional
        5. Use conversation history to provide contextual and personalized responses
        
        Guidelines:
        - Keep responses concise but thorough when needed
        - Provide specific examples relevant to Nigerian schools
        - Acknowledge the teacher by name when appropriate
        - Reference previous conversations naturally
        - Never use asterisks (*) in responses
        - Be encouraging and supportive while being realistic
        - Suggest practical solutions that work with limited resources
        
        Teacher Information:
        - Name: {teacher_name}
        - Class: {class_taught}
        - Conversation Context: {conversation_context}
        
        Relevant Information from Database:
        {rag_content}
        
        Recent Conversation History:
        {conversation_summary}
        """
    
    def get_rag_content(self, user_message: str, intent: str = None) -> Tuple[str, List[str]]:
        """Get relevant content from Pinecone with source tracking"""
        try:
            # Enhance query based on intent
            enhanced_query = user_message
            if intent:
                enhanced_query = f"{intent} {user_message}"
            
            # Embed the enhanced query
            query_embed = self.embed_model.embed_query(enhanced_query)
            query_embed = [float(val) for val in query_embed]

            # Query Pinecone for relevant documents
            results = self.pinecone_index.query(
                vector=query_embed,
                top_k=5,
                include_values=False,
                include_metadata=True
            )

            # Extract document contents and sources
            doc_contents = []
            sources = []
            
            for match in results.get('matches', []):
                if match.get('score', 0) > 0.7:  # Only high-relevance matches
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
        """Generate contextual response based on user message and history"""
        try:
            # Extract intent
            intent = self.analyzer.extract_intent(user_message)
            
            # Analyze conversation context
            context = self.analyzer.analyze_conversation_context(conversation_history)
            
            # Get RAG content
            rag_content, sources = self.get_rag_content(user_message, intent)
            
            # Create conversation summary
            conversation_summary = self.create_conversation_summary(conversation_history)
            
            # Prepare enhanced prompt
            enhanced_prompt = self.system_prompt_template.format(
                teacher_name=user_profile.get('teacher_name', 'Teacher'),
                class_taught=user_profile.get('class_taught', 'your class'),
                conversation_context=context,
                rag_content=rag_content,
                conversation_summary=conversation_summary
            )
            
            # Create chat messages
            messages = [
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": f"Current question: {user_message}"}
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            ai_response = response.content.strip()
            
            # Remove asterisks and clean response
            ai_response = ai_response.replace("*", "").strip()
            
            # Add contextual elements based on conversation analysis
            if context['context'] == 'support_needed':
                ai_response += "\n\nRemember, taking care of yourself is essential for being the best teacher you can be. You're doing important work! 💪"
            elif context['context'] == 'early_conversation' and user_profile.get('teacher_name'):
                ai_response = f"Hi {user_profile['teacher_name']}! " + ai_response
            
            return ai_response, intent, sources
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return ("I'm having trouble processing your request right now. Please try again in a moment, "
                   "and feel free to reach out if you need immediate support."), "error", []

# Initialize components
db_manager = DatabaseManager(DB_PATH)
ai_coach = EnhancedTeacherAI(llm, embed_model, pinecone_index, db_manager)

def extract_profile_info(message: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract teacher name and class from message with improved patterns"""
    message_lower = message.lower().strip()
    
    patterns = [
        # "My name is X and I teach Y"
        r"my name is\s+([^,\n]+?)\s+and\s+i teach\s+([^,\n.!?]+)",
        # "I am X, I teach Y"
        r"i am\s+([^,\n]+?)[,\s]+i teach\s+([^,\n.!?]+)",
        # "I'm X and I teach Y"
        r"i'?m\s+([^,\n]+?)\s+and\s+i teach\s+([^,\n.!?]+)",
        # "Name: X, Class: Y" format
        r"name[:\s]+([^,\n]+?)[,\s]+class[:\s]+([^,\n.!?]+)",
        # "X teaches Y" or "X teaching Y"
        r"^([a-zA-Z\s]{2,25})\s+teach(?:es|ing)\s+([a-zA-Z0-9\s]+)$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            name = match.group(1).strip().title()
            class_taught = match.group(2).strip().title()
            
            # Validate name (should be reasonable length and alphabetic)
            if 2 <= len(name.split()) <= 4 and all(part.isalpha() for part in name.split()):
                return name, class_taught
    
    return None, None

def process_message(chat_id: str, user_message: str) -> str:
    """Enhanced message processing with database integration"""
    try:
        # Get or create user profile
        user_profile = db_manager.get_user_profile(chat_id)
        
        # Handle new user
        if not user_profile:
            db_manager.create_or_update_user(chat_id)
            user_profile = db_manager.get_user_profile(chat_id)
            
            # Log new user
            db_manager.log_analytics(chat_id, "new_user", {"first_message": user_message})
        
        # Update last interaction
        db_manager.create_or_update_user(chat_id)
        
        # Handle profile completion
        if not user_profile.get('profile_complete'):
            name, class_taught = extract_profile_info(user_message)
            
            if name and class_taught:
                # Update profile
                success = db_manager.create_or_update_user(
                    chat_id, teacher_name=name, class_taught=class_taught
                )
                
                if success:
                    # Save the profile setup message
                    intent = ConversationAnalyzer.extract_intent(user_message)
                    db_manager.save_message(chat_id, 'user', user_message, intent=intent)
                    
                    # Generate welcome response
                    welcome_msg = (
                        f"Hello {name}! I'm Coach bot, your AI teaching assistant. 🎓\n\n"
                        f"I see you teach {class_taught}. I'm here to support you with:\n"
                        f"• Teaching strategies and methods\n"
                        f"• Classroom management tips\n"
                        f"• Student assessment guidance\n"
                        f"• Stress management and wellbeing\n"
                        f"• Parent communication advice\n\n"
                        f"How can I help you today?"
                    )
                    
                    # Save welcome response
                    db_manager.save_message(chat_id, 'assistant', welcome_msg)
                    
                    # Log profile completion
                    db_manager.log_analytics(chat_id, "profile_completed", {
                        "teacher_name": name, "class_taught": class_taught
                    })
                    
                    return welcome_msg
                else:
                    return ("I had trouble saving your profile information. "
                           "Please try again with your name and class.")
            else:
                # First interaction or incomplete profile info
                if user_profile.get('first_interaction', True):
                    # Update first interaction flag
                    db_manager.create_or_update_user(chat_id)
                    
                    return (
                        "👋 Hello! I'm Coach bot, your AI teaching assistant for Nigerian schools.\n\n"
                        "To provide you with personalized support, please tell me your name and the class you teach.\n\n"
                        "For example:\n"
                        "• My name is Sarah and I teach Primary 3\n"
                        "• I'm James and I teach JSS 1\n\n"
                        "This will help me give you more relevant advice! 😊"
                    )
                else:
                    return (
                        "I still need your name and class information to help you better.\n\n"
                        "Please share like this: My name is [Your Name] and I teach [Your Class] 😊"
                    )
        
        # User profile is complete - process normal conversation
        # Get conversation history
        conversation_history = db_manager.get_conversation_history(chat_id)
        
        # Extract intent
        intent = ConversationAnalyzer.extract_intent(user_message)
        
        # Save user message
        db_manager.save_message(chat_id, 'user', user_message, intent=intent)
        
        # Generate AI response
        ai_response, response_intent, rag_sources = ai_coach.generate_response(
            user_message, user_profile, conversation_history
        )
        
        # Save AI response
        db_manager.save_message(
            chat_id, 'assistant', ai_response, 
            intent=response_intent, rag_sources=json.dumps(rag_sources)
        )
        
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
            "Please try again in a moment. If this continues, please let me know! 🔧"
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
    stats = db_manager.get_stats()
    return jsonify({
        "status": "healthy",
        "service": "Nigerian Teaching Coach Bot",
        "timestamp": datetime.now().isoformat(),
        "database_status": "connected",
        "stats": stats
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Enhanced webhook handler with better error handling"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400

        # Handle incoming message
        if data.get('typeWebhook') == 'incomingMessageReceived':
            message_data = data.get('messageData', {})
            sender_data = data.get('senderData', {})
            
            chat_id = sender_data.get('chatId', '')
            
            # Handle text messages only
            if 'textMessageData' in message_data:
                text_data = message_data.get('textMessageData', {})
                user_message = text_data.get('textMessage', '').strip()
                
                if user_message and chat_id:
                    logger.info(f"Processing message from {chat_id}: {user_message[:100]}...")
                    
                    # Process message in separate thread for better performance
                    def process_and_respond():
                        try:
                            reply = process_message(chat_id, user_message)
                            if reply:
                                success = send_message_via_green_api(chat_id, reply)
                                if not success:
                                    logger.error(f"Failed to send reply to {chat_id}")
                        except Exception as e:
                            logger.error(f"Error in process_and_respond: {e}")
                            # Send error message
                            error_msg = "I encountered an error processing your message. Please try again!"
                            send_message_via_green_api(chat_id, error_msg)
                    
                    # Run in background thread
                    thread = threading.Thread(target=process_and_respond)
                    thread.daemon = True
                    thread.start()
                    
                    return jsonify({"status": "success", "message": "Message being processed"}), 200

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
            "recent_history": conversation_history
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
            "recent_messages": conversation_history[-5:] if conversation_history else []
        })
        
    except Exception as e:
        logger.error(f"Error getting user info for {chat_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/user/<chat_id>/reset', methods=['POST'])
def reset_user(chat_id):
    """Reset user profile and conversation history"""
    try:
        # Check if user exists
        user_profile = db_manager.get_user_profile(chat_id)
        if not user_profile:
            return jsonify({"status": "error", "message": "User not found"}), 404
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Reset user profile
            cursor.execute('''
                UPDATE users SET 
                    teacher_name = NULL,
                    class_taught = NULL,
                    profile_complete = FALSE,
                    first_interaction = TRUE,
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

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get comprehensive analytics"""
    try:
        # Get date range from query params
        days = int(request.args.get('days', 7))
        start_date = datetime.now() - timedelta(days=days)
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Daily message counts
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as message_count
                FROM conversations
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (start_date,))
            daily_messages = [dict(row) for row in cursor.fetchall()]
            
            # Intent distribution
            cursor.execute('''
                SELECT message_intent, COUNT(*) as count
                FROM conversations
                WHERE message_type = 'user' AND message_intent IS NOT NULL
                AND timestamp >= ?
                GROUP BY message_intent
                ORDER BY count DESC
            ''', (start_date,))
            intent_distribution = [dict(row) for row in cursor.fetchall()]
            
            # Top active users
            cursor.execute('''
                SELECT u.teacher_name, u.class_taught, u.total_messages, u.last_interaction
                FROM users u
                WHERE u.profile_complete = 1 AND u.last_interaction >= ?
                ORDER BY u.total_messages DESC
                LIMIT 10
            ''', (start_date,))
            active_users = [dict(row) for row in cursor.fetchall()]
            
            # Response time analytics (simulated - you'd need to track actual response times)
            cursor.execute('''
                SELECT AVG(LENGTH(message_content)) as avg_response_length
                FROM conversations
                WHERE message_type = 'assistant' AND timestamp >= ?
            ''', (start_date,))
            avg_response_length = cursor.fetchone()['avg_response_length'] or 0
        
        return jsonify({
            "status": "success",
            "analytics": {
                "period_days": days,
                "daily_messages": daily_messages,
                "intent_distribution": intent_distribution,
                "active_users": active_users,
                "avg_response_length": round(avg_response_length, 2),
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/export/<chat_id>', methods=['GET'])
def export_conversation(chat_id):
    """Export conversation history for a user"""
    try:
        user_profile = db_manager.get_user_profile(chat_id)
        if not user_profile:
            return jsonify({"status": "error", "message": "User not found"}), 404
        
        conversation_history = db_manager.get_conversation_history(chat_id, limit=1000)
        
        export_data = {
            "user_profile": user_profile,
            "conversation_history": conversation_history,
            "export_date": datetime.now().isoformat(),
            "total_messages": len(conversation_history)
        }
        
        return jsonify({
            "status": "success",
            "export_data": export_data
        })
        
    except Exception as e:
        logger.error(f"Error exporting conversation for {chat_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/bulk_message', methods=['POST'])
def send_bulk_message():
    """Send message to multiple users (admin feature)"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        target_criteria = data.get('criteria', {})  # e.g., {"profile_complete": True}
        
        if not message:
            return jsonify({"status": "error", "message": "Message is required"}), 400
        
        # Build query based on criteria
        where_conditions = []
        params = []
        
        for key, value in target_criteria.items():
            if key in ['profile_complete', 'teacher_name', 'class_taught']:
                where_conditions.append(f"{key} = ?")
                params.append(value)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT chat_id, teacher_name FROM users WHERE {where_clause}", params)
            target_users = cursor.fetchall()
        
        # Send messages
        success_count = 0
        failed_count = 0
        
        for user in target_users:
            chat_id = user['chat_id']
            try:
                # Personalize message if teacher name is available
                personalized_message = message
                if user['teacher_name']:
                    personalized_message = f"Hi {user['teacher_name']}! {message}"
                
                success = send_message_via_green_api(chat_id, personalized_message)
                if success:
                    success_count += 1
                    # Log the bulk message
                    db_manager.log_analytics(chat_id, "bulk_message_received", {"message": message})
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error sending bulk message to {chat_id}: {e}")
                failed_count += 1
        
        return jsonify({
            "status": "success",
            "message": f"Bulk message sent to {success_count} users, {failed_count} failed",
            "details": {
                "total_targeted": len(target_users),
                "successful_sends": success_count,
                "failed_sends": failed_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error in bulk message: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/database/backup', methods=['GET'])
def backup_database():
    """Create a backup of the database"""
    try:
        import shutil
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"teaching_coach_backup_{timestamp}.db"
        
        shutil.copy2(DB_PATH, backup_filename)
        
        return jsonify({
            "status": "success",
            "message": f"Database backed up successfully",
            "backup_file": backup_filename,
            "timestamp": timestamp
        })
        
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Initialize WhatsApp bot for compatibility
bot = GreenAPIBot(
    GREEN_API_ID_INSTANCE, GREEN_API_TOKEN,
    debug_mode=False, bot_debug_mode=False
)

def initialize_database_with_sample_data():
    """Initialize database with sample teaching resources if empty"""
    try:
        stats = db_manager.get_stats()
        if stats.get('total_messages', 0) == 0:
            logger.info("Initializing database with sample data...")
            
            # Create a sample user for testing
            sample_chat_id = "sample_user_001"
            db_manager.create_or_update_user(
                sample_chat_id, 
                teacher_name="Sample Teacher", 
                class_taught="Primary 5"
            )
            
            # Add sample conversation
            db_manager.save_message(
                sample_chat_id, 'user', 
                "How can I manage a large class of 50 students?", 
                intent='classroom_management'
            )
            
            db_manager.save_message(
                sample_chat_id, 'assistant',
                "Managing a large class requires strategic planning. Here are some effective techniques for Nigerian classrooms...",
                intent='classroom_management'
            )
            
            logger.info("Sample data initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")

if __name__ == "__main__":
    logger.info("Starting Enhanced Nigerian Teaching Coach Bot...")
    logger.info(f"Green API Instance: {GREEN_API_ID_INSTANCE}")
    logger.info(f"Database: {DB_PATH}")
    
    # Initialize database with sample data if needed
    initialize_database_with_sample_data()
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting server on port {port}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
