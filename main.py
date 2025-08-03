import os
import asyncio
import nest_asyncio
from datetime import datetime
import re
from flask import Flask, request, jsonify
import threading
import requests
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from pinecone import Pinecone
from whatsapp_chatbot_python import GreenAPIBot, Notification

# Apply nest_asyncio patch
nest_asyncio.apply()

# Flask app initialization
app = Flask(__name__)

# Get API keys from environment variables for security
PINECONE_API_KEY = "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg" #os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw" #os.getenv("GOOGLE_API_KEY")
GREEN_API_ID = "7105287498" #os.getenv("GREEN_API_ID")
GREEN_API_TOKEN = "0017430b3b204cf28ac14a41cc5ede0ce8e5a68d91134d5fbe" #os.getenv("GREEN_API_TOKEN")

# Check for missing API keys
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing from environment variables.")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key is missing from environment variables.")
if not GREEN_API_ID:
    raise ValueError("Green API ID is missing from environment variables.")
if not GREEN_API_TOKEN:
    raise ValueError("Green API Token is missing from environment variables.")

# Initialize Pinecone and embedding model
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("coach")
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    print("Pinecone and embeddings initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    raise

# Initialize Google Generative AI
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # Updated to a more stable model
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_tokens=800  # Reduced for more concise responses
    )
    print("Google Generative AI initialized successfully")
except Exception as e:
    print(f"Error initializing Google AI: {e}")
    raise

# Define system prompt template
system_prompt_template = """
Your name is Nigerian Teaching Coach Chatbot. You are a professional teaching expert for Nigerian schools effective in class management, student handling, stress handling and teaching responsibilities. Answer questions very briefly and accurately. Use the following information to answer the user's question:

{doc_content}

Provide brief accurate and helpful response based on the provided information and your expertise. Avoid using excessive quotation marks or markdown formatting. Keep responses natural and conversational.
"""

class TeacherAI:
    def __init__(self, llm, system_prompt_template):
        self.llm = llm
        self.system_prompt_template = system_prompt_template

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Create chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=False
        )

    def generate_coaching_response(self, user_input, conversation_history, doc_content=""):
        try:
            # Convert conversation history to langchain format
            chat_history = []
            for entry in conversation_history[-6:]:  # Reduced to last 6 entries for better performance
                if entry["role"] == "user":
                    chat_history.append(("human", entry["message"]))
                elif entry["role"] == "assistant":
                    chat_history.append(("ai", entry["message"]))

            # Generate response
            response = self.chain.run(
                input=user_input,
                chat_history=chat_history,
                doc_content=doc_content
            )

            # Clean up response - remove excessive quotation marks and markdown
            cleaned_response = self.clean_response(response.strip())
            return cleaned_response

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble processing your request right now. Please try again."

    def clean_response(self, response):
        """Clean up response to remove excessive formatting"""
        # Remove excessive quotation marks around words
        response = re.sub(r'`([^`]+)`', r'\1', response)  # Remove backticks
        response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)  # Remove bold markdown
        response = re.sub(r'\*([^*]+)\*', r'\1', response)  # Remove italic markdown
        response = re.sub(r'"([^"]+)"', r'\1', response)  # Remove unnecessary quotes
        response = re.sub(r'#{1,6}\s*', '', response)  # Remove markdown headers

        # Clean up multiple spaces and newlines
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = re.sub(r'\s+', ' ', response)

        return response.strip()

# Initialize the teacher AI
teacher_ai = TeacherAI(llm, system_prompt_template)

# Green API helper class for sending messages
class GreenAPIHelper:
    def __init__(self, api_id, api_token):
        self.api_id = api_id
        self.api_token = api_token
        self.base_url = f"https://7103.api.greenapi.com/waInstance{api_id}"

    def send_message(self, chat_id, message):
        """Send message via Green API"""
        try:
            url = f"{self.base_url}/sendMessage/{self.api_token}"

            payload = {
                "chatId": chat_id,
                "message": message
            }

            headers = {
                'Content-Type': 'application/json'
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                print(f"Message sent successfully to {chat_id}")
                return True
            else:
                print(f"Failed to send message: {response.status_code}, {response.text}")
                return False

        except Exception as e:
            print(f"Error sending message: {e}")
            return False

# Initialize Green API helper
green_api_helper = GreenAPIHelper(GREEN_API_ID, GREEN_API_TOKEN)

# Initialize WhatsApp bot with error handling
try:
    bot = GreenAPIBot(
        GREEN_API_ID, GREEN_API_TOKEN,
        debug_mode=True,
        bot_debug_mode=True
    )
    print("WhatsApp bot initialized successfully")
except Exception as e:
    print(f"Error initializing WhatsApp bot: {e}")
    bot = None

# Store conversation history and teacher info for each user
conversation_histories = {}

def extract_name_and_class(message):
    """Extract teacher name and class from message"""
    message_lower = message.lower()

    # Pattern for "my name is X and I teach Y"
    pattern1 = r"my name is\s+([^,\n]+?)\s+and\s+i teach\s+([^,\n]+)"
    match1 = re.search(pattern1, message_lower)
    if match1:
        return match1.group(1).strip(), match1.group(2).strip()

    # Pattern for "I am X, I teach Y"
    pattern2 = r"i am\s+([^,\n]+?)[,\s]+i teach\s+([^,\n]+)"
    match2 = re.search(pattern2, message_lower)
    if match2:
        return match2.group(1).strip(), match2.group(2).strip()

    # Pattern for "X teaching Y" or "X teaches Y"
    pattern3 = r"([a-zA-Z\s]+?)\s+teach(?:ing|es)?\s+([a-zA-Z0-9\s]+)"
    match3 = re.search(pattern3, message_lower)
    if match3 and len(match3.group(1).strip().split()) <= 3:
        return match3.group(1).strip(), match3.group(2).strip()

    return None, None

def should_add_name_prefix(teacher_info, message_count):
    """Determine if we should add the teacher's name to the response"""
    if not teacher_info.get("initialized", False):
        return True
    return message_count % 15 == 0  # Less frequent name usage

def process_message(chat_id, user_message):
    """Process incoming message and generate response"""
    try:
        if not user_message.strip():
            return "I didn't catch that. Could you say that again?"

        # Initialize history if new user
        if chat_id not in conversation_histories:
            conversation_histories[chat_id] = {
                "history": [],
                "name": None,
                "class": None,
                "initialized": False,
                "message_count": 0
            }

        teacher_info = conversation_histories[chat_id]
        teacher_info["message_count"] += 1

        # If not initialized, show welcome and try to extract info
        if not teacher_info["initialized"]:
            name, class_taught = extract_name_and_class(user_message)

            if name and class_taught:
                teacher_info["name"] = name.title()
                teacher_info["class"] = class_taught.title()
                teacher_info["initialized"] = True

                return (
                    f"Great to meet you, {teacher_info['name']}! I've noted that you teach {teacher_info['class']}.\n\n"
                    "I'm Coach bot, your AI teaching assistant from Schoolinka. "
                    "I'm here to help with classroom management, student engagement, lesson planning, and any teaching challenges you face.\n\n"
                    "How can I support you today?"
                )
            else:
                return (
                    "Hello! I'm Coach bot, your friendly AI teaching coach assistant from Schoolinka.\n\n"
                    "Before we dive in, could you please tell me your name and the class you teach? "
                    "For example: My name is Sarah and I teach Primary 3.\n\n"
                    "This helps me provide more personalized support for your classroom!\n\n"
                    "Or feel free to ask me any teaching question right away!"
                )

        # For initialized users, process normally but try to extract info if missing
        if (teacher_info["name"] is None or teacher_info["class"] is None) and not teacher_info["initialized"]:
            name, class_taught = extract_name_and_class(user_message)
            if name and class_taught:
                teacher_info["name"] = name.title()
                teacher_info["class"] = class_taught.title()
                teacher_info["initialized"] = True

        # Add user message to conversation history
        teacher_info["history"].append({
            "role": "user",
            "message": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # Trim history to last 15 messages for better performance
        if len(teacher_info["history"]) > 15:
            teacher_info["history"] = teacher_info["history"][-15:]

        # Get relevant documents from Pinecone
        doc_content = ""
        try:
            # Embed the user's question
            query_embed = embed_model.embed_query(user_message)
            query_embed = [float(val) for val in query_embed]

            # Query Pinecone for relevant documents
            results = pinecone_index.query(
                vector=query_embed,
                top_k=3,  # Reduced for better performance
                include_values=False,
                include_metadata=True
            )

            # Extract document contents
            doc_contents = []
            for match in results.get('matches', []):
                text = match['metadata'].get('text', '')
                if text:
                    doc_contents.append(text)

            doc_content = "\n".join(doc_contents) if doc_contents else ""
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            doc_content = ""

        # Create enhanced prompt with context
        enhanced_prompt = f"""
        You are Coach bot, a supportive and experienced AI teaching coach for Nigerian teachers, an initiative of Schoolinka.
        Provide practical, empathetic advice for Nigerian classrooms. Keep responses concise and actionable.
        Always maintain a positive and encouraging tone. Avoid using excessive quotation marks or markdown formatting.

        Teacher Information:
        - Name: {teacher_info.get('name', 'Teacher')}
        - Class: {teacher_info.get('class', 'Not specified')}

        Relevant Information:
        {doc_content}

        Teacher's Question: {user_message}
        """

        # Generate AI response
        ai_response = teacher_ai.generate_coaching_response(enhanced_prompt, teacher_info["history"], doc_content)

        # Add name prefix only occasionally
        if teacher_info.get("name") and should_add_name_prefix(teacher_info, teacher_info["message_count"]):
            if not ai_response.lower().startswith(teacher_info["name"].lower()):
                ai_response = f"{teacher_info['name']}, {ai_response}"

        # Add AI response to history
        teacher_info["history"].append({
            "role": "assistant",
            "message": ai_response,
            "timestamp": datetime.now().isoformat()
        })

        # Trim history again after adding assistant's response
        if len(teacher_info["history"]) > 15:
            teacher_info["history"] = teacher_info["history"][-15:]

        return ai_response

    except Exception as e:
        print(f"Error processing message: {e}")
        return "Oops, something went wrong. Please try again shortly."

# Handle all messages (only if bot is initialized)
if bot:
    @bot.router.message()
    def ai_coaching_handler(notification: Notification) -> None:
        try:
            message_data = notification.event.get("messageData", {})
            text_data = message_data.get("textMessageData", {})
            user_message = text_data.get("textMessage", "")
            chat_id = notification.chat

            if user_message:
                response = process_message(chat_id, user_message)
                notification.answer(response)

        except Exception as e:
            print(f"Error in AI handler: {e}")
            notification.answer("Oops, something went wrong. Please try again shortly.")

# Flask routes for webhook and health check
@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "WhatsApp Teaching Coach Bot is running",
        "timestamp": datetime.now().isoformat(),
        "bot_status": "initialized" if bot else "failed_to_initialize"
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming webhook from Green API"""
    try:
        data = request.get_json()
        print(f"Received webhook data: {json.dumps(data, indent=2)}")

        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400

        # Handle incoming message
        if data.get('typeWebhook') == 'incomingMessageReceived':
            message_data = data.get('messageData', {})
            sender_data = data.get('senderData', {})

            # Extract message details
            chat_id = sender_data.get('chatId', '')
            sender = sender_data.get('sender', '')

            # Handle text messages
            text_data = message_data.get('textMessageData', {})
            user_message = text_data.get('textMessage', '')

            if user_message and chat_id:
                print(f"Processing message from {sender}: {user_message}")
                response = process_message(chat_id, user_message)

                # Send response back via Green API
                success = green_api_helper.send_message(chat_id, response)

                if success:
                    return jsonify({"status": "success", "message": "Message processed and response sent"})
                else:
                    return jsonify({"status": "error", "message": "Failed to send response"}), 500

        return jsonify({"status": "success", "message": "Webhook processed"})

    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/send_message', methods=['POST'])
def send_message_endpoint():
    """Manual endpoint to send messages for testing"""
    try:
        data = request.get_json()
        chat_id = data.get('chat_id')
        message = data.get('message')

        if not chat_id or not message:
            return jsonify({"status": "error", "message": "chat_id and message are required"}), 400

        success = green_api_helper.send_message(chat_id, message)

        if success:
            return jsonify({"status": "success", "message": "Message sent"})
        else:
            return jsonify({"status": "error", "message": "Failed to send message"}), 500

    except Exception as e:
        print(f"Send message error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get bot statistics"""
    total_users = len(conversation_histories)
    initialized_users = sum(1 for info in conversation_histories.values() if info.get("initialized", False))

    return jsonify({
        "total_users": total_users,
        "initialized_users": initialized_users,
        "active_conversations": total_users,
        "bot_status": "running" if bot else "not_running"
    })

def run_bot():
    """Run the WhatsApp bot in a separate thread"""
    if bot:
        try:
            print("Starting WhatsApp Teaching Coach Bot...")
            bot.run_forever()
        except Exception as e:
            print(f"Error running bot: {e}")
    else:
        print("Bot not initialized, skipping bot thread")

def run_flask():
    """Run Flask app"""
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    # Start the bot in a separate thread only if bot is initialized
    if bot:
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        print("Bot thread started")
    else:
        print("Skipping bot thread - using webhook mode only")

    # Run Flask app
    run_flask()
