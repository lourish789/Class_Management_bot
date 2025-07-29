import os
import asyncio
import nest_asyncio
from datetime import datetime
import re
from flask import Flask, request, jsonify
import threading
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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw")
GREEN_API_ID = os.getenv("GREEN_API_ID", "7105287498")
GREEN_API_TOKEN = os.getenv("GREEN_API_TOKEN", "0017430b3b204cf28ac14a41cc5ede0ce8e5a68d91134d5fbe")

# Check for missing API keys
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing.")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key is missing.")

# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("coach")
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Initialize Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_tokens=1000
)

# Define system prompt template
system_prompt_template = """
Your name is Nigerian Teaching Coach Chatbot. You are a professional teaching expert for Nigerian schools effective in class management, student handling, stress handling and teaching responsibilities. Answer questions very very briefly and accurately. Use the following information to answer the user's question:

{doc_content}

Provide very brief accurate and helpful response based on the provided information and your expertise. But explain concisely if need be.
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

    def generate_coaching_response(self, user_input, conversation_history):
        try:
            # Convert conversation history to langchain format
            chat_history = []
            for entry in conversation_history[-10:]:  # Use last 10 entries
                if entry["role"] == "user":
                    chat_history.append(("human", entry["message"]))
                elif entry["role"] == "assistant":
                    chat_history.append(("ai", entry["message"]))

            # Generate response
            response = self.chain.run(
                input=user_input,
                chat_history=chat_history,
                doc_content=""  # Will be filled from Pinecone results
            )

            return response.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble processing your request right now. Please try again."

# Initialize the teacher AI
teacher_ai = TeacherAI(llm, system_prompt_template)

# Initialize WhatsApp bot
bot = GreenAPIBot(
    GREEN_API_ID, GREEN_API_TOKEN,
    debug_mode=True, bot_debug_mode=True
)

# Store conversation history and teacher info for each user
conversation_histories = {}  # chat_id: {"history": [...], "name": "Teacher", "class": "Primary 3", "initialized": False}

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
    # Add name only for the first few interactions or occasionally
    if not teacher_info.get("initialized", False):
        return True
    # Add name occasionally (every 10th message) to maintain personal touch
    return message_count % 10 == 0

def process_message(chat_id, user_message):
    """Process incoming message and generate response"""
    try:
        if not user_message.strip():
            return "I didn't catch that. Could you say that again? 🤔"

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
                    "How can I support you today? 📚✨"
                )
            else:
                # Show welcome message for any first interaction
                return (
                    "👋 Hello! I'm **Coach bot**, your friendly AI teaching coach assistant from Schoolinka.\n\n"
                    "Before we dive in, could you please tell me your **name** and the **class you teach**? "
                    "For example: `My name is Sarah and I teach Primary 3.`\n\n"
                    "This helps me provide more personalized support for your classroom! 🏫💡\n\n"
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

        # Trim history to last 20 messages
        if len(teacher_info["history"]) > 20:
            teacher_info["history"] = teacher_info["history"][-20:]

        # Embed the user's question
        query_embed = embed_model.embed_query(user_message)
        query_embed = [float(val) for val in query_embed]

        # Query Pinecone for relevant documents
        results = pinecone_index.query(
            vector=query_embed,
            top_k=5,
            include_values=False,
            include_metadata=True
        )

        # Extract document contents
        doc_contents = []
        for match in results.get('matches', []):
            text = match['metadata'].get('text', '')
            if text:
                doc_contents.append(text)

        doc_content = "\n".join(doc_contents) if doc_contents else "No additional information found."

        # Create enhanced prompt with context
        enhanced_prompt = f"""
        You are Coach bot, a supportive and highly experienced AI teaching coach for Nigerian teachers, an initiative of Schoolinka.
        Your goal is to provide practical, empathetic, and contextually relevant advice for Nigerian classrooms, considering challenges like large class sizes, limited resources, and power outages.
        Reference the provided information where appropriate, but also use your general teaching expertise.
        Keep your responses concise and actionable, providing clear examples relevant to the Nigerian educational setting.
        Always maintain a positive and encouraging tone.

        Teacher Information:
        - Name: {teacher_info.get('name', 'Teacher')}
        - Class: {teacher_info.get('class', 'Not specified')}

        Relevant Information:
        {doc_content}

        Teacher's Question: {user_message}
        """

        # Generate AI response
        ai_response = teacher_ai.generate_coaching_response(enhanced_prompt, teacher_info["history"])

        # Add name prefix only occasionally to avoid repetition
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
        if len(teacher_info["history"]) > 20:
            teacher_info["history"] = teacher_info["history"][-20:]

        return ai_response

    except Exception as e:
        print(f"Error processing message: {e}")
        return "Oops, something went wrong. Please try again shortly. 💡"

# Handle all messages (no specific command required)
@bot.router.message()
def ai_coaching_handler(notification: Notification) -> None:
    try:
        message_data = notification.event.get("messageData", {})
        text_data = message_data.get("textMessageData", {})
        user_message = text_data.get("textMessage", "")
        chat_id = notification.chat

        response = process_message(chat_id, user_message)
        notification.answer(response)

    except Exception as e:
        print(f"Error in AI handler: {e}")
        notification.answer("Oops, something went wrong. Please try again shortly. 💡")

# Flask routes for webhook and health check
@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "WhatsApp Teaching Coach Bot is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming webhook from Green API"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400

        # Process the webhook data
        # This is a basic implementation - you might need to adjust based on Green API's webhook format
        if data.get('typeWebhook') == 'incomingMessageReceived':
            message_data = data.get('messageData', {})
            text_data = message_data.get('textMessageData', {})
            user_message = text_data.get('textMessage', '')
            chat_id = data.get('senderData', {}).get('chatId', '')

            if user_message and chat_id:
                response = process_message(chat_id, user_message)
                # You would send the response back via Green API here
                # This depends on your Green API setup

        return jsonify({"status": "success", "message": "Webhook processed"})

    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get bot statistics"""
    total_users = len(conversation_histories)
    initialized_users = sum(1 for info in conversation_histories.values() if info.get("initialized", False))

    return jsonify({
        "total_users": total_users,
        "initialized_users": initialized_users,
        "active_conversations": total_users
    })

def run_bot():
    """Run the WhatsApp bot in a separate thread"""
    print("Starting WhatsApp Teaching Coach Bot...")
    bot.run_forever()

def run_flask():
    """Run Flask app"""
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # Run Flask app
    print("Starting Flask server...")
    run_flask()
