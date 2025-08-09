import os
import asyncio
import nest_asyncio
from datetime import datetime
import re
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

# Apply nest_asyncio patch
nest_asyncio.apply()

# Initialize Flask app
app = Flask(__name__)

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw")
GREEN_API_ID_INSTANCE = os.getenv("GREEN_API_ID_INSTANCE", "7105287498")
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
    model="gemini-2.0-flash-exp",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_tokens=1000
)

# Define system prompt template
system_prompt_template = """
Your name is Nigerian Teaching Coach Chatbot. You are a professional teaching expert for Nigerian schools effective in class management, student handling, stress handling and teaching responsibilities. Answer questions very very briefly and accurately. Use the following information to answer the user's question:

{doc_content}

Provide very brief accurate and helpful response based on the provided information and your expertise. But explain concisely if need be. Never use asterisks (*) in your responses.
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

            # Remove asterisks from response
            response = response.replace("*", "")
            return response.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble processing your request right now. Please try again."

# Initialize the teacher AI
teacher_ai = TeacherAI(llm, system_prompt_template)

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

def clean_response(response, teacher_name=None):
    """Clean response by removing asterisks and unnecessary name repetitions"""
    # Remove asterisks
    response = response.replace("*", "")

    # Remove repetitive mentions of teacher name if it appears more than once
    if teacher_name:
        # Count occurrences of teacher name
        name_count = response.lower().count(teacher_name.lower())
        if name_count > 1:
            # Keep only the first occurrence
            response = response.replace(f"{teacher_name}, ", "", name_count - 1)

    return response.strip()

def process_message(chat_id, user_message):
    """Process incoming message and generate response"""
    try:
        # Initialize history if new user
        if chat_id not in conversation_histories:
            conversation_histories[chat_id] = {
                "history": [],
                "name": None,
                "class": None,
                "initialized": False
            }

        teacher_info = conversation_histories[chat_id]

        # If this is the first interaction, show welcome message
        if not teacher_info["initialized"]:
            teacher_info["initialized"] = True

            # Check if user already provided name and class in first message
            name, class_taught = extract_name_and_class(user_message)

            if name and class_taught:
                teacher_info["name"] = name.title()
                teacher_info["class"] = class_taught.title()

                welcome_response = (
                    f"Hello! I'm Coach bot, your friendly AI teaching coach assistant.\n\n"
                    f"An initiative of Schoolinka. For more information: https://www.schoolinka.com/\n\n"
                    f"Nice to meet you! I've noted that you teach {teacher_info['class']}. "
                    f"How can I support you today with your classroom or teaching journey? 🏫💡"
                )
                return welcome_response
            else:
                welcome_message = (
                    "👋 Hello! I'm Coach bot, your friendly AI teaching coach assistant.\n\n"
                    "An initiative of Schoolinka. For more information: https://www.schoolinka.com/\n\n"
                    "Before we begin, could you please tell me your name and the class you teach? "
                    "For example: My name is Sarah and I teach Primary 3.\n\n"
                    "Once I know that, I can provide more personalized support for your classroom journey! 🏫💡"
                )
                return welcome_message

        # If teacher name and class not provided, try to extract them
        if teacher_info["name"] is None or teacher_info["class"] is None:
            name, class_taught = extract_name_and_class(user_message)

            if name and class_taught:
                teacher_info["name"] = name.title()
                teacher_info["class"] = class_taught.title()

                return (
                    f"Thank you! I've saved that you teach {teacher_info['class']}. "
                    "How can I support you today with your class or teaching journey?"
                )
            else:
                return "Please tell me your name and class like this: My name is James and I teach JSS 1 😊"

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
        Always maintain a positive and encouraging tone. Never use asterisks in your responses.

        Teacher Information:
        - Name: {teacher_info['name']}
        - Class: {teacher_info['class']}

        Relevant Information:
        {doc_content}

        Teacher's Question: {user_message}
        """

        # Generate AI response
        ai_response = teacher_ai.generate_coaching_response(enhanced_prompt, teacher_info["history"])

        # Clean the response (remove asterisks and handle name repetition)
        ai_response = clean_response(ai_response, teacher_info["name"])

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
        print(f"Error in process_message: {e}")
        return "Oops, something went wrong. Please try again shortly. 💡"

def send_message_via_green_api(chat_id, message):
    """Send message via Green API"""
    try:
        #url = f"https://api.green-api.com/waInstance{GREEN_API_ID_INSTANCE}/sendMessage/{GREEN_API_TOKEN}"
        url = "https://7105.api.greenapi.com/waInstance7105287498/sendMessage/0017430b3b204cf28ac14a41cc5ede0ce8e5a68d91134d5fbe"
        payload = {
            "chatId": chat_id,
            "message": message
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print(f"Message sent successfully to {chat_id}")
            return True
        else:
            print(f"Failed to send message: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending message via Green API: {e}")
        return False

# Initialize WhatsApp bot (for message handling structure)
bot = GreenAPIBot(
    GREEN_API_ID_INSTANCE, GREEN_API_TOKEN,
    debug_mode=True, bot_debug_mode=True
)

# Flask routes for webhook and health check
@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "WhatsApp Teaching Coach Bot",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming webhook from Green API"""
    try:
        data = request.get_json()
        print(f"Webhook received: {json.dumps(data, indent=2)}")

        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400

        # Handle incoming message
        if data.get('typeWebhook') == 'incomingMessageReceived':
            message_data = data.get('messageData', {})
            sender_data = data.get('senderData', {})
            
            # Extract message details
            chat_id = sender_data.get('chatId', '')
            
            # Handle text messages
            if 'textMessageData' in message_data:
                text_data = message_data.get('textMessageData', {})
                user_message = text_data.get('textMessage', '')
                
                if user_message and chat_id:
                    print(f"Processing message from {chat_id}: {user_message}")
                    
                    # Process the message and get response
                    reply = process_message(chat_id, user_message)
                    
                    # Send response back
                    if reply:
                        success = send_message_via_green_api(chat_id, reply)
                        if not success:
                            print(f"Failed to send reply to {chat_id}")
                    
                    return jsonify({"status": "success", "message": "Message processed"}), 200

        # Handle other webhook types if needed
        return jsonify({"status": "success", "message": "Webhook received"}), 200

    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/status')
def bot_status():
    """Get bot status and statistics"""
    return jsonify({
        "active_conversations": len(conversation_histories),
        "bot_status": "running",
        "timestamp": datetime.now().isoformat(),
        "green_api_instance": GREEN_API_ID_INSTANCE
    })

@app.route('/test', methods=['POST'])
def test_message():
    """Test endpoint for manual message testing"""
    try:
        data = request.get_json()
        chat_id = data.get('chat_id', 'test_user')
        message = data.get('message', 'Hello')
        
        response = process_message(chat_id, message)
        
        return jsonify({
            "status": "success",
            "chat_id": chat_id,
            "user_message": message,
            "bot_response": response
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    print("Starting WhatsApp Teaching Coach Bot...")
    print(f"Green API Instance: {GREEN_API_ID_INSTANCE}")
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
