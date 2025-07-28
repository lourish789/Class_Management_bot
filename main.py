import os
import asyncio
import nest_asyncio
from datetime import datetime
import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from pinecone import Pinecone
from whatsapp_chatbot_python import GreenAPIBot, Notification

# Apply nest_asyncio patch
nest_asyncio.apply()

# Get API keys
PINECONE_API_KEY = "pcsk_zRyjS_2FyS6uk3NsKW9AHPzDvvQPzANF2S3B67MS6UZ7ax6tnJfmCbLiYXrEcBJFHzcHg"
GOOGLE_API_KEY = "AIzaSyB3N9BHeIWs_8sdFK76PU-v9N6prcIq2Hw"

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
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_tokens=1000
)

# Define system prompt template
system_prompt_template = """
Your name is Nigerian Teaching Coach Chatbot. You are a professional teaching expert for Nigerian schools effective in class management, student handling, stress handling and teaching responsibilities. Answer questions very very briefly and accurately. Use the following information to answer the user's question:

{doc_content}

Provide very brief accurate and helpful health response based on the provided information and your expertise. But explain concisely if need be
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
    "7105287498", "0017430b3b204cf28ac14a41cc5ede0ce8e5a68d91134d5fbe",
    debug_mode=True, bot_debug_mode=True
)

# Store conversation history and teacher info for each user
conversation_histories = {}  # chat_id: {"history": [...], "name": "Teacher", "class": "Primary 3"}

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

@bot.router.message(command="start")
def message_handler(notification: Notification) -> None:
    chat_id = notification.chat

    # Initialize conversation structure
    conversation_histories[chat_id] = {
        "history": [],
        "name": None,
        "class": None
    }

    welcome_message = (
        "👋 Hello! I'm **Coach bot**, your friendly AI teaching coach assistant.\n\n"
        "An initiative of Schoolinka, For more information, https://www.schoolinka.com/ \n"
        "Before we begin, could you please tell me your **name** and the **class you teach**? "
        "For example: `My name is Sarah and I teach Primary 3.`\n\n"
        "Once I know that, I can provide more personalized support for your classroom journey! 🏫💡"
    )
    notification.answer(welcome_message)
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from model_inference import get_coach_response
from dotenv import load_dotenv
import os
import time

load_dotenv()

app = Flask(__name__)

# Session memory store: {sender_phone_number: list of messages}
session_store = {}

# Number of messages to retain in session history
SESSION_LIMIT = 5

@app.route('/whatsapp', methods=['POST'])
def whatsapp_bot():
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From', '')

    if not sender or not incoming_msg:
        return "No input received"

    # Initialize or update session history
    if sender not in session_store:
        session_store[sender] = [{"role": "system", "content": "You are a Nigerian classroom management coach who provides practical advice to teachers."}]
    
    # Append user message to session
    session_store[sender].append({"role": "user", "content": incoming_msg})

    # Limit session length
    if len(session_store[sender]) > SESSION_LIMIT * 2:
        session_store[sender] = [session_store[sender][0]] + session_store[sender][-SESSION_LIMIT*2:]

    # Get assistant response using current context
    reply_text = get_coach_response(session_store[sender])

    # Append assistant's reply to session
    session_store[sender].append({"role": "assistant", "content": reply_text})

    # Return via WhatsApp
    response = MessagingResponse()
    response.message(reply_text)
    return str(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
