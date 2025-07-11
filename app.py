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
