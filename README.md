# LINKA AI - Schoolinka Teaching Assistant

LINKA AI is an intelligent WhatsApp chatbot initiative by Schoolinka, designed to support Nigerian teachers with teaching strategies, classroom management, and professional development.

## Features

- 🤖 **AI-Powered Assistance**: Uses Google Gemini 2.5 Flash for intelligent responses
- 📚 **RAG Integration**: Retrieves relevant teaching content from Pinecone vector database
- 👥 **Multi-User Support**: Handles concurrent users with isolated data and thread-safe processing
- 🗄️ **PostgreSQL Database**: Persistent storage of user profiles and conversation history
- ⏰ **90-Day Memory**: Automatic re-verification after 90 days to keep user data current
- 📊 **Real-time Logging**: Logs all interactions to Google Sheets for analytics
- 🔐 **Unique User IDs**: Each user gets a UUID to prevent data mixing
- 💬 **WhatsApp Integration**: Seamless integration via Green API

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: PostgreSQL (Render)
- **AI Model**: Google Gemini 2.5 Flash
- **Vector DB**: Pinecone
- **Messaging**: Green API (WhatsApp)
- **Logging**: Google Apps Script + Sheets

## Environment Variables

Create a `.env` file with the following variables:
```env
# Database
DATABASE_URL=your_postgresql_connection_string

# AI Services
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key

# WhatsApp Integration
GREEN_API_ID_INSTANCE=your_green_api_instance_id
GREEN_API_TOKEN=your_green_api_token

# Google Sheets Logging
APPS_SCRIPT_URL=your_google_apps_script_url
SPREADSHEET_ID=your_spreadsheet_id

# Optional
PORT=5000
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd linka-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. **Run the application**
```bash
python main.py
```

For production deployment:
```bash
gunicorn main:app --workers 4 --bind 0.0.0.0:5000
```

## Database Setup

The application automatically creates the required PostgreSQL tables on first run:

- `users` - User profiles and registration data
- `conversations` - Message history with user isolation
- `sessions` - Active conversation sessions

## API Endpoints

- `GET /` - Health check
- `POST /webhook` - WhatsApp webhook (Green API)
- `GET /user/<user_id>` - Get user profile and history
- `GET /stats` - System statistics
- `GET /health_check` - Detailed component health check
- `POST /test` - Test message processing
- `POST /test_rag` - Test RAG retrieval
- `POST /retry_sheets_logging` - Retry failed sheet logs

## User Registration Flow

1. User sends first message → LINKA AI greets and asks for name
2. Collects email address
3. Collects location (city/state)
4. Collects class taught
5. Profile complete → Ready for conversations

## 90-Day Re-verification

After 90 days of inactivity, users are automatically prompted to verify:
- Email address
- Location
- Class taught

This ensures LINKA AI provides relevant, up-to-date support.

## Features by Intent

LINKA AI can help with:
- **Teaching Strategies**: Lesson planning, engagement activities
- **Classroom Management**: Discipline, behavior control
- **Assessment**: Grading, feedback, evaluation methods
- **Wellbeing**: Teacher stress management, burnout prevention
- **Curriculum**: Syllabus planning, scheme of work
- **Parent Communication**: Meeting strategies, reporting
- **Resources**: Teaching materials and tools

## Deployment

### Render Deployment

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set environment variables in Render dashboard
4. Deploy!

Build Command: `pip install -r requirements.txt`
Start Command: `gunicorn main:app --workers 4 --bind 0.0.0.0:$PORT`

### Database (PostgreSQL)

Create a PostgreSQL database on Render and copy the connection string to `DATABASE_URL`.

## Logging

All user registrations and conversations are logged to Google Sheets in real-time via a background worker. Failed logs are automatically retried.

## Contributing

LINKA AI is an initiative by Schoolinka, founded by Oluwuwaseun Kayode.

## License

Proprietary - Schoolinka Initiative

## Support

For issues or questions, contact Schoolinka support.

---

**LINKA AI** - Empowering Nigerian Teachers, One Conversation at a Time 🎓
