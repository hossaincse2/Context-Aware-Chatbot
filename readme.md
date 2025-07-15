# Context-Aware Chatbot

A modern, intelligent chatbot application built with Flask that provides context-aware conversations using vector search and semantic understanding. The application features a clean web interface, document knowledge base management, and real-time chat capabilities.

## 🚀 Features

### Core Functionality
- **Context-Aware Conversations** - Maintains conversation history and context
- **Document Knowledge Base** - Upload and search through document collections
- **Hybrid Search** - Combines vector similarity and keyword search
- **Real-time Chat Interface** - Smooth, responsive messaging experience
- **Source Attribution** - Shows which documents were used in responses
- **Statistics Dashboard** - Monitor system performance and usage

### Technical Features
- **Vector Database (Qdrant)** - Semantic search capabilities
- **Redis Caching** - Fast session and context management
- **LangChain Integration** - Advanced NLP processing
- **GitHub Models API** - Flexible LLM integration
- **RESTful API** - Clean, well-documented endpoints
- **Production Ready** - Docker support, Gunicorn, proper logging

## 🛠️ Technology Stack

- **Backend**: Flask, Python 3.9+
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Database**: Qdrant (Vector), Redis (Cache)
- **AI/ML**: LangChain, GitHub Models API
- **Deployment**: Docker, Docker Compose, Gunicorn
- **Testing**: pytest, Flask-Testing

## 📦 Installation

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Git

### Quick Start

1. **Clone the Repository**
```bash
git clone <repository-url>
cd context-aware-chatbot
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Environment Variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start Services**
```bash
docker-compose up -d
```

5. **Run the Application**
```bash
python app.py
```

6. **Access the Application**
- Web Interface: http://localhost:5000
- API Documentation: http://localhost:5000/api

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# GitHub Models API
GITHUB_TOKEN=your_github_token_here
GITHUB_MODEL=gpt-4o-mini

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Application Settings
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DEBUG=True
```

### Docker Configuration

The application uses Docker Compose for easy deployment:

```yaml
# docker-compose.yml includes:
- Redis (Cache)
- Qdrant (Vector Database)
- Flask Application (Optional)
```

## 🎯 Usage

### Web Interface

1. **Start a Conversation**
   - Navigate to http://localhost:5000
   - Type your message in the chat input
   - Press Enter or click Send

2. **Upload Documents**
   - Click "Upload Documents" button
   - Select text files or documents
   - Documents are automatically processed and indexed

3. **View Statistics**
   - Access the stats panel to see:
     - Total conversations
     - Documents in knowledge base
     - System performance metrics

### API Endpoints

#### Chat API
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Your question here",
  "session_id": "optional-session-id"
}
```

#### Upload Documents
```bash
POST /api/add_documents
Content-Type: multipart/form-data

files: [document1.txt, document2.txt, ...]
```

#### Get Statistics
```bash
GET /api/stats
```

#### Clear History
```bash
POST /api/clear_history
Content-Type: application/json

{
  "session_id": "session-to-clear"
}
```

## 🏗️ Architecture

### Application Structure
```
context-aware-chatbot/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── css/
│   └── js/
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Docker services
├── Dockerfile           # Application container
└── README.md           # This file
```

### Data Flow
1. User sends message via web interface or API
2. Message is processed and contextualized
3. Hybrid search retrieves relevant documents
4. LLM generates response with context
5. Response is returned with source attribution
6. Conversation history is maintained

## 🚀 Deployment

### Development
```bash
# Start services
docker-compose up -d

# Run in development mode
python app.py
```

### Production
```bash
# Build production image
docker build -t chatbot-app .

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
```

### Docker Deployment
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 Monitoring

### Health Checks
- **Application Health**: `/health`
- **Redis Status**: Automatic connection testing
- **Qdrant Status**: Vector database health check

### Logging
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`
- Access logs: Nginx/Gunicorn logs

## 🛡️ Security

### Authentication
- Session-based authentication
- CSRF protection
- Rate limiting on API endpoints

### Data Protection
- Environment variable configuration
- Secure headers
- Input validation and sanitization

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-flask

# Run test suite
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

### Test Structure
```
tests/
├── test_api.py          # API endpoint tests
├── test_chat.py         # Chat functionality tests
├── test_documents.py    # Document processing tests
└── conftest.py         # Test configuration
```

## 🔍 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check Redis status
   docker-compose ps redis
   
   # Restart Redis
   docker-compose restart redis
   ```

2. **Qdrant Connection Error**
   ```bash
   # Check Qdrant status
   docker-compose ps qdrant
   
   # Reset Qdrant data
   docker-compose down
   docker volume rm context-aware-chatbot_qdrant_storage
   docker-compose up -d
   ```

3. **GitHub API Rate Limiting**
   - Check your GitHub token permissions
   - Monitor API usage in GitHub settings
   - Implement request throttling

### Debug Mode
```bash
# Enable debug mode
export FLASK_ENV=development
export DEBUG=True
python app.py
```

## 📈 Performance Optimization

### Vector Search
- Optimize embedding model selection
- Implement result caching
- Use appropriate similarity thresholds

### Caching Strategy
- Redis for session management
- In-memory caching for frequent queries
- Document embedding caching

### Scaling
- Horizontal scaling with load balancers
- Database sharding for large document collections
- Microservices architecture for components

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions
- Keep functions small and focused

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** - For NLP processing capabilities
- **Qdrant** - For vector similarity search
- **Tailwind CSS** - For modern UI styling
- **Flask** - For web framework
- **GitHub Models** - For LLM integration

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Happy Chatting!** 🤖💬