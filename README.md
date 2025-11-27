# Search UI Backend

FastAPI backend for the Search UI application.

## Features

- RESTful API endpoints for company and founder data
- Synthetic data generation from JSON file
- AI chat functionality using LangChain Groq
- CORS enabled for frontend integration
- Search functionality
- Modular code structure

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Create a `.env` file in the backend directory
   - Add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   - Get your API key from [Groq Console](https://console.groq.com/)

3. Run the server:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## Project Structure

```
backend/
├── main.py              # FastAPI application and routes
├── data_loader.py       # Module for loading data from JSON
├── chat_service.py      # Module for AI chat functionality
├── data.json            # JSON file with synthetic data
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## API Endpoints

- `GET /` - Health check
- `GET /api/companies` - Get list of companies (20 items)
- `GET /api/companies/search?query=<search_term>` - Search companies
- `POST /api/chat` - Chat with AI assistant
  - Request body:
    ```json
    {
      "message": "Your message here",
      "conversation_history": [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"}
      ]
    }
    ```
  - Response:
    ```json
    {
      "response": "AI response text"
    }
    ```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Data Files

- `data.json` - Contains companies, names, and designations used for synthetic data generation
- Data is loaded via `data_loader.py` module with caching for performance

