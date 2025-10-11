# AI Commerce Agent API Documentation

## Overview

The AI Commerce Agent API provides intelligent product discovery and recommendation services via LLM chatting and image-based search capabilities.

**Base URL:** `http://localhost:8000`

## Authentication

Currently, no authentication is required for API access.

## Endpoints

### Chat Endpoints

#### POST `/chat`

Interact with the AI commerce agent for product recommendations and general conversation.

**Request Body:**
```json
{
  "message": "string",
  "image": "string (optional)",
  "history": [
    {
      "role": "user|assistant",
      "content": "string"
    }
  ]
}
```

**Response:**
```json
{
  "message": "string",
  "products": [
    {
      "id": "string",
      "name": "string",
      "category": "string",
      "description": "string",
      "price": 0.0,
      "image_path": "string",
      "tags": ["string"]
    }
  ],
  "tool_used": "string"
}
```

**Features:**
- Natural language product search
- Image-based product discovery (base64 encoded images)
- Conversation history support
- Product recommendations

**Example Request:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me running shoes under $100",
    "history": []
  }'
```

#### POST `/chat/stream`

Real-time streaming chat with server-sent events (SSE) for improved user experience.

**Request Body:** Same as `/chat`

**Response:** Server-Sent Events stream
- Content-Type: `text/event-stream`
- Each chunk: `data: {"type": "chunk", "content": "string"}\n\n`
- End marker: `data: [DONE]\n\n`

**Example:**
```python
import requests
import json

def stream_chat(message, history=None):
    url = "http://localhost:8000/chat/stream"
    data = {
        "message": message,
        "history": history or []
    }
    
    response = requests.post(url, json=data, stream=True)
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]  # Remove 'data: ' prefix
                if data_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(data_str)
                    print(chunk.get('content', ''), end='', flush=True)
                except json.JSONDecodeError:
                    continue

# Usage
stream_chat("Find laptops under $800")
```

### Health Endpoints

#### GET `/health`

Check service health and availability.

**Response:**
```json
{
  "status": "healthy|unhealthy",
  "timestamp": 1697234567,
  "response_time_ms": 45,
  "checks": {
    "config": "ok|failed",
    "services": "ok|failed"
  }
}
```

**Status Codes:**
- `200`: Service healthy
- `503`: Service unhealthy

## Data Models

### Product
Core product information returned in search results.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique product identifier |
| `name` | string | Product name |
| `category` | string | Product category |
| `description` | string | Product description |
| `price` | number | Price in USD (> 0) |
| `image_path` | string | Path to product image |
| `tags` | array | Product tags for search |

### ChatMessage
Individual message in conversation history.

| Field | Type | Description |
|-------|------|-------------|
| `role` | string | Either "user" or "assistant" |
| `content` | string | Message text content |

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request - Invalid input parameters
- `500`: Internal Server Error - Processing failed
- `503`: Service Unavailable - Health check failed

## Rate Limiting

Currently, no rate limiting is implemented.

## Usage Examples

### Text-based Product Search
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need wireless headphones for gym workouts"
  }'
```

### Image-based Product Search
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find similar products",
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  }'
```

### Conversation with History
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me something cheaper",
    "history": [
      {"role": "user", "content": "Show me laptops"},
      {"role": "assistant", "content": "Here are some laptops..."}
    ]
  }'
```

## Development

**Local Development:**
```bash
# Start the server
python -m backend.main

# Server runs on http://localhost:8000
# API docs available at http://localhost:8000/docs
```

**Interactive API Documentation:** Available at `/docs` (Swagger UI)