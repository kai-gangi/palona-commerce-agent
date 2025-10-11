# AI Commerce Agent 🛍️

An intelligent shopping assistant that handles conversation, text-based product search, and image-based product search through a single unified agent.

**API Docs:** Available at `/docs` when running

## Features

- Natural conversations with the AI assistant
- Text-based product recommendations via semantic search
- Image-based product search using visual similarity

## Architecture

```
User → FastAPI → LLM Agent (GPT-4o-mini) → Tools
                           ↓
        ┌─────────────────────────────────────┐
        ↓                     ↓               ↓
      Chat              Text Search    Image Search
                            ↓               ↓
                       Embeddings        CLIP
                            ↓               ↓
                      ChromaDB Vector Store
```

**Flow:**
1. **User** sends request via Streamlit UI
2. **FastAPI** receives and routes the request  
3. **LLM Agent** (GPT-4o-mini) decides which tool to use
4. **Tools** execute specific functions:
   - **Chat**: Direct conversation 
   - **Text Search**: Semantic search using embeddings
   - **Image Search**: Visual similarity using CLIP
5. **Vector Store** (ChromaDB) returns relevant products

## Tech Stack Decisions

### Backend: FastAPI
- Async support for concurrent LLM calls
- Automatic API docs
- Production-ready
- **Other Options**: Flask (not suitable because of synchronous operations), Django (too heavy for small API layer)

### LLM: OpenAI GPT-4o-mini
- Fast (~500ms) for real-time chat
- Cost-effective ($0.15/1M tokens, about 6x cheaper than GPT-4o)
- Excellent function calling

### Embeddings: OpenAI text-embedding-3-small
- Strong semantic understanding
- High speed and cost-efficient ($0.02 / 1M tokens)
- Performs well for tasks like semantic similarity, text-to-image retrieval, and query expansion (without high cost of larger embedding models)

### Image Search: CLIP (ViT-B/32)
- Understands semantic similarity (not just visual)
- Zero-shot
- **Why this size:** Good accuracy/speed balance (512-dim, ~300ms CPU)

### Vector DB: ChromaDB
- Simple setup with no external dependencies
- Fast HNSW indexing
- **Other Options:** Better to migrate to Pinecone/Weaviate for 100K+ products

### Frontend: Streamlit
- Fast to deploy and intuitive 
- Eliminates lots of boilerplate code (built-in comopnents for chat messages, file uploads, etc.)
- **Other Options**: React or Next.js for more production-based UI, but for this use case Streamlit can achieve what we need with far less code

## Quick Start
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key-here"

# Run (2 terminals)
python -m backend.main             # Terminal 1: API on :8000
streamlit run streamlit_app/app.py  # Terminal 2: UI on :8501
```

## Next Steps
### Immediate Improvements
With more time, I would focus on three key areas: robustness, user experience, and evaluation.

**Robustness**: 
- Add comprehensive error handling and retry logic for API calls
- Add option to choose different LLMs
- Implement request validation to prevent malformed queries
- Introduce structured logging with request IDs for easier debugging

**User Experience:**
- Enhance the UI with product filtering by price and category
- Add a shopping cart interface
- Maintain conversation context to remember previous searches

**Evaluation:**
- Build a framework to measure search relevance
- Compare different embedding models and chunking strategies quantitatively

## Improvements for Production-Grade Deployment 
For a production-grade deployment, I would:
- Implement proper authentication and rate limits
- Migrate to a managed vector database (e.g., Pinecone) for scalability
- Add Redis caching for frequently searched queries
- Set up monitoring with Prometheus and Grafana to track latency, error rates, and tool usage
- Develop a comprehensive test suite covering:
  - Empty or malformed search queries
  - Corrupted image uploads
  - Concurrent user requests
- Add hybrid search to combine text and image inputs for more accurate product matching
- Conversational memory: persist user preferences across sessions for personalized recommendations
- Multi-language support 
- A/B testing framework (to compare prompts, models, and search strategies)