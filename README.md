
# SupportMind

> **AI-Powered Customer Support Agent with RAG** — Intelligent document Q&A with sentiment analysis, source citations, and conversational memory.

[![Python]       (https://img.shields.io/badge/Python-3.12-blue.svg)]          (https://python.org)
[![LangChain]    (https://img.shields.io/badge/LangChain-0.2.17-green.svg)]    (https://langchain.com)
[![Streamlit]    (https://img.shields.io/badge/Streamlit-1.40-red.svg)]        (https://streamlit.io)
[![Groq]         (https://img.shields.io/badge/Groq-LLM-orange.svg)]           (https://groq.com)
[![ChromaDB]     (https://img.shields.io/badge/ChromaDB-VectorDB-yellow.svg)]  (https://chromadb.io)
[![License]      (https://img.shields.io/badge/License-MIT-purple.svg)]        (LICENSE)

---

## Overview

**SupportMind** is a production-ready RAG (Retrieval-Augmented Generation) application that transforms static policy documents into an intelligent, empathetic customer support agent. Built with modern LLM engineering practices, it demonstrates expertise in vector databases, embedding models, prompt engineering, and full-stack AI application development.

### Why This Matters

Traditional FAQ systems frustrate users with rigid keyword matching. SupportMind understands **context**, detects **sentiment**, and provides **verifiable answers** with source citations — bridging the gap between static documentation and dynamic user needs.

---

## Architecture Highlights

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                            │
│              Dark-themed, responsive chat interface             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline Engine                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │
│  │   Load      │───▶│   Chunk     │───▶│   Embed (HuggingFace)││ 
│  │ PDFs/URLs   │    │ & Hash IDs  │    │  all-MiniLM-L6-v2   │   │
│  └─────────────┘    └─────────────┘    └─────────────────────┘   │
│                                              │                   │
│                                              ▼                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐ │
│  │  Sentiment  │     │   Groq      │◀───│      ChromaDB        │ │
│  │  Analysis   │───▶ │  Llama 3.3 │     │  (Vector Search)     │ │
│  │ DistilBERT  │     │   70B       │     │                      │ │
│  └─────────────┘     └─────────────┘     └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Intelligent Document Ingestion
- **PDF Processing**: `PyPDFLoader` with metadata extraction (page numbers, filenames, total pages)
- **Web Scraping**: Multi-strategy URL loading
  - **Static** (fast): `UnstructuredURLLoader` with custom headers
  - **Dynamic** (JS-rendered): Selenium headless Chrome with bot-detection bypass
  - **Auto-fallback**: Automatically switches strategies when blocked
- **Smart Chunking**: `RecursiveCharacterTextSplitter` with overlap (1000 chars, 200 overlap)
- **Deduplication**: MD5-based `chunk_id` generation for idempotent operations

### 2. Semantic Search & Retrieval
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (384-dim, CPU-optimized)
- **Similarity Search**: Top-k retrieval (k=5) with source document tracking
- **Collections**: Multi-tenant support for different document sets

### 3. Sentiment-Aware Response Generation
- **Classifier**: DistilBERT fine-tuned on SST-2 (cached with `@lru_cache`)
- **Dynamic Tone Injection**:
  | Sentiment | Tone | Example Trigger |
  |-----------|------|-----------------|
  | 😤 Frustrated | Empathetic, apologetic | "I'm sick of waiting!" |
  | 😐 Neutral | Professional, clear | "What's the policy?" |
  | 😊 Happy | Warm, friendly | "Thanks for the help!" |
- **Prompt Engineering**: Runtime tone injection via LangChain's `ChatPromptTemplate`

### 4. Conversational Memory
- **History-Aware Retriever**: Condenses follow-up questions using `create_history_aware_retriever`
- **Context Preservation**: Maintains multi-turn conversation context
- **Memory Format**: LangChain message objects (`HumanMessage`, `AIMessage`)

### 5. Source Citations & Attribution
- **Citation Extraction**: Deduplicated by `(source, page)` pairs
- **Inline Attribution**: Every response includes `[Source: filename, Page X]`
- **Transparency**: Users can verify answers against original documents

### 6. Production-Ready Frontend
- **Dark Theme**: GitHub-inspired color scheme (`#0D1117` background)
- **Real-time Feedback**: Progress bars during ingestion, sentiment badges, typing indicators
- **Session Management**: Collection switching, chat history clearing, chain rebuilding
- **Responsive Layout**: Sidebar navigation, tabbed upload interface (PDF/URL)

---

## Tech Stack

| Category          | Technology                         | Purpose                                       |
|-------------------|------------------------------------|---------------------------------------------- |
| **LLM**           | Groq API (Llama 3.3 70B)           | Fast, cost-effective inference                |
| **Orchestration** | LangChain LCEL                     | Composable RAG pipelines                      |
| **Embeddings**    | HuggingFace + SentenceTransformers | `all-MiniLM-L6-v2` semantic search            |
| **Vector DB**     | ChromaDB                           | Persistent vector storage & similarity search |
| **Frontend**      | Streamlit                          | Rapid UI development                          |
| **Scraping**      | Selenium + Unstructured            | Dynamic content extraction                    |
| **Sentiment**     | Transformers (DistilBERT)          | Real-time emotion detection                   |
| **PDF**           | PyPDF + LangChain Loaders          | Document ingestion                            |

---

## Project Structure

```
supportmind/
├── app/                          # Streamlit frontend
│   ├── main.py                   # App entry point
│   ├── pages/
│   │   ├── 01_chat.py            # Chat interface with RAG
│   │   └── 02_upload.py          # Document upload & ingestion
│   └── components/
│       └── sidebar.py            # Reusable sidebar component
│
├── core/                         # Backend business logic
│   ├── ingestion/                # Document ingestion pipeline
│   │   ├── pipeline.py           # Orchestration layer
│   │   ├── pdf_loader.py         # PDF loading with metadata
│   │   ├── url_loader.py         # URL scraping (static + JS)
│   │   └── chunker.py            # Text splitting & ID generation
│   ├── embeddings/
│   │   └── embedder.py           # HuggingFace embeddings (singleton)
│   ├── vectorstore/
│   │   └── chroma_store.py       # ChromaDB CRUD operations
│   ├── retrieval/
│   │   ├── retriever.py          # RAG chain builder
│   │   └── citation_formatter.py # Source attribution logic
│   ├── llm/
│   │   ├── groq_client.py        # Groq API client (singleton)
│   │   └── prompts.py            # Prompt templates & tone instructions
│   └── sentiment/
│       └── detector.py           # Sentiment classification
│
├── data/                         # Data storage
│   ├── raw/                      # Original documents
│   └── chroma_db/                # Vector database persistence
│
├── tests/                        # Pytest test suite
│   ├── test_ingestion.py         # PDF loading & chunking tests
│   └── test_retrieval.py         # RAG chain & memory tests
│
├── scripts/                      # Utility scripts
│   └── ingest_sample_data.py     # Sample data ingestion
│
└── evaluation/                   # RAGAS evaluation (extensible)
    └── reports/
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/imsaurabh17/supportmind.git
cd supportmind

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Run the Application

```bash
# Start the Streamlit server
streamlit run app/main.py
```

The app will be available at `http://localhost:8501`

---

## Usage Flow

1. **Upload Documents**
   - Navigate to "Upload Documents" tab
   - Upload PDFs or paste URLs
   - Documents are chunked, embedded, and stored in ChromaDB

2. **Start Chatting**
   - Switch to "Chat" tab
   - Ask questions in natural language
   - View sentiment-aware responses with source citations

3. **Manage Collections**
   - Create separate collections for different document sets
   - Switch between collections seamlessly
   - Clear chat history to start fresh

---

## Technical Highlights

### Engineering Decisions

| Decision                                 | Rationale                                                                     |
|------------------------------------------|-------------------------------------------------------------------------------|
| **Singleton Pattern**                    | LLM and embedding models use `@lru_cache` to prevent reloading (80MB+ models) |
| **LCEL (LangChain Expression Language)** | Composable, type-safe chains vs. legacy `ConversationalRetrievalChain`        |
| **Hybrid URL Loading**                   | Static first for speed, Selenium fallback for SPAs (Single Page Applications) |
| **MD5 Chunk IDs**                        | Deterministic IDs enable deduplication and traceability                       |
| **Dynamic Tone Injection**               | Runtime prompt modification vs. multiple prompt templates (DRY principle)     |

### Performance Optimizations

- **Embedding Cache**: Model loaded once per process
- **Lazy Loading**: Sentiment classifier only loads on first use
- **Chroma Persistence**: Vector DB survives restarts
- **Progressive Enhancement**: Static URL loading attempts before heavy Selenium

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_retrieval.py -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

**Test Coverage:**
- PDF loading and validation
- Document chunking and metadata enrichment
- Semantic search retrieval
- RAG chain response generation
- Citation extraction and formatting
- Conversational memory/follow-up questions

---

## Skills Demonstrated

### Core Competencies

- **RAG Architecture Design**: End-to-end pipeline from ingestion to retrieval
- **LLM Integration**: API-based inference with Groq, prompt engineering
- **Vector Database Management**: ChromaDB operations, similarity search, collections
- **Embedding Models**: HuggingFace integration, model caching strategies
- **Document Processing**: PDF parsing, web scraping, text chunking strategies

### Advanced Techniques

- **Sentiment Analysis**: Transformer-based classification for dynamic UX
- **Conversational AI**: Memory management, context preservation
- **Source Attribution**: Citation extraction, provenance tracking
- **Web Scraping**: Anti-bot detection, headless browser automation

### Software Engineering

- **Modular Architecture**: Clear separation of concerns (ingestion/retrieval/LLM/UI)
- **Type Safety**: Type hints throughout, Pydantic-like validation
- **Error Handling**: Graceful degradation, logging, user feedback
- **Testing**: Pytest fixtures, integration tests, edge case coverage
- **Documentation**: Comprehensive docstrings, README, inline comments

### DevOps & Tooling

- **DevContainer**: Pre-configured for GitHub Codespaces
- **Environment Management**: Virtual environments, `.env` configuration
- **Dependency Management**: Pinned requirements for reproducibility
- **Version Control**: Git best practices, `.gitignore` optimization

---

## Future Enhancements

- [ ] **RAGAS Evaluation**: Automated metrics for retrieval quality
- [ ] **Multi-modal Support**: Image and table extraction from PDFs
- [ ] **User Authentication**: Role-based access to collections
- [ ] **Analytics Dashboard**: Query logs, sentiment trends, popular topics
- [ ] **API Endpoint**: FastAPI backend for headless usage
- [ ] **Streaming Responses**: Token-by-token streaming for better UX
- [ ] **Re-ranking**: Cohere or cross-encoder re-ranking for better retrieval

---

## Screenshots

*Add screenshots here showing:*
- *Chat interface with sentiment badge and citations*
- *Document upload with progress bar*
- *Collection management sidebar*

---

## License

[MIT License](LICENSE) © Saurabh Maurya

---

## Connect With Me

- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]
- **Email**: [your.email@example.com]

---

## Acknowledgments

- Built with [LangChain](https://langchain.com) and [Streamlit](https://streamlit.io)
- LLM inference powered by [Groq](https://groq.com)
- Embeddings via [HuggingFace](https://huggingface.co)

---

> **Hiring?** I'm passionate about building intelligent systems that bridge AI capabilities with real-world user needs. Let's discuss how my skills in LLM engineering, RAG architectures, and full-stack development can contribute to your team.
