# RAG System Project

A robust Retrieval-Augmented Generation (RAG) system built with FastAPI that enables document upload, intelligent processing, vector-based similarity search, and AI-powered answer generation. Upload files, automatically process them into searchable chunks with embeddings, store in PostgreSQL with pgvector and Qdrant vector database, and retrieve contextual answers powered by LLMs for your AI applications.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18.0-336791.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.8.1-orange.svg)](https://github.com/pgvector/pgvector)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## 🏗️ Architecture Overview

### Core Components

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client App    │────▶│   FastAPI API   │────▶│   Controllers   │
│  (Upload/Query) │     │   Routes        │     │  (Business      │
└─────────────────┘     └─────────────────┘     │   Logic)        │
                                                └─────────┬───────┘
                                                          │
                        ┌─────────────────────────────────┼───────────────┐
                        ▼                                 ▼               ▼
                ┌───────────────┐              ┌──────────────┐  ┌──────────────┐
                │  PostgreSQL   │              │ LLM Providers│  │ VectorDB     │
                │  + pgvector   │              │ (OpenAI,     │  │ (PGVector/   │
                │  (Chunks,     │              │  Cohere)     │  │  Qdrant)     │
                │   Projects,   │              │              │  │              │
                │   Assets)     │              │              │  │              │
                └───────────────┘              └──────────────┘  └──────────────┘
                        ▲                              ▲                  ▲
                        │                              │                  │
                        └──────────┬───────────────────┴──────────────────┘
                                   ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   LangChain     │────▶│   Document      │
                        │  Text Splitter  │     │   Loaders       │
                        │  (Chunking)     │     │  (PDF, TXT)     │
                        └─────────────────┘     └─────────────────┘
                                   ▲
                                   │
                        ┌─────────────────┐
                        │   File Storage  │
                        │  (Project-based │
                        │   Organization) │
                        └─────────────────┘
```

### Data Flow

1. **Document Upload** → File validation → Unique naming → Project storage
2. **Document Processing** → Content extraction → Text chunking → Metadata preservation
3. **Data Storage** → PostgreSQL (via SQLAlchemy async) → Project organization → Asset tracking
4. **Vector Embeddings** → LLM Provider (Cohere/OpenAI) → Generate embeddings → Store in VectorDB (PGVector or Qdrant)
5. **Similarity Search** → Query vectors → VectorDB search → Retrieve top-k relevant chunks
6. **Answer Generation** → Prompt construction with context → LLM generation → AI-powered answers

### Provider Architecture

The system uses a **Factory Pattern** for extensible provider management:

**LLM Providers:**

- Abstract `LLMInterface` defines the contract
- `LLMProviderFactory` creates provider instances
- Support for OpenAI and Cohere (easily extensible)
- Unified API for text generation and embeddings
- Multi-language prompt templates with dynamic imports

**VectorDB Providers:**

- Abstract `VectorDBInterface` defines the contract
- `VectorDBProviderFactory` creates provider instances
- **PGVector** implementation for PostgreSQL with pgvector extension
- **Qdrant** implementation for standalone vector storage
- Support for collection management and similarity search
- Configurable distance metrics (cosine, dot product, L2)

## 🛠️ Technical Stack

- **Backend Framework**: FastAPI with async/await patterns and lifespan context management
- **Database**: PostgreSQL 18.0 with pgvector extension (v0.8.1) for vector similarity
- **ORM**: SQLAlchemy 2.0 with async support (asyncpg driver)
- **Database Migrations**: Alembic for schema version control
- **Vector Database**: PGVector (PostgreSQL) or Qdrant for vector storage and similarity search
- **LLM Providers**: OpenAI and Cohere with factory pattern (supports custom OpenAI-compatible APIs)
- **Template Engine**: Multi-language prompt template system with Python string.Template
- **Document Processing**: LangChain (text splitting, document loading)
- **PDF Processing**: PyMuPDF (FitzPDF) for efficient PDF extraction
- **Data Validation**: Pydantic v2 with custom validators
- **File Handling**: aiofiles for async I/O operations
- **Task Queue**: Celery with Redis for background processing
- **Monitoring**: Prometheus metrics with starlette-exporter
- **Containerization**: Docker & Docker Compose
- **Python Version**: 3.12+ / 3.13
- **Additional Libraries**: asyncpg, sqlalchemy, alembic, aiofiles, python-dotenv, python-multipart, qdrant-client, openai, cohere, langchain, motor, redis, flower

## 📁 Project Structure

```text
src/
├── main.py                          # FastAPI application & lifespan context
├── requirements.txt                 # Python dependencies
├── helper/
│   ├── __init__.py
│   └── config.py                    # Application settings management
├── routes/
│   ├── __init__.py
│   ├── base.py                      # Health/version endpoints
│   ├── data_route.py                # File upload & processing endpoints
│   ├── nlp.py                       # RAG endpoints (push, search, generate)
│   └── schemas/
│       ├── __init__.py
│       ├── dataproces_schemas.py    # Request/response schemas
│       └── nlp.py                   # NLP/RAG schemas
├── controllers/
│   ├── __init__.py
│   ├── BaseContoller.py             # Base controller functionality
│   ├── DataController.py            # File validation & storage
│   ├── ProcessController.py         # Document processing & chunking
│   └── NLPController.py             # RAG logic (search, answer generation)
├── models/
│   ├── __init__.py
│   ├── BaseDataModel.py             # Base async SQLAlchemy model
│   ├── ChunkModel.py                # DataChunk DAL (async)
│   ├── ProjectModel.py              # Project DAL (async)
│   ├── AssetModel.py                # Asset DAL (async)
│   ├── db_schemas/
│   │   ├── __init__.py              # Public schema exports
│   │   ├── rag/                     # RAG database schemas
│   │   │   ├── alembic.ini          # Alembic configuration
│   │   │   ├── alembic/             # Migration scripts
│   │   │   │   ├── env.py           # Migration environment
│   │   │   │   └── versions/        # Migration versions
│   │   │   └── schemas/
│   │   │       ├── __init__.py
│   │   │       ├── rag_base.py      # SQLAlchemy Base
│   │   │       ├── chunks_schemas.py # DataChunk model
│   │   │       ├── project_shemas.py # Project model
│   │   │       └── asset.py         # Asset model
│   ├── enums/
│   │   ├── __init__.py
│   │   ├── ProcesseEnums.py         # Document type enums
│   │   ├── ResponseEnums.py         # API response enums
│   │   └── __pycache__/
│   └── __pycache__/
├── stores/                          # External service providers
│   ├── llm/                         # LLM providers (OpenAI, Cohere)
│   │   ├── __init__.py
│   │   ├── LLMInterface.py          # Abstract LLM interface
│   │   ├── LLMEnums.py              # LLM provider enums
│   │   ├── LLMProviderFactory.py    # Factory for LLM providers
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── OpenAIProvider.py    # OpenAI implementation
│   │   │   └── CoHereProvider.py    # Cohere implementation
│   │   └── templete/
│   │       ├── __init__.py
│   │       ├── templete_parser.py   # Template parser for prompts
│   │       └── locales/
│   │           ├── ar/              # Arabic templates
│   │           │   ├── __init__.py
│   │           │   └── rag.py
│   │           └── en/              # English templates
│   │               ├── __init__.py
│   │               └── rag.py
│   └── vectordb/                    # Vector database providers
│       ├── __init__.py
│       ├── VectorDBInterface.py     # Abstract VectorDB interface
│       ├── VectorDBEnums.py         # VectorDB provider enums
│       ├── VectorDBProviderFactory.py # Factory for VectorDB providers
│       └── providers/
│           ├── __init__.py
│           ├── QdrantDBProvider.py  # Qdrant implementation
│           └── PGVectorProvider.py  # PostgreSQL pgvector implementation
└── assets/
    └── files/                       # File storage (organized by project)
        └── {project_id}/            # Project-specific directories

docker/
├── docker-compose.yml               # PostgreSQL + pgvector service
├── .env                             # Database credentials (not committed)
├── DATABASE_CONNECTIONS.md          # Connection guide for DBeaver/pgAdmin
└── QUICK_REFERENCE.txt              # Quick reference card

.gitignore                          # Root gitignore
README.md                           # This file
LICENSE                            # Project license
```

## 🚀 API Endpoints

### Base Endpoints

- `GET /api/v1/` - Application information and health check health

### Data Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/data/upload/{project_id}` | Upload files to a project (returns asset_id) |
| `POST` | `/api/v1/data/processall/{project_id}` | Process all files in project, save chunks to PostgreSQL |
| `POST` | `/api/v1/data/processone/{project_id}` | Process single file, save chunks with optional reset |

### NLP/RAG Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/nlp/push` | Push documents to vector database with embeddings |
| `POST` | `/api/v1/nlp/search` | Search for similar documents using vector similarity |
| `POST` | `/api/v1/nlp/generate` | Generate AI-powered answers based on query and context |

### Response Structure

All endpoints return JSON responses with status indicators:

```json
{
  "status": "success_code",
  "message": "Descriptive message",
  "data": {}
}
```

### Request/Response Examples

**Upload File:**

```bash
curl -X POST "http://localhost:8000/api/v1/data/upload/my_project" \
     -F "file=@document.pdf"

# Response:
{
  "status": "file_upload_success",
  "file_path": "/path/to/file",
  "file_id": "unique_filename",
  "asset_id": "507f1f77bcf86cd799439011"
}
```

**Process Single File:**

```bash
curl -X POST "http://localhost:8000/api/v1/data/processone/my_project" \
     -H "Content-Type: application/json" \
     -d '{
       "file_id": "abc123_document.pdf",
       "chunk_size": 1000,
       "overlap_size": 100,
       "do_reset": false
     }'

# Response:
{
  "status": "processing_success",
  "total_chunks": 42,
  "inserted_chunks": 42,
  "chunks": [...]
}
```

**Process All Files:**

```bash
curl -X POST "http://localhost:8000/api/v1/data/processall/my_project"

# Response:
{
  "status": "processing_success",
  "total_files": 5,
  "processed_files": 5,
  "failed_files": 0,
  "total_chunks": 150,
  "inserted_chunks": 150
}
```

**Push Documents to Vector Database:**

```bash
curl -X POST "http://localhost:8000/api/v1/nlp/push" \
     -H "Content-Type: application/json" \
     -d '{
       "project_id": "my_project",
       "do_reset": false
     }'

# Response:
{
  "status": "success",
  "message": "Successfully pushed 150 chunks to vector database",
  "total_chunks": 150,
  "embedding_model": "embed-v4.0",
  "vector_dimension": 256
}
```

**Search Similar Documents:**

```bash
curl -X POST "http://localhost:8000/api/v1/nlp/search" \
     -H "Content-Type: application/json" \
     -d '{
       "project_id": "my_project",
       "query": "What is the main topic?",
       "top_k": 5
     }'

# Response:
{
  "status": "success",
  "results": [
    {
      "chunk_text": "The main topic discusses...",
      "score": 0.89,
      "chunk_id": "507f1f77bcf86cd799439011"
    },
    ...
  ],
  "total_results": 5
}
```

**Generate AI-Powered Answer:**

```bash
curl -X POST "http://localhost:8000/api/v1/nlp/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "project_id": "my_project",
       "query": "What is the main topic?",
       "language": "en",
       "top_k": 5
     }'

# Response:
{
  "status": "success",
  "answer": "Based on the documents, the main topic discusses...",
  "context_documents_count": 5
}
```

## 🔧 Configuration

### RAG Workflow

The RAG system follows a complete pipeline from document upload to AI-powered answer generation:

1. **Upload Documents**: Upload PDF or text files to project-specific directories
2. **Process & Chunk**: Extract text and split into semantic chunks with overlap
3. **Generate Embeddings**: Create vector embeddings using Cohere or OpenAI
4. **Store Vectors**: Index embeddings in vector database (PGVector or Qdrant) for similarity search
5. **Query Processing**: Convert user queries into embeddings
6. **Retrieve Context**: Find top-k most relevant document chunks via vector similarity
7. **Prompt Construction**: Build context-aware prompts with multi-language templates
8. **Generate Answers**: Use LLM to generate answers based on retrieved context

### Key Features

- **Multi-Provider Support**: Switch between OpenAI and Cohere for embeddings and generation
- **Custom LLM Endpoints**: Use OpenAI-compatible APIs (e.g., local Ollama models via ngrok)
- **Vector Search**: Similarity search with configurable distance metrics (cosine, dot product)
- **Template System**: Multi-language prompt templates with dynamic variable substitution
- **Async Processing**: Non-blocking I/O for efficient file processing and database operations
- **Lazy Loading**: Optimized startup with on-demand provider initialization
- **Flexible Chunking**: Configurable chunk sizes and overlap for optimal retrieval
- **Project Isolation**: Separate vector collections per project for organization

### Environment Variables

Create a `.env` file in the `src/` directory with the following variables (see `src/.env.example` for a template):


## 📋 Prerequisites & Installation

### Prerequisites

- Python 3.12+ (or 3.13)
- Docker & Docker Compose
- Git
- PostgreSQL client (optional, for direct database access)

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SaeedNeamtallah/AI-Customer-Copilot.git
   cd AI-Customer-Copilot
   ```

2. **Create and activate virtual environment:**

   ```bash
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   cd src
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   ```bash
   # Create .env file in src/ directory
   cp .env.example .env
   # Edit .env with your API keys and database credentials
   ```

5. **Start PostgreSQL with Docker Compose:**

   ```bash
   cd ../docker
   # Create .env file for Docker (see docker/.env.example)
   docker-compose up -d
   ```

6. **Run database migrations:**

   ```bash
   cd ../src/models/db_schemas/rag
   alembic upgrade head
   ```

7. **Run the application:**

   ```bash
   cd ../../../
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

8. **Access the API:**
   - API Documentation: `http://localhost:8000/docs`
   - ReDoc Documentation: `http://localhost:8000/redoc`
   - API Base URL: `http://localhost:8000/api/v1`

## 📊 Database Schema

### PostgreSQL Tables (with pgvector extension)

#### `projects` Table

```sql
CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE
);
```

#### `assets` Table

```sql
CREATE TABLE assets (
    asset_id SERIAL PRIMARY KEY,
    asset_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    asset_project_id INTEGER NOT NULL REFERENCES projects(project_id),
    asset_type VARCHAR(50) NOT NULL,
    asset_name VARCHAR(255) NOT NULL,
    asset_size INTEGER,
    asset_config JSONB,
    asset_pushed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    UNIQUE (asset_project_id, asset_name)
);
```

#### `chunks` Table

```sql
CREATE TABLE chunks (
    chunk_id SERIAL PRIMARY KEY,
    chunk_uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    chunk_text TEXT NOT NULL,
    chunk_metadata JSONB,
    chunk_order INTEGER NOT NULL,
    chunk_project_id INTEGER NOT NULL REFERENCES projects(project_id),
    chunk_asset_id INTEGER NOT NULL REFERENCES assets(asset_id),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX ix_chunk_project_id ON chunks(chunk_project_id);
CREATE INDEX ix_chunk_asset_id ON chunks(chunk_asset_id);
```

### Schema Features

- **UUID Support**: All tables have UUID fields for external references
- **JSONB**: Flexible metadata storage for chunks and asset configurations
- **Foreign Keys**: Proper relationships between projects, assets, and chunks
- **Timestamps**: Automatic tracking of creation and update times
- **Indexes**: Optimized for common query patterns

## 🧪 Testing

### Manual Testing - Complete RAG Workflow

```bash
# 1. Upload a PDF file
curl -X POST "http://localhost:8000/api/v1/data/upload/test_project" \
     -F "file=@sample.pdf"

# 2. Process the uploaded file into chunks
curl -X POST "http://localhost:8000/api/v1/data/processone/test_project" \
     -H "Content-Type: application/json" \
     -d '{
       "file_id": "abc123_sample.pdf",
       "chunk_size": 1000,
       "overlap_size": 100,
       "do_reset": false
     }'

# 3. Push chunks to vector database with embeddings
curl -X POST "http://localhost:8000/api/v1/nlp/push" \
     -H "Content-Type: application/json" \
     -d '{
       "project_id": "test_project",
       "do_reset": false
     }'

# 4. Search for similar documents
curl -X POST "http://localhost:8000/api/v1/nlp/search" \
     -H "Content-Type: application/json" \
     -d '{
       "project_id": "test_project",
       "query": "What is the main topic?",
       "top_k": 5
     }'

# 5. Generate AI-powered answer
curl -X POST "http://localhost:8000/api/v1/nlp/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "project_id": "test_project",
       "query": "What is the main topic?",
       "language": "en",
       "top_k": 5
     }'

# 6. Verify data in PostgreSQL
# Option 1: Using psql
docker exec -it vector-postgres psql -U postgres -d ai_vectors
# SELECT * FROM projects WHERE project_id = 3;
# SELECT COUNT(*) FROM chunks WHERE chunk_project_id = 3;

# Option 2: Using DBeaver (see docker/DATABASE_CONNECTIONS.md)
```

## 📝 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## 👥 Contributors

- Saeed Neamtallah (@SaeedNeamtallah)
