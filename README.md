# Context Forge (Backend)

A high-performance FastAPI backend that powers a Retrieval-Augmented Generation (RAG) system for document intelligence. Context Forge allows users to upload PDF documents, processes them into vector embeddings, and exposes a streaming conversational interface to query the document's contents.

## 🚀 Features

- **Document Processing**: Upload PDFs which are automatically chunked and embedded.
- **Advanced Retrieval**:
  - **Hybrid Search**: Combines semantic search (ChromaDB) with keyword sparse search (BM25) using reciprocal rank fusion.
  - **Document Reranking**: Uses Flashrank (cross-encoder) to re-score and compress initial retrieved chunks, surfacing the most relevant context.
  - **Query Contextualization**: Uses an LLM to rewrite ambiguous follow-up questions into standalone queries before retrieval.
  - **Isolated Search**: Restricts vector search strictly to the active document being queried.
- **Streaming Responses**: Streams LLM responses back to the client using Server-Sent Events (SSE).
- **Conversational Memory**: Automatically manages chat history with a sliding window approach, persisting conversation history in PostgreSQL.
- **Robust Architecture**: Built with FastAPI, SQLAlchemy (async), and LangChain.

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL (via Async SQLAlchemy & Alembic)
- **Vector Store**: ChromaDB
- **LLM Orchestration**: LangChain
- **Models**: OpenAI `gpt-4o-mini` (chat) & `text-embedding-3-small` (embeddings)

## 📦 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gyananshu07/context-forge.git
   cd context-forge
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/context_forge
   OPENAI_API_KEY=your_openai_api_key
   ```

5. **Run Migrations:**
   ```bash
   alembic upgrade head
   ```

6. **Start the Server:**
   ```bash
   uvicorn main:app --reload
   ```

## 📚 API Endpoints

- `GET /documents`: List all uploaded documents.
- `POST /documents`: Upload a new PDF document.
- `GET /chat/{document_id}`: Fetch the chat history for a specific document.
- `POST /chat`: Send a query to a document (returns an SSE stream).
