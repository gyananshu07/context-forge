import json

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from db.models import Document
from db.models.chat_message import ChatMessage

_global_vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY
    ),
    persist_directory="./chroma_db",
)


def format_docs(docs):
    return "\n\n".join(
        f"Source [{i + 1}]:\n{doc.page_content}" for i, doc in enumerate(docs)
    )


class EmbeddingService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
        )

        self.vector_store = _global_vector_store

    def upload_pdf(self, file_path: str):
        # Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)

        # Store embeddings
        self.vector_store.add_documents(chunks)

        return len(chunks)

    async def _fetch_and_format_history(
        self, session: AsyncSession, document_id: int
    ) -> list:
        """Fetches the conversation history for a given document and formats it for LangChain."""
        result = await session.execute(
            select(ChatMessage)
            .where(ChatMessage.document_id == document_id)
            .order_by(ChatMessage.created_at)
        )
        db_messages = result.scalars().all()
        await session.commit()  # Free the DB transaction during long LLM streams

        chat_history = []
        for msg in db_messages[:-1]:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_history.append(AIMessage(content=msg.content))

        # Return only the last 10 messages (sliding window)
        return chat_history[-10:]

    def _create_chat_chain(self):
        """Builds the LangChain LCEL pipeline for the conversational retrieval agent."""
        llm_without_tools = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            streaming=True,
        )

        tool = {"type": "web_search"}
        llm = llm_without_tools.bind_tools([tool])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the following question based only on the provided context. You must cite your sources inline using [1], [2], etc. corresponding to the Source number provided in the context.\n\n<context>\n{context}\n</context>",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        return prompt | llm

    def _create_query_rewrite_chain(self):
        """Builds the LCEL chain for query contextualization (rewriting)."""
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
        )
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        return prompt | llm | StrOutputParser()

    def _extract_text_from_chunk(self, chunk_content) -> str:
        """Safely extracts string text from LangChain message chunks, handling tool calls."""
        if isinstance(chunk_content, str):
            return chunk_content
        if isinstance(chunk_content, list):
            return "".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in chunk_content
            )
        return str(chunk_content)

    async def _save_assistant_message(
        self,
        session: AsyncSession,
        document_id: int,
        content: str,
        citations: list = None,
    ):
        """Persists the final assistant response to the database."""
        assistant_msg = ChatMessage(
            document_id=document_id,
            role="assistant",
            content=content,
            citations=citations,
        )
        session.add(assistant_msg)
        await session.commit()

    async def chat_stream(self, query: str, document_id: int, session: AsyncSession):
        """Main entrypoint for streaming chat responses to the frontend."""
        # Fetch the document to get its file_key for filtering
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        file_key = (
            doc.document_metadata.get("file_key")
            if doc and doc.document_metadata
            else None
        )

        chat_history = await self._fetch_and_format_history(session, document_id)
        chain = self._create_chat_chain()

        if chat_history:
            rewrite_chain = self._create_query_rewrite_chain()
            standalone_query = await rewrite_chain.ainvoke(
                {"input": query, "chat_history": chat_history}
            )
        else:
            standalone_query = query

        search_kwargs = {"k": 10}
        if file_key:
            search_kwargs["filter"] = {"source": file_key}

        chroma_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)

        # Build document-specific BM25 index dynamically
        if file_key:
            docs_data = self.vector_store.get(where={"source": file_key})
        else:
            docs_data = {"documents": [], "metadatas": []}

        documents = docs_data.get("documents", [])
        metadatas = docs_data.get("metadatas", [])

        if documents:
            doc_objects = [
                LangchainDocument(page_content=d, metadata=m or {})
                for d, m in zip(documents, metadatas)
            ]
            bm25_retriever = BM25Retriever.from_documents(doc_objects)
            bm25_retriever.k = 10
            base_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
            )
        else:
            base_retriever = chroma_retriever

        compressor = FlashrankRerank(top_n=4)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

        docs = await retriever.ainvoke(standalone_query)
        context = format_docs(docs)

        full_content = ""
        async for chunk in chain.astream(
            {"input": query, "chat_history": chat_history, "context": context}
        ):
            if chunk.content:
                text_chunk = self._extract_text_from_chunk(chunk.content)
                if text_chunk:
                    full_content += text_chunk
                    payload = json.dumps({"content": text_chunk})
                    yield f"data: {payload}\n\n"

        citations = []
        for i, doc in enumerate(docs):
            source = str(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", 1)
            text = doc.page_content

            citations.append(
                {
                    "id": f"cit-{i}",
                    "source": source.split("\\")[-1].split("/")[-1],
                    "text": text,
                    "page": page,
                }
            )

        if citations:
            yield f"data: {json.dumps({'citations': citations})}\n\n"

        await self._save_assistant_message(
            session, document_id, full_content, citations
        )

        yield "data: [DONE]\n\n"
