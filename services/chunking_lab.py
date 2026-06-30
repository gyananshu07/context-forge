import uuid
from typing import Any, Dict, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from core.config import settings


class ChunkingLabService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY
        )

    def load_document(self, file_path: str, file_type: str = "application/pdf") -> List[Document]:
        if file_type == "application/pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
        else:
            # Assume text/markdown
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()

    def process_file(
        self, file_path: str, file_type: str, strategy: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        docs = self.load_document(file_path, file_type)

        if strategy == "recursive":
            return self._process_recursive(docs, params)
        elif strategy == "semantic":
            return self._process_semantic(docs, params)
        elif strategy == "markdown":
            return self._process_markdown(docs, params)
        elif strategy == "parent-child":
            return self._process_parent_child(docs, params)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _format_chunks(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        return [
            {
                "id": str(uuid.uuid4()),
                "text": chunk.page_content,
                "metadata": chunk.metadata,
                "length": len(chunk.page_content),
            }
            for chunk in chunks
        ]

    def _process_recursive(self, docs: List[Document], params: Dict[str, Any]):
        chunk_size = int(params.get("chunk_size", 1000))
        chunk_overlap = int(params.get("chunk_overlap", 200))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        return {"chunks": self._format_chunks(chunks)}

    def _process_semantic(self, docs: List[Document], params: Dict[str, Any]):
        threshold = float(params.get("breakpoint_threshold", 95.0))
        # Note: SemanticChunker combines text before splitting, so we'll join the docs.
        text = "\n\n".join(doc.page_content for doc in docs)
        
        splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold,
        )
        chunks = splitter.create_documents([text])
        return {"chunks": self._format_chunks(chunks)}

    def _process_markdown(self, docs: List[Document], params: Dict[str, Any]):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        text = "\n\n".join(doc.page_content for doc in docs)
        chunks = splitter.split_text(text)
        
        # Optionally recursively split large markdown chunks
        if params.get("recursive_fallback"):
            chunk_size = int(params.get("chunk_size", 1000))
            chunk_overlap = int(params.get("chunk_overlap", 200))
            rec_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = rec_splitter.split_documents(chunks)
            
        return {"chunks": self._format_chunks(chunks)}

    def _process_parent_child(self, docs: List[Document], params: Dict[str, Any]):
        parent_chunk_size = int(params.get("parent_chunk_size", 2000))
        child_chunk_size = int(params.get("child_chunk_size", 400))
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=0
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=50
        )
        
        parent_docs = parent_splitter.split_documents(docs)
        
        results = []
        for parent in parent_docs:
            parent_id = str(uuid.uuid4())
            child_docs = child_splitter.split_documents([parent])
            results.append({
                "parent": {
                    "id": parent_id,
                    "text": parent.page_content,
                    "metadata": parent.metadata,
                    "length": len(parent.page_content),
                },
                "children": self._format_chunks(child_docs)
            })
            
        return {"parent_child_chunks": results}

    def evaluate_retrieval(
        self, query: str, file_path: str, file_type: str, strategy: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates an ephemeral in-memory vector store, adds the chunks, and performs a search.
        """
        # Load and chunk the document using the requested strategy
        chunk_data = self.process_file(file_path, file_type, strategy, params)
        
        # We need Langchain Documents to add to a vectorstore
        docs_to_index = []
        
        if strategy == "parent-child":
            # For parent-child lab evaluation, we simulate ParentDocumentRetriever
            # Index children, retrieve them, then map to parent.
            # Real ParentDocumentRetriever uses a Docstore. We will do a simple manual version for the lab.
            from langchain_chroma import Chroma
            
            pc_chunks = chunk_data.get("parent_child_chunks", [])
            parent_store = {}
            for item in pc_chunks:
                parent_id = item["parent"]["id"]
                parent_store[parent_id] = item["parent"]
                for child in item["children"]:
                    # Inject parent_id into child metadata
                    meta = child["metadata"].copy()
                    meta["parent_id"] = parent_id
                    docs_to_index.append(
                        Document(page_content=child["text"], metadata=meta)
                    )
                    
            if not docs_to_index:
                return {"results": []}
                
            vectorstore = Chroma.from_documents(docs_to_index, self.embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            retrieved_child_docs = retriever.invoke(query)
            
            # Map back to parents
            unique_parents = {}
            for doc in retrieved_child_docs:
                pid = doc.metadata.get("parent_id")
                if pid and pid not in unique_parents:
                    unique_parents[pid] = parent_store[pid]
                    
            return {
                "retrieved_parents": list(unique_parents.values()),
                "retrieved_children": self._format_chunks(retrieved_child_docs)
            }
            
        else:
            # Standard single-level chunking
            from langchain_chroma import Chroma
            chunks = chunk_data.get("chunks", [])
            for c in chunks:
                docs_to_index.append(Document(page_content=c["text"], metadata=c["metadata"]))
                
            if not docs_to_index:
                return {"results": []}
                
            vectorstore = Chroma.from_documents(docs_to_index, self.embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            retrieved_docs = retriever.invoke(query)
            
            return {"results": self._format_chunks(retrieved_docs)}
