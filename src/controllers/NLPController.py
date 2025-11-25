"""
NLP Controller for RAG operations.

Handles vector database indexing, similarity search, and answer generation
using configurable LLM and embedding providers.
"""
from .BaseContoller import BaseController
from models.db_schemas import DataChunk, Project
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
import logging

logger = logging.getLogger(__name__)

class NLPController(BaseController):
    """Controller for RAG workflow operations including indexing, search, and generation."""
    
    def __init__(self, vector_client, generation_client, embedding_client, templete_parser, settings=None):
        """Initialize NLP controller with required clients and settings."""
        super().__init__()
        self.vector_client = vector_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.templete_parser = templete_parser
        self.settings = settings


    def create_collection_name(self, project_id):
        """Create collection name from project_id (accepts int or str)"""
        collection_name = f"collection_{project_id}".strip()
        return collection_name
    
    async def reset_vector_db_collection(self, project: Project):
        """Delete and recreate vector database collection for a project."""
        collection_name = self.create_collection_name(project.project_id)
        await self.vector_client.delete_collection(collection_name)
        await self.vector_client.create_collection(collection_name, embedding_size=self.embedding_client.embedding_size)

    async def get_vector_db_collection_info(self, project: Project):
        """Get information about vector database collection."""
        collection_name = self.create_collection_name(project.project_id)
        collection_info = await self.vector_client.get_collection_info(collection_name)

        return json.loads(json.dumps(collection_info, default=lambda o: o.__dict__))

    async def index_into_vector_db(self, project: Project, chunks: List[DataChunk], chunk_ids: List[int], do_reset: bool = False):
        """Generate embeddings and index chunks into vector database."""
        collection_name = self.create_collection_name(project.project_id)
        vectors = []
        ids = []
        metadatas = []
        
        # Check if collection exists and validate embedding size
        if await self.vector_client.is_collection_existed(collection_name):
            try:
                collection_info = await self.vector_client.get_collection_info(collection_name)
                existing_size = collection_info.config.params.vectors.size
                expected_size = self.embedding_client.embedding_size
                
                if existing_size != expected_size:
                    error_msg = f"Collection '{collection_name}' exists with dimension {existing_size}, but current embedding model uses {expected_size}. Please use do_reset=true to recreate the collection."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            except AttributeError:
                # If we can't get the size, just log a warning and continue
                logger.warning(f"Could not validate embedding dimensions for collection '{collection_name}'")
        
        # Create collection if not exists (will skip if already exists)
        _ = await self.vector_client.create_collection(collection_name, embedding_size=self.embedding_client.embedding_size)
        
        texts = []
        for chunk, chunk_id in zip(chunks, chunk_ids):
            embedding = self.embedding_client.embed_text(chunk.chunk_text, document_type=DocumentTypeEnum.DOCUMENT.value)
            vectors.append(embedding)
            ids.append(chunk_id)
            texts.append(chunk.chunk_text)
            metadatas.append({
                "chunk_project_id": str(project.project_id),
                "chunk_text": chunk.chunk_text,
                "chunk_order": chunk.chunk_order,
                "chunk_metadata": chunk.chunk_metadata
            })
        # step4: insert into vector db
        _ = await self.vector_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadatas,
            vectors=vectors,
            record_ids=chunk_ids,
        )
        return {
            "indexed_count": len(chunks)
        }
    
    async def search_vector_db(self, project: Project, text: str, limit: int = 5):
        """Search for similar documents in vector database."""
        collection_name = self.create_collection_name(project.project_id)
        query_embedding = self.embedding_client.embed_text(text, document_type=DocumentTypeEnum.QUERY.value)
        search_results = await self.vector_client.search_by_vector(
            collection_name=collection_name,
            vector=query_embedding,
            limit=limit,
        )
        return search_results
    
    async def answer_rag_question(self, project: Project, query: str, limit: int = 5):
        """Generate answer using RAG: retrieve context and generate response with LLM."""
        # step1: search vector db
        search_results = await self.search_vector_db(project, query, limit)
        if not search_results:
            return None, None, None
        
        # Get settings or use defaults
        max_prompt_length = getattr(self.settings, 'RAG_MAX_PROMPT_LENGTH', 8000) if self.settings else 8000
        max_documents = getattr(self.settings, 'RAG_MAX_DOCUMENTS', 5) if self.settings else 5
        
        # Limit number of documents to configured maximum
        actual_limit = min(limit, max_documents)
        search_results = search_results[:actual_limit]
        
        # step2: construct llm prompt
        system_prompt = self.templete_parser.get("rag","system_prompt")
        
        # Extract text from vector search results (compatible with both Qdrant and PGVector)
        document_texts = []
        for idx, doc in enumerate(search_results):
            # PGVector returns RetrievedDocument with text attribute
            if hasattr(doc, 'text') and not hasattr(doc, 'payload'):
                chunk_text = doc.text
            # Qdrant returns ScoredPoint with payload
            elif hasattr(doc, 'payload'):
                chunk_text = doc.payload.get("chunk_text", "")
            else:
                chunk_text = ""
            
            document_texts.append(
                self.templete_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": chunk_text,
                })
            )
        
        documents_prompt = "\n".join(document_texts)
        
        footer_prompt = self.templete_parser.get("rag","footer_prompt",{"query": query})
        
        full_prompt = "\n".join([documents_prompt, footer_prompt])
        
        # Truncate prompt if it exceeds max length
        if len(full_prompt) > max_prompt_length:
            logger.warning(f"Prompt truncated: {len(full_prompt)} -> {max_prompt_length} chars")
            # Truncate from documents, keeping footer intact
            available_space = max_prompt_length - len(footer_prompt) - 2  # -2 for newline
            if available_space > 0:
                documents_prompt = documents_prompt[:available_space]
                full_prompt = "\n".join([documents_prompt, footer_prompt])
            else:
                # If footer is too long, truncate the full prompt
                full_prompt = full_prompt[:max_prompt_length]

        # Construct chat history as list with system message
        system_message = self.generation_client.construct_prompt(
            prompt=system_prompt,
            role=self.generation_client.enums.SYSTEM.value,
        )
        
        chat_history = [system_message]

        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history,
        )

        return answer, full_prompt, chat_history
