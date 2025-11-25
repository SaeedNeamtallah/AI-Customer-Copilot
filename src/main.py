"""
FastAPI application for RAG System.

This module initializes the FastAPI application with PostgreSQL, LLM providers,
and VectorDB providers for document processing, embedding generation, and 
retrieval-augmented generation.
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from routes import base_router, datarouter ,nlp_router
from helper import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce SQLAlchemy logging verbosity
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)

# Reduce httpx logging (for API calls to Cohere, OpenAI)
logging.getLogger('httpx').setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown operations."""

    settings = get_settings()
    
    # Store settings in app.state for access in routes
    app.state.settings = settings

    # Startup - PostgreSQL Connection
    logger.info("Connecting to PostgreSQL...")
    try:
        # Use asyncpg driver for async PostgreSQL connections
        postgres_conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"

        app.state.db_engine = create_async_engine(postgres_conn, echo=False)  # Disable SQL echo
        app.state.async_session = sessionmaker(
            bind=app.state.db_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("✅ Connected to PostgreSQL")


    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
        raise

    # Startup - LLM Provider Factory
    try:
        llm_factory = LLMProviderFactory(settings)

        # Initialize generation provider
        app.state.generation_client = llm_factory.get_provider(settings.GENERATION_BACKEND)
        app.state.generation_client.set_generation_model(settings.GENERATION_MODEL_ID)

        # Initialize embedding provider
        app.state.embedding_client = llm_factory.get_provider(settings.EMBEDDING_BACKEND)
        app.state.embedding_client.set_embedding_model(
            settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_SIZE
        )
        
        logger.info(f"✅ LLM initialized: {settings.GENERATION_BACKEND}/{settings.GENERATION_MODEL_ID}, Embedding: {settings.EMBEDDING_BACKEND}/{settings.EMBEDDING_MODEL_ID} ({settings.EMBEDDING_SIZE}D)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM providers: {e}")
        raise

    # Startup - VectorDB Provider Factory
    try:
        vectordb_factory = VectorDBProviderFactory(settings)
        app.state.vector_db_client = vectordb_factory.get_provider(settings.VECTOR_DB_BACKEND)
        
        # If using PGVector, inject the db session
        from stores.vectordb.VectorDBEnums import VectorDBEnums
        if settings.VECTOR_DB_BACKEND == VectorDBEnums.PGVECTOR.value:
            app.state.vector_db_client.db_client = app.state.async_session
            await app.state.vector_db_client.connect()
        else:
            await app.state.vector_db_client.connect()
        
        logger.info(f"✅ VectorDB initialized: {settings.VECTOR_DB_BACKEND}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize VectorDB provider: {e}")
        raise

    # Startup - TemplateParser for RAG
    try:
        from stores.llm.templete.templete_parser import TemplateParser
        app.state.template_parser = TemplateParser()
    except Exception as e:
        logger.error(f"❌ Failed to initialize TemplateParser: {e}")
        raise

    logger.info("✅ Application started successfully")
    
    yield

    # Shutdown - Close all connections
    try:
        if hasattr(app.state, "vector_db_client") and app.state.vector_db_client:
            await app.state.vector_db_client.disconnect()
        if hasattr(app.state, "db_engine") and app.state.db_engine:
            await app.state.db_engine.dispose()
        logger.info("✅ Application shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Initialize FastAPI application
app = FastAPI(
    title="MiniRAG API",
    description="A lightweight Retrieval-Augmented Generation (RAG) application for document processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Include routers
app.include_router(base_router)
app.include_router(datarouter)
app.include_router(nlp_router)