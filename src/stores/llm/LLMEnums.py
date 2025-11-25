"""
LLM Enumerations Module

This module defines enumerations for different LLM providers and their
specific role types, as well as document type enums for embeddings.
"""

from enum import Enum


class LLMEnums(Enum):
    """Supported LLM providers."""
    OPENAI = "OPENAI"
    COHERE = "COHERE"


class OpenAIEnums(Enum):
    """Role types for OpenAI chat messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class CoHereEnums(Enum):
    """Role and input types for Cohere API."""
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"
    
    # Embedding input types
    DOCUMENT = "search_document"
    QUERY = "search_query"


class DocumentTypeEnum(Enum):
    """Document types for embedding generation."""
    DOCUMENT = "document"
    QUERY = "query"