"""VectorDB enumerations for providers and distance methods."""

from enum import Enum


class VectorDBEnums(Enum):
    """Supported VectorDB providers."""
    QDRANT = "QDRANT"
    PGVECTOR = "PGVECTOR"


class DistanceMethodEnums(Enum):
    """Distance calculation methods for vector similarity."""
    COSINE = "cosine"
    DOT = "dot"


class PgVectorTableSchemeEnums(Enum):
    ID = 'id'
    TEXT = 'text'
    VECTOR = 'vector'
    CHUNK_ID = 'chunk_id'
    METADATA = 'metadata'
    _PREFIX = 'pgvector'

class PgVectorDistanceMethodEnums(Enum):
    COSINE = "vector_cosine_ops"
    DOT = "vector_l2_ops"

class PgVectorIndexTypeEnums(Enum):
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"