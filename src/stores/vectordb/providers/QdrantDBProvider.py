"""Qdrant vector database provider implementation."""

from ..VectorDBInterface import VectorDBInterface
from qdrant_client import QdrantClient
from ..VectorDBEnums import DistanceMethodEnums
from qdrant_client import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantDBProvider(VectorDBInterface):
    """Qdrant vector database provider."""

    def __init__(self, db_path: str, distance_method: str):
        """Initialize Qdrant provider with database path and distance method."""
        self.client = None
        self.db_path = db_path
        self.distance_method = None

        # Set distance method based on enum value
        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT
        else:
            raise ValueError(f"Unsupported distance method: {distance_method}")

        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Connect to Qdrant database."""
        self.client = QdrantClient(path=self.db_path)

    async def disconnect(self):
        """Disconnect from Qdrant database."""
        if self.client:
            self.client.close()
            self.client = None

    async def is_collection_existed(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return self.client.collection_exists(collection_name=collection_name)

    async def list_all_collections(self) -> list:
        """List all collection names."""
        collections = self.client.get_collections()
        return [collection.name for collection in collections.collections]

    async def get_collection_info(self, collection_name: str):
        """Get information about a collection."""
        return self.client.get_collection(collection_name=collection_name)

    async def delete_collection(self, collection_name: str):
        """Delete a collection. Safely handles non-existent collections."""
        try:
            if await self.is_collection_existed(collection_name):
                self.client.delete_collection(collection_name=collection_name)
                self.logger.info(f"Successfully deleted collection: {collection_name}")
        except Exception as e:
            self.logger.warning(f"Error deleting collection {collection_name}: {e}")

    async def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        """Create a new collection. Optionally reset if exists."""
        if do_reset and await self.is_collection_existed(collection_name):
            await self.delete_collection(collection_name)

        if not await self.is_collection_existed(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )
            return True

        return False

    async def insert_one(self, collection_name: str, text: str, vector: list, metadata: dict = None, record_id: str = None):
        """Insert a single record into collection."""
        if not await self.is_collection_existed(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            return False

        try:
            self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=record_id,
                        vector=vector,
                        payload=metadata
                    )
                ])
        except Exception as e:
            self.logger.error(f"Error inserting record into {collection_name}: {e}")
            return False

        return True

    async def insert_many(self, collection_name: str, texts: list, vectors: list, metadata: list = None, record_ids: list = None, batch_size: int = 50):
        """Insert multiple records into collection in batches."""
        if not await self.is_collection_existed(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist.")
            return False

        # Set defaults if not provided
        if metadata is None:
            metadata = [None] * len(vectors)
        if record_ids is None:
            record_ids = [None] * len(vectors)

        # Process in batches
        total_records = len(vectors)
        for start_idx in range(0, total_records, batch_size):
            end_idx = min(start_idx + batch_size, total_records)
            batch_records = []
            for i in range(start_idx, end_idx):
                record = models.Record(
                    id=record_ids[i],
                    vector=vectors[i],
                    payload=metadata[i]
                )
                batch_records.append(record)
            try:
                self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records
                )
            except Exception as e:
                self.logger.error(f"Error inserting batch into {collection_name}: {e}")
                return False

        return True

    async def search_by_vector(self, collection_name: str, vector: list, limit: int):
        """Search for similar vectors in collection."""
        if not await self.is_collection_existed(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist. Available collections: {await self.list_all_collections()}")
            return None

        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit
            )
            self.logger.info(f"Search in {collection_name} returned {len(search_result) if search_result else 0} results")
            return search_result
        except Exception as e:
            self.logger.error(f"Error searching in {collection_name}: {e}", exc_info=True)
            return None
