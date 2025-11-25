import json
import logging
from typing import List

from sqlalchemy.sql import text as sql_text

from models.db_schemas import RetrievedDocument
from ..VectorDBEnums import (
    DistanceMethodEnums,
    PgVectorDistanceMethodEnums,
    PgVectorIndexTypeEnums,
    PgVectorTableSchemeEnums,
)
from ..VectorDBInterface import VectorDBInterface


class PGVectorProvider(VectorDBInterface):
    """PostgreSQL with pgvector extension provider for vector database operations."""

    def __init__(self, db_client, default_vector_size: int = 1024,
                 distance_method: str = None, index_threshold: int = 100):
        """
        Initialize PGVector provider.
        
        Args:
            db_client: SQLAlchemy async session factory
            default_vector_size: Default vector dimension size
            distance_method: Distance calculation method (cosine or dot)
            index_threshold: Minimum records before creating index
        """
        self.db_client = db_client
        self.default_vector_size = default_vector_size
        self.index_threshold = index_threshold
        self.logger = logging.getLogger(__name__)

        # Map distance method to pgvector operators
        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = PgVectorDistanceMethodEnums.COSINE.value
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = PgVectorDistanceMethodEnums.DOT.value
        else:
            # Default to cosine if not specified
            self.distance_method = PgVectorDistanceMethodEnums.COSINE.value
        
        self.pgvector_table_prefix = PgVectorTableSchemeEnums._PREFIX.value
        self.default_index_name = lambda collection_name: f"{collection_name}_vector_idx"

    async def connect(self):
        """Connect and ensure pgvector extension is installed."""
        async with self.db_client() as session:
            try:
                # Check if vector extension already exists
                result = await session.execute(sql_text(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                ))
                extension_exists = result.scalar_one_or_none()
                
                if not extension_exists:
                    # Only create if it doesn't exist
                    await session.execute(sql_text("CREATE EXTENSION vector"))
                    await session.commit()
                    
            except Exception as e:
                # If extension already exists or any other error, just log and continue
                self.logger.debug(f"Vector extension setup: {str(e)}")
                await session.rollback()

    async def disconnect(self):
        """Disconnect from database (no-op as connection is managed by db_client)."""
        pass

    async def is_collection_existed(self, collection_name: str) -> bool:
        """Check if a collection (table) exists."""
        async with self.db_client() as session:
            try:
                list_tbl = sql_text('SELECT 1 FROM pg_tables WHERE tablename = :collection_name')
                result = await session.execute(list_tbl, {"collection_name": collection_name})
                exists = result.scalar_one_or_none()
                return exists is not None
            except Exception as e:
                self.logger.error(f"Error checking collection existence: {e}")
                return False
    
    async def list_all_collections(self) -> List[str]:
        """List all vector collections (tables with pgvector prefix)."""
        async with self.db_client() as session:
            try:
                list_tbl = sql_text('SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix')
                results = await session.execute(list_tbl, {"prefix": f"{self.pgvector_table_prefix}%"})
                records = results.scalars().all()
                return list(records) if records else []
            except Exception as e:
                self.logger.error(f"Error listing collections: {e}")
                return []
    
    async def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a collection."""
        async with self.db_client() as session:
            try:
                table_info_sql = sql_text('''
                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes 
                    FROM pg_tables 
                    WHERE tablename = :collection_name
                ''')

                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')

                table_info = await session.execute(table_info_sql, {"collection_name": collection_name})
                table_data = table_info.fetchone()
                
                if not table_data:
                    return None
                
                record_count = await session.execute(count_sql)
                
                return {
                    "table_info": {
                        "schemaname": table_data[0],
                        "tablename": table_data[1],
                        "tableowner": table_data[2],
                        "tablespace": table_data[3],
                        "hasindexes": table_data[4],
                    },
                    "record_count": record_count.scalar_one(),
                }
            except Exception as e:
                self.logger.error(f"Error getting collection info: {e}")
                return None
            
    async def delete_collection(self, collection_name: str):
        """Delete a collection (table)."""
        async with self.db_client() as session:
            try:
                delete_sql = sql_text(f'DROP TABLE IF EXISTS {collection_name} CASCADE')
                await session.execute(delete_sql)
                await session.commit()
                return True
            except Exception as e:
                self.logger.error(f"Error deleting collection {collection_name}: {e}")
                await session.rollback()
                return False

    async def create_collection(self, collection_name: str,
                                embedding_size: int,
                                do_reset: bool = False):
        """Create a new collection (table) for vector storage."""
        if do_reset:
            await self.delete_collection(collection_name=collection_name)

        is_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_existed:
            async with self.db_client() as session:
                try:
                    create_sql = sql_text(
                        f'CREATE TABLE {collection_name} ('
                        f'{PgVectorTableSchemeEnums.ID.value} bigserial PRIMARY KEY, '
                        f'{PgVectorTableSchemeEnums.TEXT.value} text, '
                        f'{PgVectorTableSchemeEnums.VECTOR.value} vector({embedding_size}), '
                        f'{PgVectorTableSchemeEnums.METADATA.value} jsonb DEFAULT \'{{}}\', '
                        f'{PgVectorTableSchemeEnums.CHUNK_ID.value} integer, '
                        f'FOREIGN KEY ({PgVectorTableSchemeEnums.CHUNK_ID.value}) '
                        f'REFERENCES chunks(chunk_id) ON DELETE CASCADE'
                        ')'
                    )
                    await session.execute(create_sql)
                    await session.commit()
                    self.logger.debug(f"Successfully created collection: {collection_name}")
                    return True
                except Exception as e:
                    self.logger.error(f"Error creating collection {collection_name}: {e}")
                    await session.rollback()
                    return False
        
        return False
    
    async def is_index_existed(self, collection_name: str) -> bool:
        """Check if vector index exists for a collection."""
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            try:
                check_sql = sql_text(""" 
                    SELECT 1 
                    FROM pg_indexes 
                    WHERE tablename = :collection_name
                    AND indexname = :index_name
                """)
                result = await session.execute(check_sql, {
                    "index_name": index_name,
                    "collection_name": collection_name
                })
                exists = result.scalar_one_or_none()
                return exists is not None
            except Exception as e:
                self.logger.error(f"Error checking index existence: {e}")
                return False
            
    async def create_vector_index(self, collection_name: str,
                                   index_type: str = PgVectorIndexTypeEnums.HNSW.value):
        """Create vector index if threshold is met and index doesn't exist."""
        is_existed = await self.is_index_existed(collection_name=collection_name)
        if is_existed:
            self.logger.debug(f"Index already exists for {collection_name}")
            return False
        
        async with self.db_client() as session:
            try:
                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')
                result = await session.execute(count_sql)
                records_count = result.scalar_one()

                if records_count < self.index_threshold:
                    self.logger.debug(f"Record count ({records_count}) below threshold ({self.index_threshold})")
                    return False
                
                self.logger.debug(f"Creating vector index for collection: {collection_name} ({records_count} records)")
                
                index_name = self.default_index_name(collection_name)
                create_idx_sql = sql_text(
                    f'CREATE INDEX {index_name} ON {collection_name} '
                    f'USING {index_type} ({PgVectorTableSchemeEnums.VECTOR.value} {self.distance_method})'
                )

                await session.execute(create_idx_sql)
                await session.commit()
                self.logger.debug(f"Successfully created vector index for collection: {collection_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error creating vector index: {e}")
                await session.rollback()
                return False

    async def reset_vector_index(self, collection_name: str, 
                                  index_type: str = PgVectorIndexTypeEnums.HNSW.value) -> bool:
        """Drop and recreate vector index."""
        index_name = self.default_index_name(collection_name)
        async with self.db_client() as session:
            try:
                drop_sql = sql_text(f'DROP INDEX IF EXISTS {index_name}')
                await session.execute(drop_sql)
                await session.commit()
                self.logger.debug(f"Dropped index: {index_name}")
            except Exception as e:
                self.logger.error(f"Error dropping index: {e}")
                await session.rollback()
        
        return await self.create_vector_index(collection_name=collection_name, index_type=index_type)

    async def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, record_id: int = None):
        """Insert a single vector record."""
        is_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_existed:
            self.logger.error(f"Cannot insert into non-existent collection: {collection_name}")
            return False
        
        if not record_id:
            self.logger.error(f"Cannot insert record without chunk_id")
            return False
        
        async with self.db_client() as session:
            try:
                insert_sql = sql_text(
                    f'INSERT INTO {collection_name} '
                    f'({PgVectorTableSchemeEnums.TEXT.value}, '
                    f'{PgVectorTableSchemeEnums.VECTOR.value}, '
                    f'{PgVectorTableSchemeEnums.METADATA.value}, '
                    f'{PgVectorTableSchemeEnums.CHUNK_ID.value}) '
                    'VALUES (:text, :vector, :metadata, :chunk_id)'
                )
                
                metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata is not None else "{}"
                vector_str = "[" + ",".join([str(v) for v in vector]) + "]"
                
                await session.execute(insert_sql, {
                    'text': text,
                    'vector': vector_str,
                    'metadata': metadata_json,
                    'chunk_id': record_id
                })
                await session.commit()
                
                # Try to create index if threshold is met
                await self.create_vector_index(collection_name=collection_name)
                return True
                
            except Exception as e:
                self.logger.error(f"Error inserting record: {e}")
                await session.rollback()
                return False

    async def insert_many(self, collection_name: str, texts: list,
                          vectors: list, metadata: list = None,
                          record_ids: list = None, batch_size: int = 50):
        """Insert multiple vector records in batches."""
        is_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_existed:
            self.logger.error(f"Cannot insert into non-existent collection: {collection_name}")
            return False
        
        if len(vectors) != len(record_ids):
            self.logger.error(f"Vectors count ({len(vectors)}) != record_ids count ({len(record_ids)})")
            return False
        
        if not metadata or len(metadata) == 0:
            metadata = [{}] * len(texts)
        
        async with self.db_client() as session:
            try:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_vectors = vectors[i:i + batch_size]
                    batch_metadata = metadata[i:i + batch_size]
                    batch_record_ids = record_ids[i:i + batch_size]

                    values = []
                    for _text, _vector, _metadata, _record_id in zip(
                        batch_texts, batch_vectors, batch_metadata, batch_record_ids
                    ):
                        metadata_json = json.dumps(_metadata, ensure_ascii=False) if _metadata is not None else "{}"
                        vector_str = "[" + ",".join([str(v) for v in _vector]) + "]"
                        
                        values.append({
                            'text': _text,
                            'vector': vector_str,
                            'metadata': metadata_json,
                            'chunk_id': _record_id
                        })
                    
                    batch_insert_sql = sql_text(
                        f'INSERT INTO {collection_name} '
                        f'({PgVectorTableSchemeEnums.TEXT.value}, '
                        f'{PgVectorTableSchemeEnums.VECTOR.value}, '
                        f'{PgVectorTableSchemeEnums.METADATA.value}, '
                        f'{PgVectorTableSchemeEnums.CHUNK_ID.value}) '
                        f'VALUES (:text, :vector, :metadata, :chunk_id)'
                    )
                    
                    await session.execute(batch_insert_sql, values)
                    self.logger.debug(f"Inserted batch {i // batch_size + 1} ({len(values)} records)")
                
                await session.commit()
                
                # Try to create index if threshold is met
                await self.create_vector_index(collection_name=collection_name)
                return True
                
            except Exception as e:
                self.logger.error(f"Error inserting batch: {e}")
                await session.rollback()
                return False
    
    async def search_by_vector(self, collection_name: str, vector: list, limit: int) -> List[RetrievedDocument]:
        """Search for similar vectors using cosine distance."""
        is_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_existed:
            self.logger.error(f"Cannot search in non-existent collection: {collection_name}")
            return []
        
        vector_str = "[" + ",".join([str(v) for v in vector]) + "]"
        
        async with self.db_client() as session:
            try:
                # Using cosine distance: 1 - (vector <=> query_vector)
                # <=> is the cosine distance operator in pgvector
                search_sql = sql_text(
                    f'SELECT {PgVectorTableSchemeEnums.TEXT.value} as text, '
                    f'1 - ({PgVectorTableSchemeEnums.VECTOR.value} <=> :vector) as score '
                    f'FROM {collection_name} '
                    'ORDER BY score DESC '
                    f'LIMIT :limit'
                )
                
                result = await session.execute(search_sql, {"vector": vector_str, "limit": limit})
                records = result.fetchall()
                
                self.logger.debug(f"Search in {collection_name} returned {len(records)} records")

                return [
                    RetrievedDocument(
                        text=record.text,
                        score=float(record.score)
                    )
                    for record in records
                ]
                
            except Exception as e:
                self.logger.error(f"Error searching collection {collection_name}: {e}")
                return []


