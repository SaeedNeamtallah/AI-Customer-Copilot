from typing import List, Optional
from datetime import datetime, timezone
from .db_schemas import DataChunk 
from sqlalchemy.future import select
from sqlalchemy import func, delete
from .BaseDataModel import BaseDataModel

class ChunkModel(BaseDataModel):
    def __init__(self, db_client):
        super().__init__(db_client)

    @classmethod
    async def create_instance(cls, db_client):
        return cls(db_client)
    
    async def create_chunk(self,chunk:DataChunk):
        async with self.db_client() as session:
            async with session.begin():
                session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
        return chunk
    
    async def get_chunk(self,chunk_id:int) -> Optional[DataChunk]:
        async with self.db_client() as session:
            result = await session.execute(
                select(DataChunk).where(DataChunk.chunk_id == chunk_id)
            )
            chunk = result.scalars().first()
        return chunk
    
    async def insert_many_chunks(self,chunks:List[DataChunk],batch=100):
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(chunks), batch):
                    batch_chunks = chunks[i:i+batch]
                    session.add_all(batch_chunks)
            await session.commit()
        return len(chunks)
    
    async def del_chunks_by_project_id(self,project_id:int):
        async with self.db_client() as session:
            async with session.begin():
                stmt = delete(DataChunk).where(DataChunk.chunk_project_id == project_id)
                result = await session.execute(stmt)
            await session.commit()
        return result.rowcount
    
    async def get_total_chunk_count(self,project_id):

        async with self.db_client() as session:
            result = await session.execute(
                select(func.count(DataChunk.chunk_id)).where(DataChunk.chunk_project_id == project_id)
            )
            total_count = result.scalar_one()
        return total_count
    
    async def get_project_chunks_paginated(self,project_object_id:int,page:int=1,page_size:int=100) -> List[DataChunk]:

        async with self.db_client() as session:
            offset = (page - 1) * page_size
            result = await session.execute(
                select(DataChunk)
                .where(DataChunk.chunk_project_id == project_object_id)
                .order_by(DataChunk.chunk_order)
                .offset(offset)
                .limit(page_size)
            )
            chunks = result.scalars().all()
        return chunks
    
    