
from .db_schemas import Project
from .BaseDataModel import BaseDataModel
from sqlalchemy.future import select
from sqlalchemy import func

class ProjectModel(BaseDataModel):
    def __init__(self, db_client) -> None:
        super().__init__(db_client)
        # self.db_client = db_client # base class already sets self.db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance


    async def create_project(self, project: Project):
        async with self.db_client() as session:
            async with session.begin():
                session.add(project)
            await session.commit()
        return project
    
    async def get_project_or_create_one(self, project_name: str) -> Project:
        async with self.db_client() as session:
            result = await session.execute(
                select(Project).where(Project.project_id == project_name)
            )
            project = result.scalars().first()
            if project:
                return project
            # Create new project if not found
            new_project = Project(project_id=project_name)
            session.add(new_project)
            await session.commit()
            await session.refresh(new_project)
            return new_project
        

    async def get_all_projects(self, page: int=1, page_size: int=10):

        async with self.db_client() as session:
            result = await session.execute(select(func.count(Project.project_id)))
            total_projects = result.scalar_one()

            total_pages = (total_projects + page_size - 1) // page_size

            if page > total_pages and total_pages != 0:
                return [], total_pages
            if total_pages == 0:
                return [], 0
            
            offset = (page - 1) * page_size

            result = await session.execute(
                select(Project).offset(offset).limit(page_size)
            )
            projects = result.scalars().all()
            return projects, total_pages