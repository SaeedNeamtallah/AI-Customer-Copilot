import os
import random
import string
from helper import Settings, get_settings


class BaseController:
    def __init__(self):
        self.settings: Settings = get_settings()
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Parent directory of the current file
        self.file_path = os.path.join(self.base_path, 'assets/files')

    def get_file_path(self, project_id: int, file_name: str) -> str:
        return os.path.join(self.file_path, str(project_id), file_name)
        # Example: /path/to/current/directory/assets/files/{project_id}/{file_name}

    def get_project_path(self, project_id: int) -> str:
        project_dir = os.path.join(self.file_path, str(project_id))
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        return project_dir
        # Example: /path/to/current/directory/assets/files/{project_id}


    def generate_random_string(self, length: int=12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def get_database_path(self, db_name: str) -> str:
        db_path = os.path.join(self.file_path, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        return db_path
