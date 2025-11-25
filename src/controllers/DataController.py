import os
from .BaseContoller import BaseController
from models import ResponseStatus
from fastapi import UploadFile
import logging
import re


class DataController(BaseController):
    def __init__(self):
        super().__init__()
        # self.settings is now available here

    def validate_file(self, file: UploadFile) -> bool:
        # Example validation logic using settings
        if file.size > self.settings.FILE_MAX_SIZE:
            logging.warning(f"File size exceeds limit: {file.size} bytes")
            return False, ResponseStatus.FILE_SIZE_EXCEEDED
        
        if not any(file.filename.endswith(ext) for ext in self.settings.FILE_ALLOWED_TYPES):
            logging.warning(f"File type not allowed: {file.filename}")
            return False, ResponseStatus.FILE_TYPE_NOT_SUPPORTED

        return True, ResponseStatus.FILE_VALIDATED_SUCCESS

    def generate_unique_filepath(self, project_id: str, original_filename: str) -> str:
        original_filename = self.get_clean_file_name(original_filename)
        random_str = self.generate_random_string(8)
        unique_filename = f"{random_str}_{original_filename}"
        unique_file_path = self.get_file_path(project_id, unique_filename)
        if not os.path.exists(os.path.dirname(unique_file_path)):
            os.makedirs(os.path.dirname(unique_file_path))
        logging.info(f"Generated unique file path: {unique_file_path}")
        return unique_file_path ,unique_filename
        # Example: /path/to/current/directory/assets/files/{project_id}/{random_str}_{original_filename}

    def get_clean_file_name(self, orig_file_name: str):

        # remove any special characters, except underscore and .
        cleaned_file_name = re.sub(r'[^\w.]', '', orig_file_name.strip())
        # replace spaces with underscore
        cleaned_file_name = cleaned_file_name.replace(" ", "_")
        logging.info(f"Cleaned file name: {cleaned_file_name}")
        return cleaned_file_name

    

    
    
    
