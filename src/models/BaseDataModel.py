# file: models/BaseDataModel.py
from helper.config import get_settings, Settings


class BaseDataModel:
    def __init__(self, db_client) -> None:
        self.settings: Settings = get_settings()
        self.db_client = db_client
        
