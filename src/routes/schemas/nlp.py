from pydantic import BaseModel
from typing import Optional

class PushRequest(BaseModel):
    do_reset: Optional[bool] = False
    page: Optional[int] = 1
    page_size: Optional[int] = 50


class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5