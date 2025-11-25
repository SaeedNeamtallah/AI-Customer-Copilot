from pydantic import BaseModel
from typing import Optional

class ProcessFileRequest(BaseModel):
    file_id: str  # This is the filename (e.g., "lmj99sjz_EfficientPythonforDataScientists.pdf")
    chunk_size: Optional[int] = None  # Optional, will use default from settings if not provided
    overlap_size: Optional[int] = None  # Optional, will use default from settings if not provided
    do_reset: Optional[int] = 0