from .BaseContoller import BaseController
from models import ProcessingEnum 
from langchain_community.document_loaders import PyMuPDFLoader ,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import os


class ProcessControllers(BaseController):
    def __init__(self):
        super().__init__()
        #self.settings: Settings = get_settings()

    def process_document(self, file_path: str, chunk_size: int = None, overlap_size: int = None):
        file_extension = os.path.splitext(file_path)[1] # Get the file extension (e.g., .txt, .pdf) #os.path.splitext("example.pdf") == ('example', '.pdf')

        if file_extension == ProcessingEnum.TXT.value:
            loader = TextLoader(file_path)
        elif file_extension == ProcessingEnum.PDF.value:
            loader = PyMuPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not loader:
            raise ValueError(f"Failed to initialize loader for file: {file_path}")
        

        document = loader.load()
        
        # Use provided chunk_size and overlap_size, or fall back to settings
        chunk_size_to_use = chunk_size if chunk_size is not None else self.settings.CHUNK_SIZE
        overlap_size_to_use = overlap_size if overlap_size is not None else self.settings.CHUNK_OVERLAP
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_to_use,
            chunk_overlap=overlap_size_to_use,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        file_content="".join([doc.page_content for doc in document])
        if not file_content or file_content.strip() == "":
            raise ValueError(f"File content is empty for file: {file_path}")

        file_content_metadata= "".join([str(doc.metadata) for doc in document])
        if not file_content_metadata:
            logging.warning(f"No metadata found in file: {file_path}")

        chunks = text_splitter.create_documents([file_content], metadatas=[{"source": file_path, "metadata": file_content_metadata}])
        if not chunks:
            raise ValueError(f"Text splitting resulted in no chunks for file: {file_path}")


        return chunks # List of Document objects
        # Example: [Document(page_content="...", metadata={"source": "...", ...}), ...]
        # Each Document object contains a chunk of text and its associated metadata

        
        
