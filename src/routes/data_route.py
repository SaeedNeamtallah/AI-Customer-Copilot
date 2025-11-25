"""
Data processing routes for file upload and document processing.

This module handles file uploads, document chunking, and data persistence
for RAG system projects.
"""
import logging
from fastapi import APIRouter, status, UploadFile,Request
from fastapi.responses import JSONResponse
from .schemas.dataproces_schemas import ProcessFileRequest 
from controllers import DataController, ProcessControllers
import aiofiles
import os
from models import ProjectModel ,ChunkModel
from models.db_schemas import DataChunk 
from models.db_schemas import Asset 
from models.AssetModel import AssetModel

logger = logging.getLogger(__name__)

datarouter= APIRouter(
    prefix="/api/v1/data",
    tags=["data"],
)

@datarouter.post("/upload/{project_id}")
async def process_data(request: Request, project_id: int, file: UploadFile):
    """Upload a file to a project and create asset record."""
    project_model = await ProjectModel.create_instance(request.app.state.async_session)
    # Use get_or_create to automatically create project if it doesn't exist
    project = await project_model.get_project_or_create_one(project_id)

    data_controller = DataController()
    is_valid, response_status = data_controller.validate_file(file)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": response_status.value, "message": "File validation failed."}
        )

    unique_file_path , unique_filename = data_controller.generate_unique_filepath(project_id, file.filename)

    try:
        async with aiofiles.open(unique_file_path, 'wb') as out_file:
            content = await file.read()  # async read
            if not content or len(content) == 0:
                logging.error(f"File {file.filename} is empty")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"status": "file_empty", "message": "Uploaded file is empty."}
                )
            await out_file.write(content)  # async write
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "file_upload_failed", "message": "Failed to upload file."}
        )

    asset_model = await AssetModel.create_instance(request.app.state.async_session)
    asset_resources = Asset(
        asset_project_id=project.project_id,
        asset_type="file",
        asset_name=unique_filename,
        asset_size=os.path.getsize(unique_file_path))
    asset_record = await asset_model.create_asset(asset_resources)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "file_upload_success", 
                 "file_path": f"{unique_file_path}",
                 "file_id": f"{unique_filename}",
                 "asset_id": str(asset_record.asset_id)
                 }
    )



@datarouter.post("/processall/{project_id}")
async def process_all_files(request: Request, project_id: int):
    """Process all files in a project directory into chunks and save to database."""
    data_controller = DataController()
    process_controller = ProcessControllers()

    project_model = await ProjectModel.create_instance(request.app.state.async_session)
    project = await project_model.get_project_or_create_one(project_id)

    if not project:
        logging.warning(f"Project not found for project_id: {project_id}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "project_not_found", "message": "Project not found."}
        )

    project_path = data_controller.get_project_path(project_id)
    if not project_path:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "project_not_found", "message": "Project not found."}

        )
    all_files = [f for f in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, f))]
    if not all_files:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "no_files_found", "message": "No files found in the project."}
        )
    
    # Get the project's ObjectId for chunk references
    project_object_id = project.project_id if hasattr(project, 'project_id') and project.project_id else None
    if not project_object_id:
        logger.error(f"Project {project_id} has no valid ObjectId")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "Project ObjectId not found."}
        )
    
    asset_model = await AssetModel.create_instance(request.app.state.async_session)
    chunk_model = await ChunkModel.create_instance(request.app.state.async_session)
    
    all_chunks_data = []
    failed_files = []
    total_inserted_chunks = 0
    
    # Process each file separately to maintain asset association
    for file_name in all_files:
        file_path = data_controller.get_file_path(project_id, file_name)
        
        try:
            # Process the document to get chunks
            chunks = process_controller.process_document(file_path)
            
            if not chunks:
                logging.warning(f"No chunks extracted from file {file_name}")
                continue
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Determine file type from extension
            file_extension = os.path.splitext(file_name)[1].lower()
            
            # Create Asset record for this file
            asset = Asset(
                asset_type=file_extension.lstrip('.'),
                asset_name=file_name,
                asset_size=file_size,
                asset_config={"source": file_path},
                asset_project_id=project_object_id
            )
            created_asset = await asset_model.create_asset(asset)
            
            # Convert langchain Document objects to DataChunk objects with asset_id
            file_chunks = [DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=idx + 1,
                chunk_project_id=project_object_id,
                chunk_asset_id=created_asset.asset_id
            ) for idx, chunk in enumerate(chunks)]
            
            # Insert chunks for this file (returns count as integer)
            await chunk_model.insert_many_chunks(file_chunks)
            total_inserted_chunks += len(file_chunks)
            all_chunks_data.extend([{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks])
            
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}", exc_info=True)
            failed_files.append({"file": file_name, "error": str(e)})
            continue  # Skip files that cause errors
    
    if total_inserted_chunks == 0:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "processing_failed", 
                "message": "Failed to process any files. All files are either empty or invalid.",
                "failed_files": failed_files
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "processing_success", 
            "total_files": len(all_files),
            "processed_files": len(all_files) - len(failed_files),
            "failed_files": len(failed_files),
            "total_chunks": len(all_chunks_data),
            "inserted_chunks": total_inserted_chunks,
            "failed_file_details": failed_files if failed_files else [],
            "chunks": all_chunks_data

        }
    )

@datarouter.post("/processone/{project_id}")
async def process_one_file(request: Request, project_id: int, body: ProcessFileRequest):
    """Process a single file into chunks with optional reset."""
    file_name = body.file_id  # file_id is actually the filename
    chunk_size = body.chunk_size
    overlap_size = body.overlap_size
    do_reset = body.do_reset

    project_model = await ProjectModel.create_instance(request.app.state.async_session)
    project = await project_model.get_project_or_create_one(project_id)


    if not project:
        logging.warning(f"Project not found for project_id: {project_id}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "project_not_found", "message": "Project not found."}
        )


    # logging.info(f"Processing one file request for project_id: {project_id}, file_name: {file_name}")
    data_controller = DataController()
    process_controller = ProcessControllers()

    project_path = data_controller.get_project_path(project_id)
    if not project_path:
        logging.warning(f"Project path not found for project_id: {project_id}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "project_not_found", "message": "Project not found."}

        )
    
    file_path = data_controller.get_file_path(project_id, file_name)
    if not os.path.exists(file_path):
        logger.warning(f"File not found at path: {file_path}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"status": "file_not_found", "message": f"File {file_name} not found in the project."}
        )
    
    try:
        chunks = process_controller.process_document(file_path, chunk_size, overlap_size)
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {e}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "processing_failed", 
                "message": f"Failed to process file {file_name}. It may be empty or invalid.",
                "error": str(e)
            }
        )
    
    
    if not chunks:
        logger.warning(f"No chunks were created from file: {file_name}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "processing_failed", 
                "message": f"Failed to process file {file_name}. It may be empty or invalid."
            }
        )
    
    # Get the project's ObjectId for chunk references
    project_object_id = project.project_id if hasattr(project, 'project_id') and project.project_id else None
    if not project_object_id:
        logger.error(f"Project {project_id} has no valid ObjectId")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "Project ObjectId not found."}
        )
    
    # Get or create Asset record for this file
    asset_model = await AssetModel.create_instance(request.app.state.async_session)
    asset_record = await asset_model.get_asset_record(project_object_id, file_name)
    
    if not asset_record:
        # Create new asset if it doesn't exist
        asset_record = Asset(
            asset_project_id=project_object_id,
            asset_type="file",
            asset_name=file_name,
            asset_size=os.path.getsize(file_path)
        )
        asset_record = await asset_model.create_asset(asset_record)
    
    file_chunks = [DataChunk(
        chunk_text=chunk.page_content,
        chunk_metadata=chunk.metadata,
        chunk_order=idx + 1,
        chunk_project_id=project_object_id,
        chunk_asset_id=asset_record.asset_id
                                            ) for idx, chunk in enumerate(chunks)]

    chunk_model = await ChunkModel.create_instance(request.app.state.async_session)
    
    
    if do_reset:
        await chunk_model.del_chunks_by_project_id(project_object_id)

    await chunk_model.insert_many_chunks(file_chunks)


    # logging.info(f"Successfully processed file {file_name}, created {len(chunks)} chunks.")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "processing_success", 
            "file_name": file_name,
            "total_chunks": len(chunks),
            "chunks": [ {"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks ]
        }
    )



