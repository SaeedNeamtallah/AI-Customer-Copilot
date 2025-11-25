"""
NLP and RAG routes for vector database operations and answer generation.

This module handles document indexing, similarity search, and RAG-based
question answering using vector databases and LLM providers.
"""
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from .schemas.nlp import PushRequest, SearchRequest
import logging

from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers.NLPController import NLPController

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["nlp"]
)

@router.post("/push/{project_id}")
async def push_endpoint(project_id: int, req: Request, payload: PushRequest):
    """Index project documents to vector database with embeddings."""
    project_model = await ProjectModel.create_instance(req.app.state.async_session)
    chunk_model = await ChunkModel.create_instance(req.app.state.async_session)

    project = await project_model.get_project_or_create_one(project_id)
    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Project with id {project_id} not found."}
        )

    # 3) NLP controller
    nlp_controller = NLPController(
        vector_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        templete_parser=req.app.state.template_parser,
        settings=req.app.state.settings
    )

    # 4) reset 
    if payload.do_reset:
        await nlp_controller.reset_vector_db_collection(project=project)

    # 5) paginate + index all pages
    page_no = max(payload.page or 1, 1)
    page_size = max(min(payload.page_size or 100, 1000), 1)
    total_indexed = 0

    while True:
        # Fetch chunks page by page
        chunk_list = await chunk_model.get_project_chunks_paginated(
            project_object_id=project.project_id,
            page=page_no,
            page_size=page_size
        )

        # 
        if not chunk_list:
            if total_indexed == 0 and page_no == max(payload.page or 1, 1):
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"message": f"No chunks found for project {project_id}"}
                )
            break

        # Extract actual chunk_ids from the database records
        chunk_ids = [chunk.chunk_id for chunk in chunk_list]

        index_result = await nlp_controller.index_into_vector_db(
            project=project,
            chunks=chunk_list,
            chunk_ids=chunk_ids,
            do_reset=False
        )

        indexed_now = int(index_result.get("indexed_count", 0)) if index_result else 0
        if indexed_now == 0:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"Indexing failed for project {project_id}"}
            )

        total_indexed += indexed_now
        # Next page
        page_no += 1

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Push successful",
            "project_id": project_id,
            "indexed_count": total_indexed
        }
    )




@router.get("/index/info/{project_id}")
async def get_index_info_endpoint(project_id: int, req: Request):
    """Get vector database collection information for a project."""
    project_model = await ProjectModel.create_instance(req.app.state.async_session)

    project = await project_model.get_project_or_create_one(project_id)
    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Project with id {project_id} not found."}
        )

    nlp_controller = NLPController(
        vector_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        templete_parser=req.app.state.template_parser,
        settings=req.app.state.settings
    )

    try:
        collection_info = await nlp_controller.get_vector_db_collection_info(project=project)
        if collection_info is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"Vector DB collection for project {project_id} not found."}
            )
        
    except Exception as e:
        logger.error("Error getting collection info for project %s: %s", project_id, e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Failed to retrieve collection info: {str(e)}"}
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "project_id": project_id,
            "collection_info": collection_info
        }
    )



@router.post("/search/{project_id}")
async def search_endpoint(project_id: int, req: Request, payload: SearchRequest):
    """Search for similar documents using vector similarity."""
    project_model = await ProjectModel.create_instance(req.app.state.async_session)

    project = await project_model.get_project_or_create_one(project_id)
    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Project with id {project_id} not found."}
        )

    nlp_controller = NLPController(
        vector_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        templete_parser=req.app.state.template_parser,
        settings=req.app.state.settings
        
    )

    try:
        search_results = await nlp_controller.search_vector_db(
            project=project,
            text=payload.text,
            limit=payload.limit or 5
        )

        if search_results is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"No results found or collection doesn't exist for project {project_id}. Try pushing with do_reset=true first."}
            )

        # Handle empty results
        if len(search_results) == 0:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"No search results found for query in project {project_id}"}
            )

        # Serialize vector search results - return only text and score
        serialized_results = []
        for result in search_results:
            # PGVector returns RetrievedDocument with text and score
            if hasattr(result, 'text'):
                serialized_results.append({
                    "text": result.text,
                    "score": result.score
                })
            # Qdrant returns ScoredPoint - extract text from payload
            elif hasattr(result, 'payload'):
                serialized_results.append({
                    "text": result.payload.get("chunk_text", ""),
                    "score": result.score
                })

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "project_id": project_id,
                "query": payload.text,
                "results_count": len(serialized_results),
                "results": serialized_results
            }
        )

    except Exception as e:
        logger.error("Error searching in project %s: %s", project_id, e, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Search failed: {str(e)}"}
        )



@router.post("/generate/{project_id}")
async def generate_endpoint(project_id: int, req: Request, payload: SearchRequest):
    """Generate AI-powered answers using RAG (Retrieval-Augmented Generation)."""
    project_model = await ProjectModel.create_instance(req.app.state.async_session)

    project = await project_model.get_project_or_create_one(project_id)
    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": f"Project with id {project_id} not found."}
        )

    nlp_controller = NLPController(
        vector_client=req.app.state.vector_db_client,
        generation_client=req.app.state.generation_client,
        embedding_client=req.app.state.embedding_client,
        templete_parser=req.app.state.template_parser,
        settings=req.app.state.settings
    )

    try:
        answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
            project=project,
            query=payload.text,
            limit=payload.limit or 5
        )

        if answer is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"No results found to generate answer for project {project_id}"}
            )

    except Exception as e:
        logger.error("Error generating response for project %s: %s", project_id, e, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Generation failed: {str(e)}"}
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "project_id": project_id,
            "query": payload.text,
            "answer": answer,
            "context_documents_count": len(full_prompt.split("## Document No:")) - 1 if full_prompt else 0,
            "full_prompt": full_prompt,
            "chat_history": chat_history,
            
        
        }
    )

