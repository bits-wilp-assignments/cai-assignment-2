from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.service.indexing import triggr_indexing
from src.service.inference import rag_inference
from src.config.app_config import FIXED_WIKI_PAGE_FILE, RANDOM_WIKI_PAGE_FILE
from src.util.logging_util import get_logger
from typing import Optional, Dict, Any, Union
import asyncio
from datetime import datetime
from src.config import app_config
import json
from pathlib import Path

app = FastAPI(title="Hybrid RAG System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger(__name__)

# Global state for indexing status
indexing_status = {
    "status": "idle",  # idle, running, completed, failed
    "message": None,
    "results": None,
    "started_at": None,
    "completed_at": None
}


class IndexRequest(BaseModel):
    refresh_fixed: bool = False
    refresh_random: bool = False
    limit: Optional[int] = None


class InferenceRequest(BaseModel):
    question: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Hybrid RAG System is running"}


def run_indexing_task(refresh_fixed: bool, refresh_random: bool, limit: int = None):
    """Background task to run indexing"""
    global indexing_status
    try:
        indexing_status["status"] = "running"
        indexing_status["started_at"] = datetime.now().isoformat()
        indexing_status["message"] = "Indexing in progress..."

        dense_result, sparse_result = triggr_indexing(
            is_refresh_fixed=refresh_fixed,
            is_refresh_random=refresh_random,
            limit=limit
        )

        indexing_status["status"] = "completed"
        indexing_status["completed_at"] = datetime.now().isoformat()
        indexing_status["message"] = "Indexing completed successfully"
        indexing_status["results"] = {
            "dense": dense_result,
            "sparse": sparse_result
        }
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        indexing_status["status"] = "failed"
        indexing_status["completed_at"] = datetime.now().isoformat()
        indexing_status["message"] = f"Indexing failed: {str(e)}"
        indexing_status["results"] = None


@app.post("/index/trigger")
async def trigger_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """Trigger indexing process in background"""
    global indexing_status

    if indexing_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Indexing is already in progress")

    logger.info(f"Indexing triggered with refresh_fixed={request.refresh_fixed}, refresh_random={request.refresh_random}, limit={request.limit}")

    # Reset status
    indexing_status = {
        "status": "running",
        "message": "Indexing started...",
        "results": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None
    }

    background_tasks.add_task(run_indexing_task, request.refresh_fixed, request.refresh_random, request.limit)

    return {
        "status": "accepted",
        "message": "Indexing task started in background"
    }


@app.get("/index/status")
async def get_indexing_status():
    """Check indexing status"""
    return indexing_status


@app.post("/inference")
async def inference_stream(request: InferenceRequest):
    """Run RAG inference with SSE streaming"""
    logger.info(f"Inference triggered for question: {request.question}")

    async def event_generator():
        try:
            for chunk in rag_inference(request.question, is_streaming=True):
                # SSE format: data: {content}\n\n
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/config")
async def get_all_config():
    """Get all configuration parameters"""
    config_dict = {}

    # Get all config variables from app_config module
    for key in dir(app_config):
        if not key.startswith("_"):  # Exclude private variables
            value = getattr(app_config, key)
            # Exclude imported modules and functions
            if not callable(value) and not key == "logging":
                config_dict[key] = value

    return {
        "status": "success",
        "config": config_dict
    }


@app.get("/wiki/fixed")
async def get_fixed_wiki_pages():
    """Get list of fixed wiki pages"""
    try:
        fixed_wiki_path = Path(FIXED_WIKI_PAGE_FILE)

        if not fixed_wiki_path.exists():
            raise HTTPException(status_code=404, detail="Fixed wiki pages file not found")

        with open(fixed_wiki_path, 'r', encoding='utf-8') as f:
            wiki_pages = json.load(f)

        return {
            "status": "success",
            "count": len(wiki_pages.get('pages', [])),
            "data": wiki_pages
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse fixed wiki JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to read fixed wiki pages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/wiki/random")
async def get_random_wiki_pages():
    """Get list of random wiki pages"""
    try:
        random_wiki_path = Path(RANDOM_WIKI_PAGE_FILE)

        if not random_wiki_path.exists():
            raise HTTPException(status_code=404, detail="Random wiki pages file not found")

        with open(random_wiki_path, 'r', encoding='utf-8') as f:
            wiki_pages = json.load(f)

        return {
            "status": "success",
            "count": len(wiki_pages.get('pages', [])),
            "data": wiki_pages
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse random wiki JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to read random wiki pages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
