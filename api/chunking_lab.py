import json
import os
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.chunking_lab import ChunkingLabService

router = APIRouter(prefix="/chunking-lab", tags=["chunking-lab"])
service = ChunkingLabService()


@router.post("/visualize")
async def visualize_chunks(
    file: UploadFile = File(...),
    strategy: str = Form(...),
    params: str = Form("{}"),
):
    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in params")

    upload_dir = "uploaded_docs/lab_temp"
    os.makedirs(upload_dir, exist_ok=True)
    file_key = f"{upload_dir}/{uuid.uuid4()}_{file.filename}"

    try:
        with open(file_key, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_type = file.content_type or "text/plain"

        result = service.process_file(file_key, file_type, strategy, parsed_params)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_key):
            os.remove(file_key)


@router.post("/evaluate")
async def evaluate_retrieval(
    file: UploadFile = File(...),
    strategy: str = Form(...),
    query: str = Form(...),
    params: str = Form("{}"),
):
    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in params")

    upload_dir = "uploaded_docs/lab_temp"
    os.makedirs(upload_dir, exist_ok=True)
    file_key = f"{upload_dir}/{uuid.uuid4()}_{file.filename}"

    try:
        with open(file_key, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_type = file.content_type or "text/plain"

        result = service.evaluate_retrieval(
            query, file_key, file_type, strategy, parsed_params
        )
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_key):
            os.remove(file_key)
