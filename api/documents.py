import os
import shutil
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Document
from db.session import get_session
from services.embeddings import EmbeddingService

router = APIRouter()


class DocumentCreate(BaseModel):
    title: str
    file_key: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    size: int
    status: str
    uploaded_at: datetime


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Document))
    documents = result.scalars().all()

    return [
        DocumentResponse(
            id=doc.id,
            name=doc.title,
            size=doc.document_metadata.get("file_size", "")
            if doc.document_metadata
            else "",
            status=doc.status,
            uploaded_at=doc.created_at,
        )
        for doc in documents
    ]


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    user_id = 1

    if file.content_type not in ["application/pdf"]:
        raise HTTPException(400, "Unsupported file type")

    # upload to local directory
    upload_dir = f"uploaded_docs/{user_id}"
    os.makedirs(upload_dir, exist_ok=True)
    file_key = f"{upload_dir}/{uuid.uuid4()}_{file.filename}"

    with open(file_key, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    service = EmbeddingService()
    num_chunks = service.upload_pdf(file_key)

    print(f"Stored {num_chunks} chunks")

    document = Document(
        user_id=user_id,
        title=file.filename,
        document_metadata={
            "file_key": file_key,
            "content_type": file.content_type,
            "file_size": file.size,
        },
    )

    session.add(document)
    await session.commit()
    await session.refresh(document)

    # enqueue background job
    # process_document.delay(document.id)

    return {
        "id": document.id,
        "title": document.title,
        "status": "uploaded",
    }
