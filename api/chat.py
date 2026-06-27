from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.chat_message import ChatMessage
from db.session import get_session
from services.embeddings import EmbeddingService

router = APIRouter()


class ChatRequest(BaseModel):
    document_id: int
    query: str


class ChatResponse(BaseModel):
    id: str
    role: str
    content: str


@router.post("/chat")
async def chat_with_documents(
    request: ChatRequest, session: AsyncSession = Depends(get_session)
):
    user_msg = ChatMessage(
        document_id=request.document_id, role="user", content=request.query
    )
    session.add(user_msg)
    await session.commit()

    service = EmbeddingService()
    return StreamingResponse(
        service.chat_stream(request.query, request.document_id, session),
        media_type="text/event-stream",
    )


@router.get("/chat/{document_id}")
async def get_chat_history(
    document_id: int, session: AsyncSession = Depends(get_session)
):
    result = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.document_id == document_id)
        .order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()

    return [
        {"id": f"msg-{msg.id}", "role": msg.role, "content": msg.content, "citations": msg.citations}
        for msg in messages
    ]
