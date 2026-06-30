from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.documents import router as documents_router
from api.chat import router as chat_router
from api.chunking_lab import router as chunking_lab_router

app = FastAPI(title="doc-intel API", description="AI Research Copilot Backend")

app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(chunking_lab_router)
# Enable CORS for the frontend Vite app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific origin of Vite in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
