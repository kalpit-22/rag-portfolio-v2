import os
import tempfile
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Request
from mangum import Mangum
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
# We use a custom IP extractor for AWS Lambda instead of get_remote_address
from slowapi.errors import RateLimitExceeded

from src.document_loaders import load_and_split_document
from src.vector_store import upload_temporary_documents, get_temporary_retriever, delete_temporary_documents
from src.agent import ask_portfolio

load_dotenv()

app = FastAPI(title="Pradhyumn's AI Portfolio API")

# Setup Rate Limiting for AWS Lambda
def get_real_ip(request: Request) -> str:
    """Extract real client IP when deployed behind AWS API Gateway/Lambda."""
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"

limiter = Limiter(key_func=get_real_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Stateless architecture: temporary retrievers are dynamically instantiated via Pinecone namespaces

class ChatMessage(BaseModel):
    role: str
    content: str

class SessionEndRequest(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage]
    session_id: str

@app.post("/api/chat")
@limiter.limit("20/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    try:
        # Convert pydantic models to dict format expected by ask_portfolio
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in body.chat_history]
        
        # Instantiate the temporary retriever for the session.
        # If the user hasn't uploaded anything (empty namespace), Pinecone safely returns 0 results.
        temp_retriever = get_temporary_retriever(body.session_id)
        
        response = ask_portfolio(
            query=body.query,
            chat_history=history_dicts,
            temp_retriever=temp_retriever
        )
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
@limiter.limit("5/minute")
async def upload_document(request: Request, session_id: str = Form(...), files: List[UploadFile] = File(...)):
    try:
        all_chunks = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                temp_filepath = os.path.join(temp_dir, file.filename)
                with open(temp_filepath, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                try:
                    chunks = load_and_split_document(temp_filepath)
                    for chunk in chunks:
                        chunk.metadata["source_name"] = file.filename
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file.filename}: {e}")
        
        if all_chunks:
            upload_temporary_documents(all_chunks, session_id)
            return {"message": "Files processed and indexed successfully."}
        else:
            raise HTTPException(status_code=400, detail="No readable content found in files.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/end-session")
async def end_session(body: SessionEndRequest):
    try:
        delete_temporary_documents(body.session_id)
        return {"message": "Session data cleaned up"}
    except Exception as e:
        return {"message": "Error"}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

# AWS Lambda Handler
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
