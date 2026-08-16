import os
import tempfile
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.document_loaders import load_and_split_document
from src.vector_store import create_temporary_retriever
from src.agent import ask_portfolio

load_dotenv()

app = FastAPI(title="Pradhyumn's AI Portfolio API")

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

# Dictionary to hold temporary retrievers (In-Memory per session)
session_retrievers = {}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage]
    session_id: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert pydantic models to dict format expected by ask_portfolio
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        # Get temporary retriever if the user uploaded documents in this session
        temp_retriever = session_retrievers.get(request.session_id)
        
        response = ask_portfolio(
            query=request.query,
            chat_history=history_dicts,
            temp_retriever=temp_retriever
        )
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(session_id: str = Form(...), files: List[UploadFile] = File(...)):
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
            retriever = create_temporary_retriever(all_chunks)
            session_retrievers[session_id] = retriever
            return {"message": "Files processed and indexed successfully."}
        else:
            raise HTTPException(status_code=400, detail="No readable content found in files.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
