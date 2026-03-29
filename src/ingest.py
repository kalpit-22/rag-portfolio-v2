import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_text_splitters import RecursiveCharacterTextSplitter # Required for chunking
import fitz 

load_dotenv()

# 1. Setup Clients
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-portfolio") 
NAMESPACE = "projects"
bm25 = BM25Encoder()

# 2. Setup the "Chunker" 
# 1000 chars is roughly 2 paragraphs - perfect for finding specific "caps" or "rules"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150 # Overlap ensures we don't cut a sentence in half
)

def clean_id(raw_str):
    ascii_str = raw_str.encode("ascii", "ignore").decode("ascii")
    return re.sub(r'[^a-zA-Z0-9\-_]', '-', ascii_str)

def get_dense_embedding(text):
    result = gemini_client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=[text]
    )
    return result.embeddings[0].values

def ingest_hybrid():
    all_chunks = [] # Stores (chunk_id, text, source_path)
    
    # --- PHASE 1: Read, Split, and Learn Vocabulary ---
    print("📖 Reading and chunking files...")
    for root, _, files in os.walk("src/my_projects"):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith((".md", ".txt", ".pdf")):
                if file.endswith(".pdf"):
                    doc = fitz.open(file_path)
                    content = "".join([page.get_text() for page in doc])
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                
                # BREAK THE FILE INTO SMALLER CHUNKS
                chunks = text_splitter.split_text(content)
                for i, chunk_text in enumerate(chunks):
                    # Unique ID for every chunk
                    chunk_id = f"{clean_id(file_path)}-chunk-{i}"
                    all_chunks.append({
                        "id": chunk_id, 
                        "text": chunk_text, 
                        "source": file_path
                    })

    # "Fit" BM25 on the chunks (the actual search units)
    print("🧠 Training BM25 on vocabulary...")
    bm25.fit([c["text"] for c in all_chunks])
    
    # --- PHASE 2: Generate Vectors and Upsert ---
    print(f"🚀 Upserting {len(all_chunks)} chunks to Pinecone...")
    for chunk in all_chunks:
        dense_vec = get_dense_embedding(chunk["text"])
        sparse_vec = bm25.encode_queries(chunk["text"]) 
        
        index.upsert(
            vectors=[{
                "id": chunk["id"],
                "values": dense_vec,
                "sparse_values": sparse_vec,
                "metadata": {
                    "text": chunk["text"], # THE FULL CHUNK, NO TRUNCATION
                    "source": chunk["source"]
                }
            }],
            namespace=NAMESPACE
        )
    
    bm25.dump("bm25_model.json")
    print("✅ Hybrid Ingestion Complete with Chunking!")

if __name__ == "__main__":
    ingest_hybrid()