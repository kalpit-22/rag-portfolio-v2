import os
import re
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz 

load_dotenv()

# 1. Setup Clients
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-portfolio") 
NAMESPACE = "projects"

# 2. Setup the "Chunker" 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150 
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
    all_chunks = [] 
    
    # --- PHASE 1: Read and Split ---
    print("📖 Reading and chunking files...")
    for root, _, files in os.walk("src/my_projects"):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith((".md", ".txt", ".pdf")):
                try:
                    if file.endswith(".pdf"):
                        doc = fitz.open(file_path)
                        content = "".join([page.get_text() for page in doc])
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                    
                    chunks = text_splitter.split_text(content)
                    for i, chunk_text in enumerate(chunks):
                        chunk_id = f"{clean_id(file_path)}-chunk-{i}"
                        all_chunks.append({
                            "id": chunk_id, 
                            "text": chunk_text, 
                            "source": file_path
                        })
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")

    # --- PHASE 2: Generate Vectors and Upsert in BATCHES ---
    print(f"🚀 Generating Cloud Vectors for {len(all_chunks)} chunks...")
    
    BATCH_SIZE = 96 # Pinecone recommends batches under 100
    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i+BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        
        # A. Get Dense Embeddings (Gemini)
        dense_res = gemini_client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=texts
        )
        dense_vecs = [e.values for e in dense_res.embeddings]
        
        # B. Get Sparse Embeddings (Pinecone Inference)
        sparse_res = pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=texts,
            parameters={"input_type": "passage"} # 'passage' for stored data
        )
        
        # C. Zip and Map exactly how the Database wants it
        vectors_to_upsert = []
        for chunk, dense, sparse in zip(batch, dense_vecs, sparse_res):
            
            # Extract the correct attributes from the Inference object
            # and map them to the keys the Database expects
            mapped_sparse = {
                "indices": sparse.sparse_indices, 
                "values": sparse.sparse_values
            }
            
            vectors_to_upsert.append({
                "id": chunk["id"],
                "values": dense,
                "sparse_values": mapped_sparse,
                "metadata": {
                    "text": chunk["text"], 
                    "source": chunk["source"]
                }
            })
            
        # D. Upsert the whole batch at once
        index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
        print(f"✅ Upserted batch of {len(batch)} chunks.")
        
    print("🎉 Hybrid Ingestion Complete! You are now fully Cloud-Native.")

if __name__ == "__main__":
    ingest_hybrid()