import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

def get_embeddings():
    """Gemini 2.0 Multimodal Embeddings"""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def get_permanent_retriever():
    """Cloud Hybrid Search (Dense + Sparse) Optimized for Reranking"""
    embeddings = get_embeddings()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("rag-portfolio")
    
    # Load vocabulary from bm25_model.json
    bm25 = BM25Encoder().load("bm25_model.json")
    
    return PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        # SOTA Tweak 1: Increase top_k to 10. 
        # We grab more chunks so the Cohere Reranker has more to choose from.
        top_k=10, 
        # SOTA Tweak 2: Alpha 0.3. 
        # 0.0 is pure keyword, 1.0 is pure semantic. 
        # 0.3 leans harder on keywords (BM25) to find terms like "HSBC" or "Cap".
        alpha=0.3,
        namespace="projects",
        text_key="text" 
    )

def create_temporary_retriever(document_chunks):
    """In-memory search for recruiter-uploaded files"""
    embeddings = get_embeddings()
    # FAISS is fast enough that we don't need Hybrid for small session files
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})