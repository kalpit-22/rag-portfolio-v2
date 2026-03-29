import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import PineconeHybridSearchRetriever

def get_embeddings():
    """Gemini 2.0 Multimodal Embeddings"""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

class CloudSparseEncoder:
    """SOTA Wrapper: Makes Pinecone Inference API act like LangChain's expected BM25 object."""
    def __init__(self, pc_client):
        self.pc = pc_client

    def encode_queries(self, text: str):
        # We use input_type="query" here so the AI knows we are asking a question
        res = self.pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=[text],
            parameters={"input_type": "query"} 
        )
        
        # Return the exact dictionary format LangChain's Hybrid Retriever expects
        return {
            "indices": res[0].sparse_indices,
            "values": res[0].sparse_values
        }

def get_permanent_retriever():
    """Cloud Hybrid Search (Dense + Sparse) Optimized for Reranking"""
    embeddings = get_embeddings()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("rag-portfolio")
    
    # SOTA 2026: We replaced BM25Encoder().load() with our new Cloud Encoder
    cloud_sparse_encoder = CloudSparseEncoder(pc)
    
    return PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=cloud_sparse_encoder,
        index=index,
        top_k=10, 
        alpha=0.3, # This still works perfectly! It scales the dense/sparse weights.
        namespace="projects",
        text_key="text" 
    )

def create_temporary_retriever(document_chunks):
    """In-memory search for recruiter-uploaded files"""
    embeddings = get_embeddings()
    # FAISS is fast enough that we don't need Hybrid for small session files
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})