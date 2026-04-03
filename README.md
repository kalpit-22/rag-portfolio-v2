# 🚀 Comparative RAG Portfolio Agent

An advanced, cloud-native Retrieval-Augmented Generation (RAG) system designed to act as an interactive professional portfolio. 

Instead of a standard Q&A bot, this agent utilizes a **Split-Brain Memory Architecture** (Pinecone + FAISS) and a **LangChain Ensemble Retriever**. It allows recruiters to upload a Job Description (JD) and dynamically cross-references their exact requirements against my historical projects (like DevPilot), resume, and technical documentation.

---

## 🧠 Architecture Overview

The system is built on a highly optimized, four-pillar RAG pipeline:

1. **Dual-Memory Retrieval (Split-Brain):**
   * **Long-Term Memory:** Pinecone Serverless acts as the permanent knowledge base containing my professional history (CNH Industrial experience, deployed projects, etc.).
   * **Short-Term Memory:** An ephemeral FAISS vector store sits in RAM to process user-uploaded files (JDs), deleting them securely the moment the session ends.
2. **Cloud-Native Hybrid Search:**
   * **Dense Vectors (Gemini 2.0 Multimodal):** Captures semantic meaning and conceptual alignment.
   * **Sparse Vectors (Pinecone Inference):** Captures exact keyword dominance (BM25 equivalent) via a custom `CloudSparseEncoder` wrapper, bypassing local dependencies.
3. **Contextual Compression (SOTA Filtering):**
   * A **Cohere Cross-Encoder** intercepts the retrieved chunks from both databases, reads them simultaneously against the user's prompt, and aggressively filters out noise, passing only the top 6 most mathematically relevant chunks to the generator.
4. **The Generative Brain:**
   * Powered by **Gemini 2.5 Flash**, the final generation step is locked behind strict system prompts designed to prevent hallucination, enforce professional tone, and block the leakage of Personally Identifiable Information (PII).

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Cloud-hosted chat interface and session state management |
| **Orchestration** | LangChain | Ensemble logic, routing, and chain construction |
| **LLM (Brain)** | Gemini 2.5 Flash | Blazing fast, highly factual response synthesis |
| **Embeddings** | Gemini 2.0 Preview | 768-dimensional semantic text mapping |
| **Permanent DB** | Pinecone Serverless | Hybrid (Dense + Sparse) cloud vector storage |
| **Ephemeral DB** | FAISS (CPU) | High-speed, in-memory RAM vector storage |
| **Reranker** | Cohere v3.0 | Cross-encoder contextual compression |

---

## 📂 Project Structure

```text
rag-portfolio-v2/
├── app.py                     # Streamlit frontend & memory management
├── ingest.py                  # ETL pipeline (Chunking, Hybrid API calls, Upsert)
├── requirements.txt           # Production dependencies
├── src/
│   ├── agent.py               # Ensemble logic, Cohere filtering, & Gemini chains
│   ├── vector_store.py        # Database routing & custom CloudSparseEncoder
│   ├── document_loaders.py    # Factory line for PDF, DOCX, and TXT chunking
│   └── my_projects/           # Source Markdown/PDFs for Pinecone ingestion
