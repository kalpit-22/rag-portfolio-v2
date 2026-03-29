# 🚀 SOTA Portfolio RAG: Cloud-Native Comparative AI Agent

An advanced, stateless Retrieval-Augmented Generation (RAG) system built to act as an interactive engineering portfolio. This application leverages a multi-stage retrieval architecture, combining serverless hybrid search, in-memory vector stores, and cross-encoder reranking to answer questions with extreme precision.

## 🧠 The "Comparative RAG" Architecture

Unlike standard RAG pipelines that simply pull from a single database, this application features a dual-memory system designed specifically for recruiters and hiring managers. It evaluates permanent portfolio data against temporary, user-uploaded Job Descriptions (JDs) in real-time.

1. **Long-Term Memory (Pinecone):** Stores my permanent project documentation, technical skills, and experience using **Cloud-Native Hybrid Search** (Dense + Learned Sparse Embeddings).
2. **Short-Term Memory (FAISS):** An ephemeral, in-memory vector store that instantly ingests recruiter-uploaded PDFs/JDs via `tempfile` and lives only for the active session.
3. **The Ensemble Engine:** Uses LangChain's `EnsembleRetriever` to query both databases simultaneously, assigning equal weight to my historical experience and the recruiter's active requirements.
4. **The SOTA Filter (Cohere):** Passes the combined retrieved chunks through `rerank-english-v3.0` to filter out noise, ensuring only the top 4 highest-correlated chunks reach the LLM context window.
5. **The Brain (DeepSeek-V4):** Synthesizes the reranked context to generate factual, hallucination-free comparative responses.

## 🛠️ Tech Stack & 2026 Implementations

* **LLM:** DeepSeek-Chat (Highly factual, strict temperature control).
* **Embeddings:** Google Gemini 2.0 (`gemini-embedding-2-preview`).
* **Vector Database:** Pinecone Serverless.
* **Sparse Encoding:** Pinecone Inference API (`pinecone-sparse-english-v0`). *Replaced legacy local BM25 JSON models for a fully stateless, cloud-deployable footprint.*
* **Reranking:** Cohere Contextual Compression.
* **Framework:** LangChain (Core, Community, Classic).
* **Frontend:** Streamlit Community Cloud.

## ⚡ Key Features

* **JSON-Free Hybrid Search:** Utilizes Pinecone's server-side inference for sparse vector generation, completely eliminating the need for local vocabulary fitting or disk-bound BM25 models.
* **Zero-Retention Uploads:** User-uploaded JDs are processed entirely in RAM using FAISS and instantly destroyed upon session termination.
* **Advanced Chunking:** Implements overlapping `RecursiveCharacterTextSplitter` to maintain semantic boundaries across dense technical documentation.
