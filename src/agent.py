import os
from langchain_deepseek import ChatDeepSeek

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from langchain_cohere import CohereRerank
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from src.vector_store import get_permanent_retriever

def format_chat_history(history_dicts):
    """Converts Streamlit's dictionary history into LangChain message objects."""
    langchain_history = []
    for msg in history_dicts:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        else:
            langchain_history.append(AIMessage(content=msg["content"]))
    return langchain_history

def ask_portfolio(query: str, chat_history: list, temp_retriever=None, return_sources: bool = False):
    """The main brain of the SOTA RAG. Merges retrievers, reranks, and answers."""
    
    # 1. Initialize the DeepSeek Brain
    llm = ChatDeepSeek(
        model="deepseek-chat", 
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.2 # Slightly creative but highly factual
    )

    # 2. Build the Hybrid Search Engine
    permanent_retriever = get_permanent_retriever()
    
    if temp_retriever:
        # User uploaded a file! Merge Pinecone and RAM searches
        base_retriever = EnsembleRetriever(
            retrievers=[permanent_retriever, temp_retriever],
            weights=[0.5, 0.5] # Give equal weight to your projects and their upload
        )
    else:
        # Standard search just in your projects
        base_retriever = permanent_retriever

    # 3. The SOTA Filter: Cohere Reranker
    compressor = CohereRerank(
        cohere_api_key=os.getenv("COHERE_API_KEY"), 
        model="rerank-english-v3.0",
        top_n=4 # Only let the absolute best 4 chunks reach DeepSeek
    )
    
    reranker_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 4. The Prompt Engine
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's questions. "
        "The context may include project documentation, resumes, or financial terms and conditions. "
        "Answer based ONLY on the provided context. If the answer isn't there, say you don't know."
        "\n\nContext:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

# 5. Assemble the Chain and Execute
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(reranker_retriever, question_answer_chain)

    # Invoke the chain
    result = rag_chain.invoke({
        "input": query,
        "chat_history": format_chat_history(chat_history)
    })

    # SAFE UNPACKING
    # Look for 'answer' (modern) or 'result' (classic)
    answer = result.get("answer", result.get("result", "I couldn't find an answer."))
    # Look for 'context' (modern) or 'source_documents' (classic)
    sources = result.get("context", result.get("source_documents", []))

    if return_sources:
        return answer, sources
    return answer