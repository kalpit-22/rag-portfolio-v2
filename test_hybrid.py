from src.document_loaders import load_and_split_document
from src.vector_store import add_documents_to_vector_store, get_hybrid_retriever
from src.agent import get_rag_agent
from langchain_core.messages import HumanMessage
import os

def test():
    # 1. Ingest
    print("Ingesting dummy doc...")
    with open("dummy.txt", "w") as f:
        f.write("Pradhyumn Singh Jadon is a Data Scientist and works with Analytics.")
    docs = load_and_split_document("dummy.txt")
    add_documents_to_vector_store(docs)
    
    # 2. Test Retriever
    print("\nTesting Hybrid Retriever directly for 'Pradhyumn Singh Jadon'...")
    retriever = get_hybrid_retriever(k=5)
    result_docs = retriever.invoke("Pradhyumn Singh Jadon")
    for d in result_docs:
        print(f"- {d.page_content}")
        
    # 3. Test Agent
    print("\nTesting Agent...")
    agent = get_rag_agent()
    # Provide a real question about the data
    messages = [HumanMessage(content="Is there any information about Pradhyumn Singh Jadon?")]
    response = agent.invoke({"messages": messages})
    print(response["messages"][-1].content)
    
    # cleanup
    os.remove("dummy.txt")

if __name__ == "__main__":
    test()
