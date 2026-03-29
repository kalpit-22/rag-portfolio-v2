import streamlit as st
import os
import tempfile
from src.document_loaders import load_and_split_document

from src.vector_store import create_temporary_retriever
from src.agent import ask_portfolio 

from dotenv import load_dotenv
load_dotenv()

# Initialize Streamlit application
st.set_page_config(page_title="My Engineering Portfolio AI", page_icon="🚀", layout="wide")

# Styling
st.markdown("""
<style>
    .stChatFloatingInputContainer { padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 SOTA Portfolio RAG")
st.subheader("Powered by DeepSeek-V4, Gemini 2 Vision, and Pinecone")

# Initialize session state for messages and the temporary RAM database
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_retriever" not in st.session_state:
    st.session_state.temp_retriever = None

# Sidebar for Recruiter/User uploads
with st.sidebar:
    st.header("1. Upload Documents (Optional)")
    st.write("Upload a job description or PDF. It stays private and deletes when you leave.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, Word, or Text files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Temporary Files", type="primary"):
        if uploaded_files:
            with st.spinner("Creating secure, in-memory vector store..."):
                all_chunks = []
                with tempfile.TemporaryDirectory() as temp_dir:
                    for file in uploaded_files:
                        temp_filepath = os.path.join(temp_dir, file.name)
                        with open(temp_filepath, "wb") as f:
                            f.write(file.getvalue())
                        
                        try:
                            chunks = load_and_split_document(temp_filepath)
                            for chunk in chunks:
                                chunk.metadata["source_name"] = file.name
                            all_chunks.extend(chunks)
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")
                
                # --- THE BIG CHANGE: Save to RAM, not disk ---
                if all_chunks:
                    st.session_state.temp_retriever = create_temporary_retriever(all_chunks)
                    st.success("✅ Files processed and secured in session RAM!")
        else:
            st.warning("Please upload files first.")
            
    st.divider()
    st.header("About")
    st.info(
        "This is a multimodal RAG system. It searches my permanent project documentation "
        "in the cloud alongside any temporary files you upload here. "
        "Built using DeepSeek and Google Gemini Embeddings."
    )

# Main chat interface
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User Input
if prompt := st.chat_input("Ask about my projects, or ask how your uploaded file relates to my skills..."):
    
    # Display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Process with the DeepSeek Agent
    with st.chat_message("assistant"):
        with st.spinner("Searching portfolio and analyzing..."):
            try:
                # We pass the prompt AND the temporary retriever (if it exists) to our new agent
                response = ask_portfolio(
                    query=prompt, 
                    chat_history=st.session_state.messages[:-1], # Pass history excluding current prompt
                    temp_retriever=st.session_state.temp_retriever
                )
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error engaging with agent: {e}\n\nCheck your API keys in the .env file.")