import streamlit as st
import os
import tempfile
from src.document_loaders import load_and_split_document
from src.vector_store import create_temporary_retriever
from src.agent import ask_portfolio 
from dotenv import load_dotenv

load_dotenv()

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="Pradhyumn | AI Engineer Portfolio", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium CSS Injection
st.markdown("""
<style>
    /* Gradient Title */
    .title-gradient {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Subtitle tweaking */
    .subtitle {
        color: #888888;
        font-size: 1.2em;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    
    /* Input container padding */
    .stChatFloatingInputContainer { 
        padding-bottom: 2rem; 
    }
</style>
""", unsafe_allow_html=True)

# 3. Header Section
st.markdown('<h1 class="title-gradient">🚀 Pradhyumn\'s AI Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by DeepSeek-V4, Gemini 2, and Pinecone Serverless</p>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_retriever" not in st.session_state:
    st.session_state.temp_retriever = None

# 4. Sleek Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103285.png", width=50) # Tiny decorative icon
    st.header("Comparative RAG Engine")
    st.caption("Upload a Job Description to see how my skills match your exact requirements. Files are processed in RAM and destroyed upon exit.")
    
    st.divider()
    
    uploaded_files = st.file_uploader(
        "📄 Drop JD or PDF here", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Analyze Document 🔍", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Encrypting and indexing in RAM..."):
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
                
                if all_chunks:
                    st.session_state.temp_retriever = create_temporary_retriever(all_chunks)
                    # Modern Toast Notification instead of bulky success box
                    st.toast('Document indexed successfully!', icon='✅')
        else:
            st.warning("Please upload a file first.", icon="⚠️")
            
    st.divider()
    st.info("**Architecture:**\n\n🔹 LangChain Ensemble\n🔹 Cohere Reranking\n🔹 Pinecone Inference API")

# 5. The "Cold Start" Welcome Screen
if len(st.session_state.messages) == 0:
    st.markdown("""
    ### 👋 Welcome! I am Pradhyumn's Virtual Assistant.
    I've been trained on his resume, GitHub, and technical documentation. Feel free to ask me anything!
    
    **Try asking:**
    * 💡 *"How did you save $300K+ at CNH Industrial?"*
    * ⚙️ *"What is your experience with Agentic AI and LLMs?"*
    * 📊 *"Explain the architecture of your DevPilot project."*
    """)

# 6. Main Chat Interface (With Custom Avatars)
for msg in st.session_state.messages:
    # Use custom emojis for a cleaner look
    avatar = "👤" if msg["role"] == "user" else "🤖"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# User Input
if prompt := st.chat_input("Ask about my projects, skills, or upload a JD to compare..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching neural database..."):
            try:
                response = ask_portfolio(
                    query=prompt, 
                    chat_history=st.session_state.messages[:-1],
                    temp_retriever=st.session_state.temp_retriever
                )
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Agent Offline: {e}\n\nCheck API keys.")