import os
import time
import shutil
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

# ল্যাংচেইনের কোর মডিউলসমূহ
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="SaaS PDF Chatbot Pro", page_icon="📚", layout="wide")

# =====================================================================
# 📂 STORAGE CONFIGURATION
# =====================================================================
UPLOAD_DIR = "uploaded_pdfs"       
VECTOR_DB_DIR = "faiss_index_db"   

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# =====================================================================
# 🔐 SYSTEM CONFIGURATION & SECURITY FALLBACK
# =====================================================================
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 Critical Error: API Key missing! Please set your GOOGLE_API_KEY in .streamlit/secrets.toml")
    st.stop()

# এমবেডিংস মডেল ইনিশিয়ালাইজেশন
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key
)

# =====================================================================
# 🎨 PREMIUM UI CSS STYLING (SaaS Pro Dashboard)
# =====================================================================
st.markdown("""
<style>
.stApp { 
    background: linear-gradient(135deg, #0b0914, #121026, #1a1738);
    font-family: 'Inter', system-ui, sans-serif;
}
.user-msg {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #ffffff !important; padding: 14px 20px; border-radius: 20px 20px 4px 20px;
    margin: 12px 0; max-width: 75%; margin-left: auto; 
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.25);
    font-size: 0.95rem; line-height: 1.5;
}
.bot-msg {
    background: rgba(255, 255, 255, 0.05); 
    color: #f3f4f6 !important; padding: 14px 20px; border-radius: 20px 20px 20px 4px;
    margin: 12px 0; max-width: 75%; 
    border: 1px solid rgba(255, 255, 255, 0.08); 
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    font-size: 0.95rem; line-height: 1.5;
}
h1 { color: #ffffff !important; text-align: center; font-weight: 800; letter-spacing: -0.5px; }
.subtitle { color: #9ca3af; text-align: center; margin-bottom: 2.5rem; font-size: 1.1rem; }
.stTextInput input {
    background-color: #f3f4f6 !important; 
    color: #111827 !important;            
    border: 2px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important; padding: 12px 20px !important; font-size: 1rem !important;
}
.stTextInput input:focus { border-color: #6366f1 !important; background-color: #ffffff !important; }
.stButton button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important; border: none !important; border-radius: 12px !important; 
    font-weight: 600 !important; padding: 0.6rem 2.2rem !important; transition: all 0.2s ease;
}
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4); }
section[data-testid="stSidebar"] { background-color: #0c0a17 !important; border-right: 1px solid rgba(255, 255, 255, 0.08); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #ffeb3b !important; font-weight: 700 !important; }
.file-card {
    background: rgba(255, 255, 255, 0.04); padding: 12px; border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1); color: #ffffff !important; font-size: 0.9rem; font-weight: 500; margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 📚 PDF Chatbot Pro")
st.markdown('<p class="subtitle">High-Speed SaaS Document AI Dashboard</p>', unsafe_allow_html=True)

# =====================================================================
# ⚡ HIGH-SPEED TEXT EXTRACTION PIPELINE (Error-Free)
# =====================================================================
def process_and_save_pdfs(uploaded_files, existing_vectorstore=None):
    all_documents = []
    
    if "file_map" not in st.session_state:
        st.session_state.file_map = {}

    for idx_f, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name in st.session_state.file_map:
            continue
            
        safe_name = f"safe_file_{idx_f}_{int(time.time())}.pdf"
        saved_path = os.path.join(UPLOAD_DIR, safe_name)
        
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.session_state.file_map[uploaded_file.name] = safe_name
        
        # হাই-স্পীড লোডার (Poppler বা OCR এর ঝামেলা মুক্ত)
        try:
            loader = PyMuPDFLoader(saved_path)
            loaded_docs = loader.load()
            # সোর্স মেটাডেটা ঠিক করা
            for d in loaded_docs:
                d.metadata["source"] = uploaded_file.name
            all_documents.extend(loaded_docs)
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {str(e)}")
        
    if not all_documents:
        return existing_vectorstore

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_documents)
    
    if not chunks:
        st.error("🚨 PDF থেকে কোনো টেক্সট উদ্ধার করা সম্ভব হয়নি।")
        return existing_vectorstore
        
    vectorstore = existing_vectorstore
    total_chunks = len(chunks)
    
    progress_bar = st.progress(0, text="Initializing Database Sync...")
    
    for idx, chunk in enumerate(chunks):
        if not chunk.page_content.strip():
            continue
            
        success = False
        retry_count = 0
        
        while not success and retry_count < 5:
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents([chunk], embeddings)
                else:
                    vectorstore.add_documents([chunk])
                success = True
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    retry_count += 1
                    time.sleep(3 * retry_count) 
                else:
                    st.error(f"Unexpected Pipeline Error: {str(e)}")
                    st.stop()
        
        progress_percent = min(100, int((idx + 1) / total_chunks * 100))
        progress_bar.progress(progress_percent, text=f"🔒 High-Precision Indexing... {idx + 1}/{total_chunks}")
            
    progress_bar.empty()  
    
    if vectorstore:
        vectorstore.save_local(VECTOR_DB_DIR)
        
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =====================================================================
# 🗑️ INDIVIDUAL FILE DELETION LOGIC
# =====================================================================
def delete_individual_file(filename_to_delete):
    if st.session_state.vectorstore is not None:
        index_to_id = st.session_state.vectorstore.index_to_docstore_id
        docstore = st.session_state.vectorstore.docstore
        
        ids_to_delete = []
        for idx, doc_id in list(index_to_id.items()):
            doc = docstore.search(doc_id)
            if doc and doc.metadata.get("source") == filename_to_delete:
                ids_to_delete.append(doc_id)
        
        if ids_to_delete:
            st.session_state.vectorstore.delete(ids_to_delete)
            st.session_state.vectorstore.save_local(VECTOR_DB_DIR)
            
        if "file_map" in st.session_state and filename_to_delete in st.session_state.file_map:
            safe_name = st.session_state.file_map[filename_to_delete]
            file_path = os.path.join(UPLOAD_DIR, safe_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            del st.session_state.file_map[filename_to_delete]
            
        if "last_processed_files" in st.session_state and filename_to_delete in st.session_state.last_processed_files:
            st.session_state.last_processed_files.remove(filename_to_delete)

        if not st.session_state.vectorstore.index_to_docstore_id:
            if os.path.exists(VECTOR_DB_DIR):
                shutil.rmtree(VECTOR_DB_DIR)
            st.session_state.vectorstore = None

# =====================================================================
# 💾 PERSISTENT STATE MANAGEMENT
# =====================================================================
if "vectorstore" not in st.session_state:
    if os.path.exists(VECTOR_DB_DIR):
        try:
            st.session_state.vectorstore = FAISS.load_local(
                VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            st.session_state.vectorstore = None
    else:
        st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_processed_files" not in st.session_state:
    st.session_state.last_processed_files = []

# =====================================================================
# 🎛️ SIDEBAR CONTROL PANEL
# =====================================================================
with st.sidebar:
    st.markdown("## 📂 Storage Control Panel")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files:
        current_file_names = [f.name for f in uploaded_files]
        new_files = [f for f in uploaded_files if f.name not in st.session_state.last_processed_files]
        
        if new_files:
            with st.spinner("💾 Syncing Documents to Database..."):
                st.session_state.vectorstore = process_and_save_pdfs(
                    new_files, 
                    existing_vectorstore=st.session_state.vectorstore
                )
            st.session_state.last_processed_files = current_file_names
            st.success("✅ Database updated!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📄 Currently Saved Files:")
    
    active_files = set()
    if st.session_state.vectorstore is not None:
        index_to_id = st.session_state.vectorstore.index_to_docstore_id
        docstore = st.session_state.vectorstore.docstore
        for doc_id in index_to_id.values():
            doc = docstore.search(doc_id)
            if doc and "source" in doc.metadata:
                active_files.add(doc.metadata["source"])
                
    if active_files:
        for file in sorted(active_files):
            col_file, col_btn = st.columns([4, 1.5])
            with col_file:
                st.markdown(f'<div class="file-card">📁 {file}</div>', unsafe_allow_html=True)
            with col_btn:
                if st.button("🗑️ Delete", key=f"del_{file}", use_container_width=True):
                    delete_individual_file(file)
                    st.rerun()
    else:
        st.markdown('<p style="color: #9ca3af; font-size: 0.9rem;">No files permanently saved yet.</p>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    if st.button("🚨 Wipe All Files & Reset", key="global_reset", use_container_width=True):
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        if os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR)
        
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.last_processed_files = []
        if "file_map" in st.session_state:
            st.session_state.file_map = {}
        st.success("All storage cleared!")
        st.rerun()

# =====================================================================
# 💬 CHAT INTERFACE & RETRIEVAL LOGIC
# =====================================================================
if st.session_state.vectorstore is None:
    st.info("👋 Welcome! System storage is currently empty. Please upload PDF documents in the sidebar to start chatting.")
else:
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_retries=2 
    )

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-msg">🙋 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">🤖 {message["content"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Question",
                placeholder="Ask anything about your saved documents...",
                label_visibility="collapsed"
            )
        with col2:
            submit_button = st.form_submit_button(
                label="Send ➤",
                type="primary",
                use_container_width=True
            )

    if submit_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            reterived_docs = retriever.invoke(user_input)
            context = format_docs(reterived_docs)
            
            system_prompt = (
                "You are a professional, high-precision Document AI Assistant.\n"
                "Answer the question strictly based on the provided PDF context below.\n"
                "If the answer cannot be derived from the context, respond exactly with:\n"
                "'This information is not available in the provided documents.'\n"
                "Do not answer anything outside the document scope.\n"
                "Respond in Bengali if the user asks in Bengali, otherwise in English.\n\n"
                "Context:\n{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            rag_chain = qa_prompt | llm | StrOutputParser()
            
            try:
                response_text = rag_chain.invoke({
                    "context": context,
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                
                source_files = list(set([
                    doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
                    for doc in reterived_docs
                ]))
                if source_files and "This information is not available" not in response_text:
                    response_text += f"\n\n📎 **Source:** {', '.join(source_files)}"
                    
                st.session_state.chat_history.extend([
                    ("human", user_input),
                    ("ai", response_text),
                ])
                
            except Exception as net_error:
                response_text = (
                    "⚠️ **Network Timeout Error:** গুগলের সার্ভারের সাথে সংযোগ সাময়িকভাবে বিচ্ছিন্ন হয়ে গেছে।"
                )

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()
