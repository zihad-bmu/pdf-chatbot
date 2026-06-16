import os
import io
import time
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ল্যাংচেইনের কোর মডিউলসমূহ
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# ইন-মেমোরি পিডিএফ রিড করার জন্য লাইব্রেরি
import fitz  

st.set_page_config(page_title="SaaS PDF Chatbot Pro", page_icon="📚", layout="wide")

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
st.markdown('<p class="subtitle">State-Locked In-Memory SaaS Document AI</p>', unsafe_allow_html=True)

# =====================================================================
# 💾 STATE MANAGEMENT INITIALIZATION (Persistent Across Reruns)
# =====================================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# =====================================================================
# ⚡ IN-MEMORY TEXT EXTRACTION PIPELINE (No Disk Permissions Needed)
# =====================================================================
def process_pdfs_in_memory(uploaded_files):
    all_documents = []
    
    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.read()
            pdf_stream = io.BytesIO(file_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    all_documents.append(Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name, "page": page_num + 1}
                    ))
            doc.close()
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
            
    if not all_documents:
        return st.session_state.vectorstore

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = text_splitter.split_documents(all_documents)
    
    if not chunks:
        st.error("🚨 PDF থেকে কোনো টেক্সট পাওয়া যায়নি।")
        return st.session_state.vectorstore
        
    progress_bar = st.progress(0, text="Synchronizing State-Locked Database...")
    
    try:
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            st.session_state.vectorstore.add_documents(chunks)
        progress_bar.progress(100, text="🔒 Memory Indexing Complete!")
        time.sleep(0.5)
    except Exception as e:
        st.error(f"🚨 Indexing Failed: {str(e)}")
            
    progress_bar.empty()  
    return st.session_state.vectorstore

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
            
        if filename_to_delete in st.session_state.uploaded_file_names:
            st.session_state.uploaded_file_names.remove(filename_to_delete)

        if not st.session_state.vectorstore.index_to_docstore_id:
            st.session_state.vectorstore = None

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
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_file_names]
        
        if new_files:
            with st.spinner("💾 Syncing Documents..."):
                process_pdfs_in_memory(new_files)
            for f in new_files:
                st.session_state.uploaded_file_names.append(f.name)
            st.success("✅ Database updated!")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📄 Currently Saved Files:")
    
    if st.session_state.uploaded_file_names and st.session_state.vectorstore is not None:
        for file in sorted(list(set(st.session_state.uploaded_file_names))):
            col_file, col_btn = st.columns([3.8, 1.7])
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
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.uploaded_file_names = []
        st.success("All temporary storage cleared!")
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
            try:
                reterived_docs = retriever.invoke(user_input)
                context = "\n\n".join(doc.page_content for doc in reterived_docs)
                
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
                
                response_text = rag_chain.invoke({
                    "context": context,
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                
                source_files = list(set([
                    doc.metadata.get("source", "unknown") for doc in reterived_docs
                ]))
                if source_files and "This information is not available" not in response_text:
                    response_text += f"\n\n📎 **Source:** {', '.join(source_files)}"
                    
                st.session_state.chat_history.extend([
                    ("human", user_input),
                    ("ai", response_text),
                ])
            except Exception as e:
                response_text = f"⚠️ Error generating response: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()
