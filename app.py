import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field
from typing import List, TypedDict, Annotated, Sequence
import tempfile
import time
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="RAG AI Pro", layout="wide")

# Custom CSS for Sleek Modern "Pro" Aesthetic
st.markdown("""
<style>
    /* Base Pro Theme */
    .stApp {
        background-color: #0d0d0d;
        color: #efefef;
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
    }
    
    .stChatFloatingInputContainer {
        max-width: 100% !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
        background: transparent !important;
    }
    
    /* Sidebar: Minimalist Slate */
    section[data-testid="stSidebar"] {
        background-color: #121212 !important;
        border-right: 1px solid #222222;
        width: 300px !important;
    }
    
    /* Chat Messages: Clean & High Contrast */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 2rem 5% !important;
        border-bottom: 1px solid #1a1a1a !important;
        max-width: 100% !important;
    }
    
    /* Remove Avatars */
    [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }
    
    [data-testid="stChatMessageContent"] {
        font-size: 1rem;
        line-height: 1.6;
        color: #f0f0f0;
    }
    
    /* Thinking / Tool Logs: Compact Pro */
    .thinking-chip {
        display: inline-flex;
        align-items: center;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 0.8rem;
        color: #999;
        margin: 8px 0;
        gap: 8px;
    }
    
    /* Citations: Sophisticated Emerald */
    .citation-tag {
        color: #10b981;
        font-weight: 700;
        font-size: 0.85rem;
        cursor: pointer;
        margin-left: 2px;
    }
    
    /* Admin/Sidebar Controls: Tight & Organized */
    .admin-header {
        color: #666;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 20px 0 10px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #222;
    }
    
    /* Expander Styling */
    .stExpander {
        background: #141414 !important;
        border: 1px solid #222 !important;
        border-radius: 6px !important;
    }
    
    /* Text Inputs */
    .stTextInput>div>div>input {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        color: #fff !important;
    }
    
    /* Hide distractions */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- State Definition ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    steps: List[str]

# --- Models ---
@st.cache_resource
def get_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", temperature=0)
    return embeddings, llm

# --- Helper for Activity Logs ---
def log_activity(msg, placeholder):
    placeholder.markdown(f'<div class="thinking-chip"><span>{msg}</span></div>', unsafe_allow_html=True)

# --- Agent Nodes ---
def retrieve(state):
    log_activity("Retrieving context...", st.session_state.log_box)
    documents = st.session_state.vectors.similarity_search(state["question"], k=5)
    return {"documents": [d.page_content for d in documents], "steps": state.get("steps", []) + ["retrieve"]}

def grade_documents(state):
    log_activity("Scoring relevance...", st.session_state.log_box)
    question = state["question"]
    documents = state["documents"]
    _, llm = get_models()
    
    system = "Decide if a document is relevant to a question. Respond ONLY with 'yes' or 'no'."
    grade_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Doc: {document}\nQ: {question}")])
    retrieval_grader = grade_prompt | llm | StrOutputParser()

    filtered = []
    for d in documents:
        res = retrieval_grader.invoke({"question": question, "document": d})
        if "yes" in res.lower():
            filtered.append(d)
    
    return {"documents": filtered, "steps": state["steps"] + ["grade"]}

def generate(state):
    log_activity("Synthesizing answer...", st.session_state.log_box)
    _, llm = get_models()
    
    prompt = ChatPromptTemplate.from_template("""
    You are a technical assistant. Answer based ONLY on the context.
    Do NOT use any emojis in your response.
    
    Format: Use clear paragraphs. Cite sources using [1], [2], etc.
    
    Context:
    {context}
    
    Question: {question}
    """)
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": "\n\n".join(state["documents"]), "question": state["question"]})
    return {"generation": generation, "steps": state["steps"] + ["generate"]}

def transform_query(state):
    log_activity("Rewriting query...", st.session_state.log_box)
    _, llm = get_models()
    system = "Rewrite the question for better retrieval. Return only text."
    re_write_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Input: {question}")])
    better_question = (re_write_prompt | llm | StrOutputParser()).invoke({"question": state["question"]})
    return {"question": better_question, "steps": state["steps"] + ["transform_query"]}

def decide_to_generate(state):
    # Limit number of transformation attempts to 3 to prevent GraphRecursionError
    transformation_count = state["steps"].count("transform_query")
    if state["documents"]:
        return "generate"
    elif transformation_count < 3:
        return "transform_query"
    else:
        # Fallback to generate even if no docs, to break the loop
        return "generate"

@st.cache_resource
def build_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)
    return workflow.compile()

# --- Docs ---
@st.cache_resource
def get_vectorstore(_files):
    embeddings, _ = get_models()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    for f in _files:
        ext = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        try:
            if ext == ".pdf": loader = PyPDFLoader(path)
            elif ext == ".txt": loader = TextLoader(path)
            elif ext == ".docx": loader = Docx2txtLoader(path)
            docs.extend(loader.load())
        finally: os.unlink(path)
    return FAISS.from_documents(text_splitter.split_documents(docs), embeddings)

def process_docs(files):
    st.session_state.vectors = get_vectorstore(files)

# --- UI Init ---
if "vectors" not in st.session_state: st.session_state.vectors = None
if "history" not in st.session_state: st.session_state.history = []

# --- Sidebar Admin ---
with st.sidebar:
    st.markdown('<h1 style="font-size: 1.2rem; margin-bottom: 0;">PRO CONSOLE</h1>', unsafe_allow_html=True)
    st.markdown('<div style="color: #666; font-size: 0.7rem; margin-bottom: 20px;">Agentic RAG Engine v2.0</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="admin-header">SETTINGS</div>', unsafe_allow_html=True)
    key = st.text_input("Groq Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if key: os.environ["GROQ_API_KEY"] = key
    
    st.markdown('<div class="admin-header">KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Intel", type=["pdf", "txt", "docx"], accept_multiple_files=True, label_visibility="collapsed")
    if st.button("REBUILD INDEX", use_container_width=True) and uploaded:
        with st.spinner("Indexing..."):
            process_docs(uploaded)
            st.success("Indexed Successfully")
    
    st.markdown('<div class="admin-header">SESSION</div>', unsafe_allow_html=True)
    if st.button("CLEAR CONVERSATION", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --- Chat Interface ---
st.markdown('<div style="max-width: 100%; padding-left: 5rem; padding-right: 5rem; padding-top: 2rem;">', unsafe_allow_html=True)

# Display History
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Reference Context"):
                for s_i, source in enumerate(msg["sources"]):
                    st.markdown(f"**[{s_i+1}]** {source[:300]}...")

# Chat Input
if prompt := st.chat_input("Input Query..."):
    if not st.session_state.vectors:
        st.info("System Ready. Please index a knowledge source via the sidebar.")
    else:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            st.session_state.log_box = st.empty()
            with st.status("Engine Reasoning...", expanded=False) as status:
                result = build_workflow().invoke(
                    {"question": prompt, "steps": []},
                    config={"recursion_limit": 50}
                )
                status.update(label="Response Synthesized", state="complete")
            
            st.markdown(result["generation"])
            
            if result.get("documents"):
                with st.expander("Sources Cited"):
                    for d_i, doc in enumerate(result["documents"]):
                        st.markdown(f"**[{d_i+1}]** {doc[:400]}...")
            
            st.session_state.history.append({
                "role": "assistant", 
                "content": result["generation"],
                "sources": result.get("documents", [])
            })

st.markdown('</div>', unsafe_allow_html=True)
