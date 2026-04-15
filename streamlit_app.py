import streamlit as st
import os
import tempfile
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline

# =====================================================================
# 1. Pipeline Classes and Helper Functions
# =====================================================================

def custom_generator(prompt, model, tokenizer, max_new_tokens, temperature):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model and input to the right device
    model.to(device)
    input_ids = input_ids.to(device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

class MinimalPipeline:
    def __init__(self, model, tokenizer, max_new_tokens, temperature):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.task = "text2text-generation"

    def __call__(self, text_inputs, **kwargs):
        if isinstance(text_inputs, list):
            results = []
            for prompt in text_inputs:
                generated_text = custom_generator(prompt, self.model, self.tokenizer, self.max_new_tokens, self.temperature)
                results.append([{'generated_text': generated_text}])
            return results
        else:
            generated_text = custom_generator(text_inputs, self.model, self.tokenizer, self.max_new_tokens, self.temperature)
            return [{'generated_text': generated_text}]

# =====================================================================
# 2. Cached Loaders (Only loads ONCE to keep UI fast)
# =====================================================================

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    min_pipe = MinimalPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.5
    )
    llm = HuggingFacePipeline(pipeline=min_pipe)
    return llm

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================================
# 3. Streamlit App & UI Theme Setup
# =====================================================================

st.set_page_config(page_title="AI Document QA", page_icon="📄", layout="wide")

# Custom CSS for glassmorphism / attractive modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Main Background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #e0e7ff, #f3e8ff, #dbeafe, #ede9fe);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stChatFloatingInputContainer {
        border-radius: 25px;
        box-shadow: 0 10px 40px 0 rgba(31, 38, 135, 0.1);
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .stChatFloatingInputContainer:hover {
        box-shadow: 0 15px 50px 0 rgba(31, 38, 135, 0.15);
        transform: translateY(-2px);
    }
    
    h1 {
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0rem;
    }

    .subtitle {
        color: #4B5563;
        font-weight: 400;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.4);
    }
    
    /* File uploader styles */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.5);
        border: 2px dashed #8B5CF6;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(255, 255, 255, 0.9);
        border-color: #6D28D9;
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.2);
    }
    
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
        border: 1px solid rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 AI Document Intelligence")
st.markdown("<p class='subtitle'>Upload your documents, wait for processing, and ask questions!</p>", unsafe_allow_html=True)

# State initialization
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    # Adding a welcoming assistant message
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! Upload a PDF on the left and ask me questions about it."}]

# =====================================================================
# 4. Sidebar Layout
# =====================================================================
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file...", type=["pdf"])

    if uploaded_file:
        st.success("File uploaded successfully! Processing...")

# =====================================================================
# 5. Core RAG Logic
# =====================================================================
if uploaded_file is not None and st.session_state.qa_chain is None:
    with st.spinner("🤖 Loading AI Models and Indexing Document... (May take a moment)"):
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load document
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()

            # Text Chunking
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            # Create Vector Store
            embeddings = load_embeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # Load LLM
            llm = load_llm()

            # Create QA Chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            
            st.success("✅ Document processed and ready for questions!")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# =====================================================================
# 6. Chat Interface
# =====================================================================

# Render all existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your document..."):
    # Enforce uploading a doc first
    if st.session_state.qa_chain is None:
        st.warning("Please upload and process a document in the sidebar first!")
    else:
        # Add and render user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and render assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document for an answer..."):
                result = st.session_state.qa_chain({"query": prompt})
                answer = result.get("result", "I couldn't find an answer.")
                
                # Display Answer
                st.markdown(answer)
                
                # Expandable block for details and sources
                with st.expander("📚 View Document Source Context"):
                    for i, doc in enumerate(result.get("source_documents", [])):
                        st.markdown(f"**Source Chunk {i+1}:**")
                        st.caption(f"_{doc.page_content}_")
                
                # Save answer to state
                st.session_state.messages.append({"role": "assistant", "content": answer})
