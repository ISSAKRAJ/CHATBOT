import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Together
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
PDF_PATH = "Arogya Sanjeevani Policy.pdf"

# Cache and load vectorstore
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")
    
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

# Load vectorstore once
vectordb = load_vectorstore()

# Setup Together AI LLM
llm = Together(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.3,
    max_tokens=300,
    together_api_key=TOGETHER_API_KEY
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_type="similarity", k=3),
    return_source_documents=False
)

# Streamlit UI
st.set_page_config(page_title="Policy PDF Chatbot", layout="centered")
st.title("📄 Arogya Sanjeevani Policy Chatbot")
st.markdown("Ask any question related to the policy document.")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field
query = st.text_input("🔍 Enter your question:")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                response = qa_chain.run(query).strip()
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
            
            st.session_state.chat_history.append({
                "query": query,
                "response": response
            })

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 📝 Chat History")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['query']}")
        st.markdown(f"**Bot:** {chat['response']}")
        st.markdown("---")
