import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# Set your Groq API key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Set up the Groq LLM
llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0.7)

# Define your custom data source URL
DATA_URL = "https://en.wikipedia.org/wiki/AlexNet"

# Function to load and process data from the web page
@st.cache_resource
def load_data(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# Set up the retrieval system
@st.cache_resource
def setup_retrieval_system():
    texts = load_data(DATA_URL)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Create the conversational chain
@st.cache_resource
def create_qa_chain(_vectorstore):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        return_source_documents=True
    )

# Streamlit UI setup
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("AI Chatbot powered by Groq and LangChain")

# Sidebar
with st.sidebar:
    st.title("About")
    st.info("This chatbot uses Groq's Mixtral model and LangChain for document retrieval and Q&A.")
    st.markdown(f"[Data source]({DATA_URL})")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Setup retrieval system and QA chain
vectorstore = setup_retrieval_system()
qa_chain = create_qa_chain(vectorstore)

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("You:"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        result = qa_chain({"question": prompt, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history]})
        full_response = result['answer']
        message_placeholder.markdown(full_response)
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
