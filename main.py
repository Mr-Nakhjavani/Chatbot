import os
import logging
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory

# Silence noisy logs
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup Streamlit page
st.set_page_config(page_title="💬 My AI Chatbot", page_icon="🤖")
st.title("🤖 My LangChain Chatbot")

# Load and process data only once
@st.cache_resource
def load_chain():
    # Load document
    loader = TextLoader("knowledge.txt")
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Embeddings & Vector Store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    persist_directory = "./chroma_db"

    if not os.path.exists(persist_directory):
        os.mkdir(persist_directory)

    db = Chroma.from_documents(
        texts, embeddings, persist_directory=persist_directory
    )

    retriever = db.as_retriever()

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    # LLM
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    # Conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    return chain

# Load the chain
chain = load_chain()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("🧑‍💻"):
            st.write(message["content"])
    else:
        with st.chat_message("🤖"):
            st.write(message["content"])

# User input
prompt = st.chat_input("Ask me anything...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("🧑‍💻"):
        st.write(prompt)

    # Run chain
    result = chain({"question": prompt})
    answer = result["answer"].strip()

    # Handle empty answers
    if "I don't know" in answer or answer == "":
        answer = "Hmm, I’m not sure about that. I don't have enough information."

    # Add bot message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("🤖"):
        st.write(answer)
