import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.llms import Ollama   # <-- Fix here

load_dotenv()  # loads variables from .env into os.environ

def get():
    # Initialize Ollama with Mistral
    llm = Ollama(model="mistral")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ At least one doc needed
    sample_docs = ["This is a test document to initialize FAISS."]
    vector_store = FAISS.from_texts(sample_docs, embedding=embeddings)

    # Test the model
    response = llm.invoke("Hello, can you summarize LangChain in 2 lines?")
    print("LLM Response:", response)

    return llm, embeddings, vector_store
