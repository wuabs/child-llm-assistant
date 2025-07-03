from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

def save_vectorstore(vectorstore, path='vectorstore/'):
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)

def load_vectorstore(path='vectorstore/'):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local(path, embeddings)
