from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from rag_system.vectorstore import load_vectorstore

def get_rag_chain(llm: HuggingFacePipeline, vectorstore_path="vectorstore/index"):
    """Создаёт RAG-цепочку на основе LLM и локальной векторной базы"""
    vectorstore = load_vectorstore(vectorstore_path)
    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain
