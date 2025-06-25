import os
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

def load_documents_from_folder(folder_path):
    """Загружает все .txt, .pdf, .docx из папки"""
    documents = []
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if fname.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif fname.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif fname.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            print(f"⛔ Пропущен неподдерживаемый файл: {fname}")
            continue
        docs = loader.load()
        documents.extend(docs)
    return documents
