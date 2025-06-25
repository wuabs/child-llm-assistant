from rag_system.document_loader import load_documents_from_folder
from rag_system.splitter import split_documents
from rag_system.vectorstore import save_vectorstore

# === Параметры
SOURCE_DIR = "data/textbooks"
OUTPUT_DIR = "vectorstore/index"

def build_index():
    print("📚 Загрузка документов...")
    docs = load_documents_from_folder(SOURCE_DIR)

    print("✂️ Разбиение на фрагменты...")
    chunks = split_documents(docs)

    print(f"🔢 Всего фрагментов: {len(chunks)}")

    print("💾 Сохранение векторной базы...")
    save_vectorstore(chunks, OUTPUT_DIR)

    print(f"✅ Индекс готов и сохранён в: {OUTPUT_DIR}")

if __name__ == "__main__":
    build_index()
