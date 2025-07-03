import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from rag_system.rag_chain import get_rag_chain

# Путь к модели
MODEL_PATH = "models/lora-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = "vectorstore/index"

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# HuggingFace LLM → LangChain wrapper
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
llm = HuggingFacePipeline(pipeline=pipe)

# Подключение RAG-цепочки
rag_chain = get_rag_chain(llm, VECTORSTORE_PATH)

# Системный промпт
SYSTEM_PROMPT = (
    "Ты — добрый и понимающий друг, который хорошо разбирается в детской психологии. "
    "Если ребёнок делится чувствами — поддержи. Если просит объяснить учебу — помоги понятным примером."
)

# Ключевые слова для определения запроса на учёбу
study_keywords = [
    "помоги", "объясни", "задача", "что такое", "как найти", "пример",
    "математика", "история", "наука", "физика", "биология", "глагол", "домашка"
]

def is_study_query(text):
    return any(kw in text.lower() for kw in study_keywords)

# Генерация ответа
def generate_reply(message, history):
    # Собираем диалог
    full_dialog = SYSTEM_PROMPT + "\n"
    for user, bot in history:
        full_dialog += f"Ребёнок: {user}\nДруг: {bot}\n"
    full_dialog += f"Ребёнок: {message}\nДруг:"

    # Учебный запрос → RAG
    if is_study_query(message):
        return rag_chain.run(message).strip()

    # Эмоциональный → обычный диалог через LLM
    inputs = tokenizer(full_dialog, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.9,
        top_p=0.95,
        do_sample=True
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded[len(full_dialog):].strip().split("\n")[0]
    return reply

# Gradio чат
chatbot = gr.ChatInterface(
    fn=generate_reply,
    title="🤖 Помощник-друг",
    description="Говори о своих чувствах или проси помощи с учебой — я рядом ✨"
)

if __name__ == "__main__":
    chatbot.launch()
