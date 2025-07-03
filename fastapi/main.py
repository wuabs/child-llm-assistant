from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Tuple
from langchain.llms import HuggingFacePipeline
from rag_system.rag_chain import get_rag_chain

# Конфигурация
MODEL_PATH = "models/lora-instruct"
VECTORSTORE_PATH = "vectorstore/index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Загрузка модели
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Обёртка для LangChain
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
llm = HuggingFacePipeline(pipeline=pipe)

# Загрузка RAG-цепочки
rag_chain = get_rag_chain(llm, VECTORSTORE_PATH)

# Ключевые слова для RAG
study_keywords = [
    "помоги", "объясни", "задача", "что такое", "как найти", "пример",
    "математика", "история", "наука", "физика", "биология", "глагол", "домашка"
]

def is_study_query(text: str) -> bool:
    return any(kw in text.lower() for kw in study_keywords)

# Системный промпт
SYSTEM_PROMPT = (
    "Ты — добрый и понимающий друг, который хорошо разбирается в детской психологии. "
    "Если ребёнок делится чувствами — поддержи. Если просит объяснить учебу — помоги понятным примером."
)

# FastAPI-приложение
app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]]  # Список пар (вопрос, ответ)

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Сбор диалога
    dialog = SYSTEM_PROMPT + "\n"
    for user, bot in req.history:
        dialog += f"Ребёнок: {user}\nДруг: {bot}\n"
    dialog += f"Ребёнок: {req.message}\nДруг:"

    if is_study_query(req.message):
        answer = rag_chain.run(req.message).strip()
    else:
        inputs = tokenizer(dialog, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.9,
            top_p=0.95,
            do_sample=True
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded[len(dialog):].strip().split("\n")[0]

    return ChatResponse(response=answer)
