import json
import os
import random
from transformers import pipeline
from friend_agent import build_prompt

# 📌 Настройка модели (можно заменить на свою fine-tuned модель)
MODEL_NAME = "gpt2"  # Заменить на свою, например: "mistralai/Mistral-7B-Instruct-v0.2"
generator = pipeline("text-generation", model=MODEL_NAME)

# 🔢 Параметры генерации
GENERATION_KWARGS = {
    "max_new_tokens": 80,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.9,
    "repetition_penalty": 1.1
}

# 💬 Настройки фраз ребёнка
ages = [7, 9, 11, 13, 15]
themes = ["друзья", "учёба", "родители", "самооценка", "будущее", "одиночество"]
expression_styles = {
    "сомнение": "в стиле сомнения",
    "вопрос": "в форме вопроса",
    "тревога": "в стиле тревоги",
    "жалоба": "в форме жалобы",
    "размышление": "в стиле размышления"
}

def generate_child_phrase():
    age = random.choice(ages)
    theme = random.choice(themes)
    style_key = random.choice(list(expression_styles.keys()))
    style = expression_styles[style_key]

    prompt = (
        f"Ты ребёнок {age} лет. Напиши одну короткую фразу (1–2 предложения), "
        f"в которой ты делишься своими переживаниями на тему \"{theme}\" {style}."
    )

    try:
        response = generator(prompt, **GENERATION_KWARGS)[0]["generated_text"]
        phrase = response[len(prompt):].strip().split("\n")[0]

        return {
            "text": phrase,
            "age": age,
            "theme": theme,
            "style": style_key
        }

    except Exception as e:
        print(f"[Ошибка генерации фразы ребёнка]: {e}")
        return None

def generate_dialogs(n=30, output_path='data/child_dialogs.jsonl'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(n):
            child = generate_child_phrase()
            if not child: continue

            prompt = build_prompt(child['text'])
            try:
                response = generator(prompt, **GENERATION_KWARGS)[0]['generated_text']
                reply = response[len(prompt):].strip().split("\n")[0]

                f.write(json.dumps({
                    "child_input": child['text'],
                    "child_meta": {
                        "age": child['age'],
                        "theme": child['theme'],
                        "style": child['style']
                    },
                    "friend_reply": reply
                }, ensure_ascii=False) + '\n')

                print(f"[{i+1}/{n}] ✅")

            except Exception as e:
                print(f"[Ошибка генерации ответа]: {e}")

# Пример запуска
if __name__ == "__main__":
    generate_dialogs(n=50)
