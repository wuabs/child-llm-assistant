import os
import json
import random
import openai
from friend_agent import build_prompt

# 🔑 Установи свой ключ через переменные окружения или напрямую
openai.api_key = os.getenv("OPENAI_API_KEY") or "sk-..."  # вставь сюда свой ключ

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

def chat(prompt, system="Ты — ребёнок, говорящий о своих переживаниях.", max_tokens=100):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=max_tokens,
        top_p=0.95,
        frequency_penalty=0.1,
        presence_penalty=0.1
    )
    return response.choices[0].message["content"].strip()

def generate_child_phrase():
    age = random.choice(ages)
    theme = random.choice(themes)
    style_key = random.choice(list(expression_styles.keys()))
    style = expression_styles[style_key]

    prompt = (
        f"Ты ребёнок {age} лет. Напиши одну короткую фразу (1–2 предложения), "
        f"в которой ты делишься своими переживаниями на тему «{theme}» {style}."
    )

    try:
        phrase = chat(prompt)
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
                reply = chat(prompt, system="Ты — заботливый, поддерживающий друг, который отвечает на русском ребёнку.")
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
                print(f"[Ошибка генерации ответа друга]: {e}")

# Пример запуска
if __name__ == "__main__":
    generate_dialogs(n=10)
