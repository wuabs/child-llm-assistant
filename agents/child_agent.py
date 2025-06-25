import random
import json
import os
from transformers import pipeline

# Загружаем генератор (можно заменить на свою fine-tuned модель)
generator = pipeline("text-generation", model="gpt2")  # Заменить на свою модель

# Категории
ages = [7, 9, 11, 13, 15]
themes = ["друзья", "учёба", "родители", "самооценка", "будущее", "одиночество"]
expression_styles = {
    "сомнение": "в стиле сомнения",
    "вопрос": "в форме вопроса",
    "тревога": "в стиле тревоги",
    "жалоба": "в форме жалобы",
    "размышление": "в стиле размышления"
}

def generate_child_input():
    age = random.choice(ages)
    theme = random.choice(themes)
    style_key = random.choice(list(expression_styles.keys()))
    style = expression_styles[style_key]

    prompt = (
        f"Ты ребёнок {age} лет. Напиши одну короткую фразу (1–2 предложения), "
        f"в которой ты делишься своими переживаниями на тему \"{theme}\" {style}."
    )

    output = generator(prompt, max_new_tokens=40, do_sample=True, top_k=50, temperature=0.9)[0]["generated_text"]
    child_phrase = output[len(prompt):].strip().split("\n")[0]

    return {
        "age": age,
        "theme": theme,
        "style": style_key,
        "text": child_phrase
    }

def generate_child_inputs(n=20, output_path='data/child_inputs.jsonl'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for _ in range(n):
            item = generate_child_input()
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Пример вызова
if __name__ == "__main__":
    generate_child_inputs(n=50)
