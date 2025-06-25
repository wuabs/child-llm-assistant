import os
import re

def clean_text(text):
    # Пример простой очистки текста от спецсимволов и лишних пробелов
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"[^А-Яа-я0-9.,!?\\-\\n ]", "", text)
    return text.strip()

def clean_all_texts(input_dir="data/texts", output_dir="data/cleaned_texts"):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(input_dir, fname), encoding="utf-8") as f:
                raw = f.read()
            cleaned = clean_text(raw)
            with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
                f.write(cleaned)

if __name__ == "__main__":
    clean_all_texts()
