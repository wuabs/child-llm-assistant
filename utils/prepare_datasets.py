import os

def prepare_lm_dataset(input_dir='data/cleaned_texts', output_path='data/lm_dataset.txt'):
    os.makedirs('data', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for fname in os.listdir(input_dir):
            if fname.endswith('.txt'):
                with open(os.path.join(input_dir, fname), encoding='utf-8') as f:
                    text = f.read().strip()
                    out_file.write(text + '\n\n')

if __name__ == "__main__":
    prepare_lm_dataset()
