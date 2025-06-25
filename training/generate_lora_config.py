lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

with open("lora_config.json", "w", encoding="utf-8") as f:
    json.dump(lora_config.__dict__, f, indent=4)
