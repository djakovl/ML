from config import *
from model_setup import model, tokenizer  # Модель уже загружена с LoRA

def generate_chastushka(model, tokenizer, theme=None):
   
    prompt = f"<START>{theme}" if theme else "<START>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            repetition_penalty=GEN_REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<START>" in generated and "<END>" in generated:
        chastushka = generated.split("<START>")[1].split("<END>")[0].strip()
    else:
        chastushka = generated.replace("<START>", "").replace("<END>", "").strip()
    
    return chastushka

print("🎵 Примеры генерации:\n")

for i in range(5):
    print("=" * 50)
    print(f"Частушка #{i+1}")
    print("=" * 50)
    print(generate_chastushka(model, tokenizer))
    print()
