from imports_and_config import *

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(MODELNAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(FINALMODELDIR)
model = PeftModel.from_pretrained(base_model, FINALMODELDIR)

def generate_chastushka(prompt):
    messages = [{"role": "system", "content": SYSTEMPROMPT}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=GENMAXTOKENS,
        min_new_tokens=GENMINTOKENS,
        temperature=GENTEMPERATURE,
        do_sample=True,
        top_p=GENTOPP,
        top_k=GENTOPK,
        repetition_penalty=GENREPETITIONPENALTY,
        no_repeat_ngram_size=GENNOREPEATNGRAM,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return result

if __name__ == "__main__":
    for i in range(5):
        prompt = PROMPTS[i % len(PROMPTS)]
        chastushka = generate_chastushka(prompt)
        print(f"{i+1}. {prompt}")
        print(chastushka)
        print("-" * 50)
