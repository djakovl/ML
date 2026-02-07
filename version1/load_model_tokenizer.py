from imports_and_config import *

bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE4BIT,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODELNAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODELNAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORAR,
    lora_alpha=LORAALPHA,
    target_modules=LORATARGETMODULES,
    lora_dropout=LORADROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("Модель и LoRA готовы!")

torch.save(model.state_dict(), 'base_model.pt')
tokenizer.save_pretrained('tokenizer_dir')

if __name__ == "__main__":
    print("Модель загружена.")
