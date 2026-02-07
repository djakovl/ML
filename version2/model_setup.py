from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from config import *


tokenizer = AutoTokenizer.from_pretrained(MODELNAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


special_tokens = {"additional_special_tokens": ["<START>", "<END>"]}
tokenizer.add_special_tokens(special_tokens)


model = AutoModelForCausalLM.from_pretrained(
    MODELNAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=LORAR,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ Модель и LoRA готовы")
print(f"Обучаемых параметров: {model.num_parameters(only_trainable=True):,}")
