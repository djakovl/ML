import torch


CSVPATH = "chastushki750.csv"
MODELNAME = "Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct"
OUTPUTDIR = "./chastushki-model"

NUM_EPOCHS = 5
BATCHSIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128


LORAR = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_MODULES = ["q_proj", "v_proj"]


GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.85
GEN_TOP_K = 30
GEN_REPETITION_PENALTY = 1.3


bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4"
}

print("✅ Конфигурация загружена")
