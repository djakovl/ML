import os
import torch
import pandas as pd
from datasets import Dataset
import random
import numpy as np
import re
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import login


login(token="hftDLvCOdabqjtRCBbSdokSRWojMAkUxwsgq")  
CSVPATH = "chastushki500.csv"
MODELNAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUTDIR = "./qwen-500samples"
FINALMODELDIR = "./qwen-500-chastushki-final"
RANDOMSEED = 42
NUMEPOCHS = 4
BATCHSIZE = 2
GRADIENTACCUMULATION = 4
LEARNINGRATE = 5e-5
WARMUPRATIO = 0.1
LORAR = 32
LORAALPHA = 64
LORADROPOUT = 0.1
LORATARGETMODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
USE4BIT = True
GENMAXTOKENS = 60
GENMINTOKENS = 20
GENTEMPERATURE = 0.7
GENTOPP = 0.9
GENTOPK = 40
GENREPETITIONPENALTY = 1.2
GENNOREPEATNGRAM = 3
SYSTEMPROMPT = "Ты мастер русских частушек. Сделай короткую смешную частушку на заданную тему."
PROMPTS = ["о коте", "о водке", "о тракторе"]  # Пример из notebook

print("="*60)
print("QWEN 2.5-1.5B")
print("="*60)
print(f"MODELNAME: {MODELNAME}")
print(f"CSVPATH: {CSVPATH}")
print("Train/Val: 450/50")
print(f"NUMEPOCHS: {NUMEPOCHS}")
print(f"batch: {BATCHSIZE} * {GRADIENTACCUMULATION}")
print(f"LoRA r={LORAR}, modules={len(LORATARGETMODULES)}")
print(f"4-bit: {USE4BIT}")
print("="*60)

if __name__ == "__main__":
    print("Конфиг загружен.")
