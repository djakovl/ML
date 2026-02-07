from trl import SFTTrainer, SFTConfig
import torch
from config import *
from model_setup import model, tokenizer
from data_prep import train_dataset, eval_dataset

print("🎓 Настройка обучения...")

training_args = SFTConfig(
    output_dir=OUTPUTDIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCHSIZE,
    per_device_eval_batch_size=BATCHSIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    fp16=False,
    bf16=True,
    
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("\n🚀 Начинаю обучение...\n")

torch.cuda.empty_cache()
trainer.train()

print("\n✅ Обучение завершено!")

trainer.save_model(OUTPUTDIR + "/final")
tokenizer.save_pretrained(OUTPUTDIR + "/final")

print(f"💾 Модель сохранена в {OUTPUTDIR}/final")
