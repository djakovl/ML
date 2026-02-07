from imports_and_config import *
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(MODELNAME, device_map="auto")  # Загрузка base
model = PeftModel.from_pretrained(model, OUTPUTDIR)  # Или из pt

training_args = SFTConfig(
    output_dir=OUTPUTDIR,
    num_train_epochs=NUMEPOCHS,
    per_device_train_batch_size=BATCHSIZE,
    gradient_accumulation_steps=GRADIENTACCUMULATION,
    learning_rate=LEARNINGRATE,
    fp16=False,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    warmup_ratio=WARMUPRATIO,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=load_from_disk('train_dataset'),
    eval_dataset=load_from_disk('eval_dataset'),
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model(FINALMODELDIR)
tokenizer.save_pretrained(FINALMODELDIR)
print(f"Модель сохранена в {FINALMODELDIR}")

if __name__ == "__main__":
    print("Обучение завершено.")
