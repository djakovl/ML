from imports_and_config import *
import pickle

train_df = pd.read_pickle('train_df.pkl')
eval_df = pd.read_pickle('eval_df.pkl')
tokenizer = AutoTokenizer.from_pretrained('tokenizer_dir')

def format_chat(example):
    messages = [
        {"role": "system", "content": SYSTEMPROMPT},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['output']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
train_dataset = train_dataset.map(format_chat)
eval_dataset = eval_dataset.map(format_chat)

# Токенизация (с EOS, truncate)
def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, max_length=256)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_dataset.save_to_disk('train_dataset')
eval_dataset.save_to_disk('eval_dataset')

if __name__ == "__main__":
    print("Датасеты подготовлены.")
