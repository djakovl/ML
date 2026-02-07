import pandas as pd
from datasets import Dataset
from config import *
from model_setup import tokenizer


df = pd.read_csv(CSVPATH)
df_valid = df[df['text'].notna()].copy()

def format_and_tokenize_row(row):
    text = row['text']
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    text_clean = '\n'.join(lines)
    formatted = f"<START>{text_clean}<END>"
    
    tokenized = tokenizer(
        formatted,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None
    )
    

    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


train_df = df_valid.sample(frac=0.9, random_state=42)
eval_df = df_valid.drop(train_df.index)

print(f"Train: {len(train_df)}, Eval: {len(eval_df)}")


train_dataset = Dataset.from_pandas(train_df).map(
    format_and_tokenize_row,
    remove_columns=train_df.columns.tolist()
)

eval_dataset = Dataset.from_pandas(eval_df).map(
    format_and_tokenize_row,
    remove_columns=eval_df.columns.tolist()
)

print(f"Train dataset: {len(train_dataset[0]['input_ids'])}, Eval: {len(eval_dataset[0]['input_ids'])}")
print("✅ Данные подготовлены")
