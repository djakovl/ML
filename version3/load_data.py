from imports_and_config import *

def calculate_rhyme_quality(row):
    endings = re.findall(r'\w{3,}$', row['text'].lower())
    counter = Counter(endings)
    most_common = counter.most_common(10)
    return len(most_common) / len(endings) if endings else 0


df = pd.read_csv(CSVPATH)
df['rhyme_words'] = df.apply(lambda x: eval(x['rhyme_words']) if isinstance(x['rhyme_words'], str) else [], axis=1)
df['rhyme_quality'] = df['rhyme_words'].apply(calculate_rhyme_quality)
df = df.sort_values('rhyme_quality', ascending=False)
df_good = df[df['rhyme_quality'] >= 0.5].copy()
print(len(df_good))

train_df = df_good.sample(frac=0.9, random_state=RANDOMSEED)
eval_df = df_good.drop(train_df.index)

train_df.to_pickle('train_df.pkl')
eval_df.to_pickle('eval_df.pkl')
print(f"Train: {len(train_df)}, Eval: {len(eval_df)}")

if __name__ == "__main__":
    print("Данные загружены и разделены.")
