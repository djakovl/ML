from imports_and_config import *
from inference import model, tokenizer, generate_chastushka  # Reuse

generated_texts = []
for i in range(75):
    prompt = random.choice(PROMPTS)
    chastushka = generate_chastushka(prompt)
    generated_texts.append(chastushka)
    if i % 10 == 0:
        print(f"{i}/75")


def calculate_distinct_ngrams(texts, n):
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0

distinct1 = calculate_distinct_ngrams(generated_texts, 1)
distinct2 = calculate_distinct_ngrams(generated_texts, 2)
distinct3 = calculate_distinct_ngrams(generated_texts, 3)


def calculate_perplexity(texts, model, tokenizer):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

perplexity = calculate_perplexity(generated_texts, model, tokenizer)


emb_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
def calculate_coherence(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 2: return 0.0
    embeddings = emb_model.encode(lines)
    sims = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        sims.append(sim)
    return np.mean(sims)

coherence_scores = [calculate_coherence(t) for t in generated_texts]
avg_coherence = np.mean(coherence_scores)

print("="*60)
print("Distinct N-grams")
print(f"Distinct-1: {distinct1:.4f}")
print(f"Distinct-2: {distinct2:.4f}")
print(f"Distinct-3: {distinct3:.4f}")
print(f"PERPLEXITY: {perplexity:.2f}")
print()
print(f"Coherence: {avg_coherence:.4f} (1.0=perfect)")

with open("metrics_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Distinct-1: {distinct1:.4f}\n")
    f.write(f"Distinct-2: {distinct2:.4f}\n")
    f.write(f"Distinct-3: {distinct3:.4f}\n")
    f.write(f"Perplexity: {perplexity:.2f}\n")
    f.write(f"Coherence: {avg_coherence:.4f}\n")

if __name__ == "__main__":
    print("Оценка завершена.")
