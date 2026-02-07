import numpy as np
from sentence_transformers import SentenceTransformer
from generate import generate_chastushka, model, tokenizer

print("🔍 Оценка когерентности...")

emb_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def calculate_coherence(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 2:
        return 0.0
    
    embeddings = emb_model.encode(lines)
    sims = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        sims.append(sim)
    
    return np.mean(sims)

# Генерация 50 частушек
print("=" * 50)
generated = []
for i in range(50):
    chastushka = generate_chastushka(model, tokenizer)
    generated.append(chastushka)
    if i % 10 == 0:
        print(f"Генерация: {i+1}/50...")

# Оценка
coherence_scores = [calculate_coherence(c) for c in generated]
avg_coherence = np.mean(coherence_scores)

print(f"Средняя когерентность: {avg_coherence:.3f}")
print("Отлично (>0.60) ✓" if avg_coherence > 0.60 else "Хорошо (0.45-0.60) ⚠️" else "Нужно дообучить ❌")

print("\nПримеры:")
for i in range(3):
    print(f"{coherence_scores[i]:.3f}: {generated[i]}")
    print("-" * 40)
