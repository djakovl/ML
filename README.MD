# Генератор частушек

Проект по fine-tuning русскоязычных языковых моделей для автоматической генерации частушек с сохранением структуры, рифмы и народного юмора.

**Авторы:** Дьяков Лев, Оспищева Татьяна, Убушаева Яна  
**Университет:** АНОО ВО «Университет Сириус»  
**Курс:** Машинное обучение и глубокие нейросети  
**Дата:** Февраль 2026

---

## Содержание

1. [Описание задачи](#описание-задачи)
2. [Выбор датасета](#выбор-датасета-и-источники-данных)
3. [Выбор модели](#выбор-модели)
4. [Методы обучения](#методы-обучения)
5. [Подготовка данных](#подготовка-данных)
6. [Обучение моделей](#обучение-моделей)
7. [Метрики оценки](#метрики-оценки)
8. [Генерация частушек](#генерация-частушек)
9. [Результаты](#результаты-и-выводы)
10. [Структура проекта](#структура-репозитория)
11. [Список литературы](#список-литературы)

---

# Описание задачи

Частушка — жанр русского словесного творчества, короткая рифмованная песенка юмористического содержания. Задача проекта — обучить русскоязычную нейросеть генерировать частушки, которые будут соответствовать следующим критериям:

- **4 строки** — классическая структура частушки
- **Наличие рифмы** — текст должен читаться нараспев, рифмоваться по схемам AABB, ABAB или ABCB
- **Наличие юмора** — хотя бы какого-то

## Мотивация

Нужно поддерживать моральное состояние одногруппников, а то половина хочет отчислиться. Генерация веселых частушек — один из способов поднять настроение и одновременно изучить методы fine-tuning языковых моделей на специфичных данных.

---

# Выбор датасета и источники данных

Для обучения моделей был собран корпус частушек из различных источников. Главной проблемой оказалась нехватка открытых датасетов с качественными частушками, поэтому данные собирались вручную и с помощью AI-аугментации.

## Источники данных

Данные собирались из следующих источников:

### Сайты с частушками:

1. [Русские народные частушки](https://skazka-dubki.ru/zhizn-slavyan/russkiy-folklor/russkie-narodnye-chastushki/)
2. [Искусство звука](https://iskusstvo-zvuka.livejournal.com/302566.html)
3. [Sibmama - Частушки](https://sibmama.ru/ciastuscichi2.htm)
4. [Perplexity AI](https://www.perplexity.ai/)
5. [Popugaichik Blog](https://popugaichik.livejournal.com/4496106.html)
6. [Gallika Blog](https://gallika.livejournal.com/46550.html)
7. [Частушка PDF](https://ocktula.ru/dokumenty/dokumenty_cnt/chastushka-1.pdf)
8. [Частушки 18+](https://xn--80afteigabfepf5lc.xn--p1ai/tpost/4isscf6kp1-chastushki-18-ostorozhno-mat)

### Тексты песен:

1. [Сектор Газа - Частушки](https://genius.com/Sektor-gaza-ditties-lyrics)
2. [YouTube - Частушки](https://www.youtube.com/watch?v=8cyZf2iUMI8)
3. [CMH, MZLFF, Bratishkin - Частушки 3](https://genius.com/Cmh-mzlff-and-bratishkin-3-ditties-3-lyrics)

### AI-генерация:

Использовался Claude для аугментации датасета. Генерация проводилась с промптами, описывающими структуру и стиль частушек.

## Статистика датасетов

| Файл | Количество частушек | Пропусков | Дубликатов | AI-частушки | AI % |
|------|---------------------|-----------|------------|-------------|------|
| chastushki.csv | 160 | 0 | 0 | 0 | 0% |
| chastushki_500.csv | 500 | 0 | 0 | 340 | 68% |
| chastushki_750.csv | 746 | 0 | 0 | 586 | 78.6% |

### Пример структуры данных

Датасет представлен в формате CSV с колонками:

- **№** — номер частушки
- **Источник** — источник данных
- **Текст** — текст частушки (4 строки)
- **Строк** — количество строк
- **Схема_рифмовки** — схема рифмовки (AABB, ABCB, ABAB)
- **Рифмующиеся_слова** — список рифмующихся слов

### Разброс схем рифмовок

В собранных датасетах присутствуют частушки с разными схемами рифмовки:

| Датасет | ABCB | AABB | ABAB |
|---------|------|------|------|
| 160 | 53% | 29% | 18% |
| 500 | 85% | 10% | 5% |
| 750 | 90% | 6% | 4% |

Основная схема — **ABCB** (классическая частушка, где рифмуются 2-я и 4-я строки).


# Выбор модели

Для задачи генерации частушек были выбраны малые русскоязычные языковые модели, поддерживающие инструкции (instruct). Критерии выбора:

1. **Размер модели** — 0.5B-3B параметров (возможность обучения на обычных GPU)
2. **Поддержка русского языка** — хорошее качество генерации русского текста
3. **Открытая лицензия** — возможность коммерческого использования
4. **Поддержка fine-tuning** — совместимость с библиотеками PEFT и TRL

## Подопытные модели

| Модель | Параметров | Понимание русского | Ожидаемый результат |
|--------|------------|-------------------|---------------------|
| Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct | 1B | Хорошее | Связный текст + 4 строки |
| Vikhrmodels/Vikhr-Gemma-2B-instruct | 2B | Хорошее | Связный текст + 4 строки + возможная рифма хоть иногда + юмор |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Нормальное | Ломаный, но связанный текст + 4 строки + юмор |
| Qwen/Qwen2.5-3B-Instruct | 3B | Нормальное | В теории лучший выбор, но, возможно, модели на русских датасетах лучше |
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | Нормальное | Экспериментальная |

### Почему Vikhr?

Модели семейства **Vikhr** представляют собой адаптацию зарубежных моделей (Llama, Gemma) под русский язык. Они были дообучены на большом корпусе русскоязычных текстов, что делает их идеальными для генерации русского контента.

### Почему Qwen?

**Qwen** — это семейство моделей от Alibaba, изначально ориентированных на китайский язык, но показывающих хорошие результаты и на других языках, включая русский. Модели имеют отличное соотношение качества и размера.

## Источники по выбору моделей:

1. [Vikhr Models на Hugging Face](https://huggingface.co/Vikhrmodels)
2. [Qwen Models на Hugging Face](https://huggingface.co/Qwen)
3. [LLM Leaderboard для русского языка](https://huggingface.co/spaces/clips/russian_llm_leaderboard)

---

# Методы обучения

## LoRA (Low-Rank Adaptation)

Для эффективного fine-tuning моделей используется метод **LoRA**, который позволяет обучать только малую часть параметров модели, добавляя адаптеры с низким рангом.

### Математическое описание LoRA

Пусть **W₀ ∈ ℝ^(d×k)** — замороженные веса предобученной модели. LoRA представляет обновление весов в виде произведения двух матриц низкого ранга:

**W = W₀ + ΔW = W₀ + BA**

где:
- **B ∈ ℝ^(d×r)** — матрица «вниз»
- **A ∈ ℝ^(r×k)** — матрица «вверх»  
- **r ≪ min(d, k)** — ранг адаптации (обычно 8-64)

Таким образом, вместо обучения **d × k** параметров, мы обучаем только **r(d + k)** параметров.

### Преимущества LoRA:

- **Эффективность памяти:** обучается ~0.1% параметров
- **Быстрое обучение:** меньше вычислений на backward pass
- **Модульность:** можно сохранять несколько адаптеров для одной базовой модели
- **Качество:** сопоставимо с full fine-tuning

## 4-bit квантизация (QLoRA)

Для дальнейшего снижения требований к памяти используется **4-bit квантизация** базовой модели с помощью библиотеки bitsandbytes.
Эта часть использовалась в основном для больших моделей.

### Параметры квантизации:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

Где:
- **NF4** — специальный 4-битный формат для нормально распределённых весов
- **Double quantization** — квантизация самих квантизационных констант
- **bfloat16** — формат для вычислений (баланс точности и скорости) (основная проблема в случае если обучение не запускается)

## Конфигурация обучения

### Параметры LoRA:

```python
LORAR = 16    #чем мощнее модель, тем больше можем поставить, но из наблюдений это замедляет обучение, но немного увеличивает осознанность
LORA_ALPHA = 32
LORA_DROPOUT = 0.1    #пытаемся не переобучить
LORA_MODULES = ["q_proj", "v_proj"]    #какие слои мы обучаем (меняем в зависимости от модели)
```

### Гиперпараметры обучения:

```python
NUM_EPOCHS = 5    #чем хуже решения, тем больше ставим, замедляет достаточно сильно
BATCHSIZE = 1    #Есть видеопамять - увеличиваем, нет - уменьшаем
GRADIENT_ACCUMULATION = 16    # обратное батчсайзу, но можем в целом уменьшить, но надо учесть время
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
```

## Источники по методам обучения:

1. [LoRA: Low-Rank Adaptation of Large Language Models (arXiv)](https://arxiv.org/abs/2106.09685)
2. [QLoRA: Efficient Finetuning of Quantized LLMs (arXiv)](https://arxiv.org/abs/2305.14314)
3. [PEFT Documentation](https://huggingface.co/docs/peft/)
4. [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)

---

# Подготовка данных

Данные для обучения подготавливаются в следующем формате с использованием специальных токенов для структурирования.

## Формат входных данных

Каждая частушка обрамляется специальными токенами:

```xml
<chastushka>
Строка 1
Строка 2
Строка 3
Строка 4
</chastushka>
```

### Код подготовки данных:

```python
import pandas as pd
from datasets import Dataset

def format_and_tokenize_row(row):
    text = row['text']
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    text_clean = '\n'.join(lines)

    formatted = f"<chastushka>{text_clean}</chastushka>"

    tokenized = tokenizer(
        formatted,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None
    )

    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

df = pd.read_csv(CSVPATH)
df_valid = df[df['text'].notna()].copy()
```

## Разделение на train/validation

Данные разделяются в соотношении 90/10:

```python
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
```

## Источники по подготовке данных:

1. [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
2. [Tokenization Best Practices](https://huggingface.co/docs/transformers/preprocessing)

---

# Обучение моделей

## Процесс обучения

Обучение проводится с использованием библиотеки **TRL** (Transformer Reinforcement Learning), которая предоставляет удобный интерфейс для supervised fine-tuning.

### Настройка модели:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(MODELNAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

special_tokens = {"additional_special_tokens": ["<chastushka>", "</chastushka>"]}
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
```

### Конфигурация тренера:

```python
from trl import SFTTrainer, SFTConfig

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
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

trainer.save_model(OUTPUTDIR + "/final")
tokenizer.save_pretrained(OUTPUTDIR + "/final")
```

## Эксперименты с Vikhr-Llama-3.2-1B

| Версия | Датасет | LR | Эпох | Val Loss | Структура (4 строки) | Рифма |
|--------|---------|-----|------|----------|----------------------|-------|
| v1 | 500 | 5e-5 | 6 | 0.720 | 80% | 0% |
| v2 | 500 | 1e-4 | 4 | 0.754 | 40% | 0% |
| v3 | 500 | 3e-5 | 3 | 0.911 | 40% | 0% |
| v4 | 500 | 5e-5 | 4 | 0.756 | 60% | 0% |
| 750 | 750 | 4e-5 | 5 | 0.770 | 80% | 0% |

### Лучшая модель

Модель **v1** на 500 частушках показала лучшее соотношение loss и структуры (80% текстов имеют 4 строки). Однако рифма не генерируется ни в одной из версий.

### Метрики лучшего решения:

- Perplexity: 24.07
- Distinct-1: 0.465
- Coherence: 0.398

### Средние метрики:

- Perplexity: 27.36
- Distinct-1: 0.746
- Coherence: 0.411

### Примеры генерации (Vikhr-Llama):

**Попытка 1:**
```
Я любила мельницкого,
Он стройные дюни пилал.
А когда домохозяйка пришла?
Не хватала колешки!
```
*Проблемы:* Артефакты, бессмыслица, нет рифмы

**Попытка 2:**
```
Я работала в банке,
Банковские карты я продавал;
Но один клиент был так странный —
Не подписался на что-то!
```
*Проблемы:* Нет рифмы, зацикленность на теме банка (4 из 5 частушек про банк)

**Попытка 4:**
```
Никто не знал,
Огонь сгорел в доме.
Второй день — старший!
Говорил он так разные!
```
*Проблемы:* Полная бессмыслица

## Эксперименты с Vikhr-Gemma-2B

Результаты оказались еще хуже — модель генерировала множество артефактов в виде тегов, которые почему-то не удалялись, и иностранных символов. Модель часто переходила на английский язык.

### Примеры генерации:

```
Милый мой работал в банке,
Аденье выводил он день за днем!
Но домой возвращался не богат,
И все счетаpayers проверяла he!<|ecot_id|<|eoc_id_|></eocid_idtehende><eos>
```

## Эксперименты с Qwen2.5-1.5B

### Системный промпт (50 частушек):

```
Ты — народный поэт-частушечник. Сочиняешь задорные русские частушки.

СТРОГИЙ ФОРМАТ:
• Ровно 4 строки
• Рифма: AABB (1-2, 3-4) или ABAB (1-3, 2-4)
• 6-8 слогов в строке

СОДЕРЖАНИЕ:
• ОДНА короткая история от начала до конца
• Все 4 строки про ОДНО событие
• Логическая связь: действие → результат → шутка/поворот

СТРУКТУРА ЧАСТУШКИ:
Строка 1-2: Что случилось
Строка 3-4: Чем закончилось (неожиданно/смешно)

ЗАПРЕЩЕНЫ:
• Выдуманные слова
• Разные темы в разных строках
• Обрывки фраз без смысла
```

### Метрики (50 частушек):

- Perplexity: 27.60
- Distinct-1: 0.5441 (не очень здорово, слишком часто повторяются слова)
- Distinct-2: 0.9399 (ужас)
- Distinct-3: 0.9965
- Coherence: 0.4393 (это плохо)

### Метрики (235 частушек):

После увеличения датасета до 235 частушек результаты не изменились существенно.

### Примеры генерации:

**Относительно успешная:**
```
Я была миленькой девчонкой,
Меня любили все ребяточки.
А теперь я — взрослый человек,
Мне любят только старые деды.
```

**Неудачная:**
```
А на той стороне солнце светит,
а у нас темно, я вижу только тень.
Давай плясок, давай беготки,
Мир — это гигантская куча булочек.
```

**Абстрактная:**
```
Сочинила я частушку
Во всеуслышанье,
Придумала я весёлую
Абстрактную п**ду
```

## Эксперименты с Qwen2.5-0.5B

Модель оказалась слишком малой для задачи. Результаты показали катастрофу.

### Примеры:

```
Меня зовут Маша и я бывала в США,
а она - в России.
Я была там неделимой девочкой, с мамой и папой,
Но потом у меня появилась новая жизнь: мальчиком
```

```
Я не любила садовником,
Так как он ухаживал за детьми.
Но когда я открыла его окно,
Собственно, радовался мне все!

észert megisztettünk a hátrak
```
*(переходит на венгерский)*

## Изменение подхода для Qwen2.5-3B

Было решено упростить промпт. Вместо детального описания структуры частушки, модель обучалась по принципу "дай частушку" - "на частушку".

К сожалению, модель **не запустилась** из-за недостатка памяти GPU.

## Источники по обучению:

1. [TRL Documentation](https://huggingface.co/docs/trl/)
2. [SFT Trainer Guide](https://huggingface.co/docs/trl/sft_trainer)
3. [Training Tips for LLMs](https://github.com/huggingface/blog/blob/main/llm-training-tips.md)

---

# Метрики оценки

Для оценки качества генерации частушек используются как автоматические, так и ручные метрики.

## Автоматические метрики

### 1. Perplexity

Perplexity измеряет, насколько хорошо модель предсказывает текст:

**PPL = exp(-(1/N) Σ log P(wᵢ | w<ᵢ))**

где **N** — количество токенов, **P(wᵢ | w<ᵢ)** — вероятность токена при условии предыдущих.

**Интерпретация:** чем ниже perplexity, тем лучше модель предсказывает текст.

```python
def calculate_perplexity(model, tokenizer, texts):
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    return np.exp(total_loss / total_tokens)
```

### 2. Distinct N-grams

Distinct N-grams измеряет разнообразие генерируемого текста:

**Distinct-n = |unique n-grams| / |total n-grams|**

**Интерпретация:** значение близкое к 1 означает высокое разнообразие (мало повторений).

```python
def calculate_distinct_ngrams(texts, n=1):
    all_ngrams = []

    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)
```

### 3. Coherence

Coherence измеряет семантическую связность текста с помощью sentence embeddings:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_coherence = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def compute_coherence(text):
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return 0.0

    embeddings = model_coherence.encode(lines)
    similarities = []

    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        similarities.append(sim)

    return np.mean(similarities)
```

**Интерпретация:** значение от 0 до 1, где 1 — полная семантическая связность строк.

## Ручная оценка

Эксперты (жертвы) оценивают сгенерированные частушки по шкале от 1 до 5:

| Метрика | Описание |
|---------|----------|
| **Metre** | Правильность ритма и размера |
| **Fluency** | Грамматическая корректность и естественность речи |
| **Creativity** | Оригинальность содержания и наличие юмора |

## Структурные метрики

Автоматически проверяется:
- Количество строк (должно быть 4)
- Наличие рифмы в парах строк (1-2, 3-4) или (2-4)

```python
def check_structure(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return len(lines) == 4

def check_rhyme(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) != 4:
        return False

    endings = [line.split()[-1][-3:] for line in lines if line.split()]

    rhyme_aabb = endings[0] == endings[1] and endings[2] == endings[3]
    rhyme_abcb = endings[1] == endings[3]

    return rhyme_aabb or rhyme_abcb
```

## Источники по метрикам:

1. [Perplexity Explained](https://huggingface.co/docs/transformers/perplexity)
2. [Distinct N-grams for Text Generation](https://arxiv.org/abs/1510.03055)
3. [Sentence-Transformers Documentation](https://www.sbert.net/)

---

# Генерация частушек

После обучения модель может генерировать частушки с настраиваемыми параметрами.

## Код генерации

```python
def generate_chastushka(model, tokenizer, theme=None):
    prompt = f"<chastushka>{theme}" if theme else "<chastushka>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            top_k=GEN_TOP_K,
            repetition_penalty=GEN_REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if "<chastushka>" in generated and "</chastushka>" in generated:
        chastushka = generated.split("<chastushka>")[1].split("</chastushka>")[0].strip()
    else:
        chastushka = generated.replace("<chastushka>", "").replace("</chastushka>", "").strip()

    return chastushka

print("Примеры генерации:")

for i in range(5):
    print("=" * 50)
    print(f"Частушка #{i+1}")
    print("=" * 50)
    print(generate_chastushka(model, tokenizer))
    print()
```

## Параметры генерации

| Параметр | Значение | Описание |
|----------|----------|----------|
| temperature | 0.6 | Креативность (0.1 = консервативно, 1.0 = креативно) |
| top_p | 0.85 | Nucleus sampling (probability mass) |
| top_k | 30 | Top-K sampling (количество вариантов) |
| repetition_penalty | 1.3 | Штраф за повторение токенов |
| max_new_tokens | 60 | Максимальное количество новых токенов |

### Влияние temperature:

- **0.3-0.5:** более предсказуемые, но скучные тексты
- **0.6-0.8:** баланс между предсказуемостью и креативностью
- **0.9-1.0:** очень креативные, но иногда бессмысленные тексты

## Источники по генерации:

1. [Text Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
2. [Sampling Methods Explained](https://huggingface.co/blog/how-to-generate)

---

# Результаты и выводы

## Критерии успеха

| Критерий | Цель | Достигнуто | Статус |
|----------|------|------------|--------|
| 4 строки | >70% | 80% | Выполнено |
| Рифма | >50% | 0-10% | Не выполнено |
| Мы живые | 100% | 100% | Главное |

## Лучшие метрики

### Vikhr-Llama-3.2-1B (версия 1):
- Val Loss: 0.720 (лучший)
- Структура: 80%
- Perplexity: 24.07
- Distinct-1: 0.465
- Coherence: 0.398

### Qwen2.5-1.5B (235 примеров):
- Perplexity: 27.60
- Distinct-1: 0.544
- Distinct-2: 0.940
- Coherence: 0.439

## Сравнительная таблица всех моделей

| Модель | Датасет | Структура | Рифма | Perplexity | Примечание |
|--------|---------|-----------|-------|------------|------------|
| Vikhr-Llama v1 | 500 | 80% | 0% | 24.07 | Лучший вариант |
| Vikhr-Gemma | 500 | 0% | 0% | - | Артефакты, английский |
| Qwen-1.5B (50) | 50 | 60% | 0% | 27.60 | Бессмыслица |
| Qwen-1.5B (235) | 235 | 60% | 0% | 27.60 | Лучше, но рифмы нет |
| Qwen-0.5B | 235 | 20% | 0% | - | Катастрофа |
| Qwen3-0.6B | 235 | 30% | 0% | - | Переход на венгерский |
| Qwen2.5-3B | - | - | - | - | Не запустилась |

## Проблемы

1. **Отсутствие рифмы:** ни одна из моделей не смогла стабильно генерировать рифмованный текст
2. **Артефакты:** частое появление странных токенов, особенно в Vikhr-Gemma
3. **Переход на другие языки:** Qwen-модели иногда переключались на английский или венгерский
4. **Несвязность:** некоторые частушки не имеют смысловой связи между строками
5. **Зацикленность:** модели часто повторяют одну и ту же тему

## Причины неудач

1. **Малый размер датасета:** 160-750 частушек недостаточно для обучения сложным паттернам
2. **Специфика задачи:** генерация рифмованного текста — сложная задача даже для больших моделей
3. **Отсутствие явного обучения рифме:** модель не получала прямого сигнала о необходимости рифмовать
4. **Размер модели:** 1-2B параметров может быть недостаточно для понимания тонкостей рифмовки
5. **AI-аугментация:** 68-78% датасета сгенерировано AI, что может снизить качество

## Возможные улучшения

1. **Увеличение датасета:** собрать 5000+ качественных частушек без AI
2. **Reinforcement Learning:** добавить reward за наличие рифмы (RLHF)
3. **Постобработка:** автоматическая замена последних слов на рифмующиеся варианты
4. **Фонетические embeddings:** явно обучать модель фонетической близости слов
5. **Большие модели:** использовать 7B+ параметров с более эффективной квантизацией
6. **Контролируемая генерация:** использовать constrained decoding для гарантии рифмы

---

# Неожиданный результат: Claude Sonnet 4.5

Во время работы над проектом было обнаружено, что современная коммерческая модель **Claude Sonnet 4.5** от Anthropic генерирует частушки значительно лучше, чем обученные модели.

## Примеры от Claude:

**Про технологии:**
```
Я скачала нейросеть,
Чтоб стихи мне сочиняла.
А она весь день грузилась —
Видно, много не читала!
```
**Рифма:** AABB

**Про любовь:**
```
Я любила паренька,
Он мне песни напевал.
А потом узнала правду —
Просто Яндекс.Музыку включал!
```
**Рифма:** ABCB

**Про быт:**
```
Я купила робот-пылесос,
Чтоб полы мне убирал.
А он к кошке прицепился —
Весь квартал переполошал!
```
**Рифма:** ABCB

## Статистика Claude:
- Структура: 100%
- Рифма: 100%
- Юмор: присутствует
- Связность: отличная
- Современность: использует актуальные темы

Это показывает, что для генерации качественных частушек требуются либо очень большие модели (100B+ параметров), либо специализированное обучение с явными ограничениями на рифму.

---

# Итоги проекта

## Что удалось изучить:

1. Fine-tuning малых LLM с использованием LoRA и 4-bit квантизации
2. Работу с русскоязычными моделями (Vikhr, Qwen)
3. Подготовку специализированных датасетов для поэтических текстов
4. Метрики оценки генеративных моделей (Perplexity, Distinct N-grams, Coherence)
5. Ограничения малых моделей в сложных творческих задачах

## Выводы:

- Малые модели (1-2B) могут генерировать связный текст с правильной структурой
- Малые модели не могут стабильно генерировать рифмованный текст без специального обучения
- AI-аугментация датасета (68-78%) может снижать качество обучения
- Для поэтических текстов нужны либо большие модели (7B+), либо специализированные методы
- Claude Sonnet 4.5 показал, что задача решаема при достаточном размере модели

Несмотря на то, что не удалось достичь всех поставленных целей (особенно по рифме), проект дал ценный опыт в работе с языковыми моделями и показал, что генерация качественных поэтических текстов остаётся сложной задачей.

---

# Структура репозитория

```
ML/
├── config.py                    # Конфигурация параметров
├── model_setup.py               # Загрузка и настройка модели с LoRA
├── data_prep.py                 # Подготовка и токенизация датасетов
├── training.py                  # Скрипт обучения с SFTTrainer
├── generate.py                  # Генерация частушек
├── coherence.py                 # Метрики связности текста
├── evaluate.py                  # Оценка качества моделей
│
├── dataset1.csv                 # 160 частушек (чистые)
├── dataset2.csv                 # 500 частушек (68% AI)
├── dataset3.csv                 # 750 частушек (78.6% AI)
├── dataset4.csv                 # 750+ частушек (расширенный)
│
├── chastushki-model/            # Сохранённые чекпоинты
│   ├── checkpoint-epoch-1/
│   ├── checkpoint-epoch-2/
│   └── final/                   # Финальная модель
│
├── Generator-chastushek.pdf     # Презентация проекта
└── README.md                    # Этот файл
```

---

# Установка и запуск

## Требования

- Python 3.8+
- CUDA 11.8+ (для GPU)
- 16GB RAM
- 8-16GB VRAM

## Установка зависимостей

```bash
pip install torch transformers datasets peft trl bitsandbytes \
            accelerate sentence-transformers pandas numpy scikit-learn
```

## Обучение модели

```bash
python training.py
```

## Генерация

```bash
python generate.py
```

## Использование в коде

```python
from generate import generate_chastushka
from model_setup import model, tokenizer

print(generate_chastushka(model, tokenizer))

print(generate_chastushka(model, tokenizer, theme="про кота"))
```

---

# Список литературы

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
3. [Как научить свою нейросеть генерировать стихи](https://habr.com/ru/articles/334046/) — Habr
4. [Как генерировать стихи с помощью силлабо-тонической трансформенной языковой модели](https://habr.com/ru/companies/sberdevices/articles/660717/) — SberDevices
5. [Vikhr Models](https://huggingface.co/Vikhrmodels) — Hugging Face
6. [Qwen Models](https://huggingface.co/Qwen) — Hugging Face
7. [PEFT Documentation](https://huggingface.co/docs/peft/) — Hugging Face
8. [TRL Documentation](https://huggingface.co/docs/trl/) — Hugging Face
9. [Sentence-Transformers](https://www.sbert.net/) — SBERT.net
10. [Distinct N-grams metric](https://arxiv.org/abs/1510.03055) — Li et al., 2015

---

# Благодарности

- **Hugging Face** за библиотеки Transformers, PEFT и TRL
- **Vikhrmodels** за русскоязычные адаптации моделей
- **Qwen team** за открытые модели
- **Claude Sonnet 4.5** за помощь в создании этой документации
- **Одногруппники** за моральную поддержку
- **Преподаватели Университета Сириус** за возможность работать над интересным проектом

---

**Проект:** Генератор частушек  
**GitHub:** https://github.com/djakovl/ML  
**Дата:** Февраль 2026
