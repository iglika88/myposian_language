# ============================================================
# Myposian token-level resemblance classifier
# ============================================================

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# =========================
# User settings
# =========================

LANGUAGE_CORPORA = {
    "English": "/kaggle/working/en.csv",
    "Italian": "/kaggle/working/it.csv",
    "Greek": "/kaggle/working/gr.csv",
    "Russian": "/kaggle/working/ru.csv",
    "Japanese": "/kaggle/working/jp.csv",
}

SCAT_CORPUS = "/kaggle/working/scat.txt"

MYPOSIAN_DICTIONARY = "/kaggle/input/datasets/iglikastoupak/myposian-language/myposian_dictionary.csv"
MYPOSIAN_TEXT_COLUMN = "myposian_entry"

OUTPUT_DETAILED = "/kaggle/working/myposian_token_classification_detailed.csv"
OUTPUT_SUMMARY = "/kaggle/working/myposian_token_classification_summary.csv"

RANDOM_SEED = 42

# Try up to 3 character n-gram settings
NGRAM_SETTINGS = [
    (2, 4),
    (2, 5),
    (3, 5),
]

# =========================
# Helper functions
# =========================

def clean_token(token):
    token = str(token).strip().lower()
    token = re.sub(r"[^a-z]", "", token)
    return token

def tokenize_latin_text(text):
    return [
        clean_token(tok)
        for tok in re.findall(r"[A-Za-z]+", str(text))
        if clean_token(tok)
    ]

def load_language_csv(path, label):
    df = pd.read_csv(path)

    if "transcription" in df.columns:
        source_col = "transcription"
    elif "word" in df.columns:
        source_col = "word"
    else:
        raise ValueError(f"{path} must contain either 'word' or 'transcription' column.")

    tokens = [
        clean_token(x)
        for x in df[source_col].dropna().astype(str)
    ]
    tokens = [t for t in tokens if t]

    return pd.DataFrame({
        "token": tokens,
        "label": label
    })

def load_scat_txt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    tokens = tokenize_latin_text(text)

    return pd.DataFrame({
        "token": tokens,
        "label": "Scat"
    })

# =========================
# Load training data
# =========================

training_parts = []

for label, path in LANGUAGE_CORPORA.items():
    part = load_language_csv(path, label)
    training_parts.append(part)
    print(f"{label}: {len(part):,} training tokens loaded")

scat_df = load_scat_txt(SCAT_CORPUS)
training_parts.append(scat_df)
print(f"Scat: {len(scat_df):,} training tokens loaded")

train_df = pd.concat(training_parts, ignore_index=True)
train_df = train_df.drop_duplicates().reset_index(drop=True)

print("\nTotal unique training tokens:", len(train_df))

print("\nTraining class distribution:")
print(train_df["label"].value_counts())

# =========================
# Compare n-gram settings
# =========================

print("\nComparing character n-gram settings")
print("-----------------------------------")

best_score = -1
best_ngram = None
best_model = None

for ngram_range in NGRAM_SETTINGS:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            lowercase=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced"
        ))
    ])

    scores = cross_val_score(
        model,
        train_df["token"],
        train_df["label"],
        cv=5,
        scoring="accuracy"
    )

    mean_score = scores.mean()

    print(f"Character n-grams {ngram_range}: mean CV accuracy = {mean_score:.3f}")

    if mean_score > best_score:
        best_score = mean_score
        best_ngram = ngram_range
        best_model = model

print(f"\nBest setting: character n-grams {best_ngram}")
print(f"Best mean CV accuracy: {best_score:.3f}")

# =========================
# Train final model
# =========================

best_model.fit(train_df["token"], train_df["label"])

print("\nFinal model trained.")

# =========================
# Load and tokenize Myposian dictionary
# =========================

myposian_df = pd.read_csv(MYPOSIAN_DICTIONARY)

if MYPOSIAN_TEXT_COLUMN not in myposian_df.columns:
    raise ValueError(
        f"Column '{MYPOSIAN_TEXT_COLUMN}' not found. "
        f"Available columns: {list(myposian_df.columns)}"
    )

myposian_rows = []

for entry_id, row in myposian_df.iterrows():
    entry = str(row[MYPOSIAN_TEXT_COLUMN])
    tokens = tokenize_latin_text(entry)

    for token in tokens:
        myposian_rows.append({
            "entry_id": entry_id,
            "myposian_entry": entry,
            "token": token
        })

myposian_tokens_df = pd.DataFrame(myposian_rows)

print("\nMyposian tokenisation")
print("--------------------")
print(f"Total Myposian tokens: {len(myposian_tokens_df):,}")
print(f"Unique Myposian tokens: {myposian_tokens_df['token'].nunique():,}")
print("\nFirst 30 Myposian tokens:")
print(myposian_tokens_df["token"].head(30).tolist())

# =========================
# Predict category for every Myposian token
# =========================

predictions = best_model.predict(myposian_tokens_df["token"])
probabilities = best_model.predict_proba(myposian_tokens_df["token"])
classes = best_model.classes_

myposian_tokens_df["predicted_category"] = predictions
myposian_tokens_df["confidence"] = probabilities.max(axis=1)

for i, cls in enumerate(classes):
    myposian_tokens_df[f"prob_{cls}"] = probabilities[:, i]

# =========================
# Statistics
# =========================

counts = myposian_tokens_df["predicted_category"].value_counts()
percentages = myposian_tokens_df["predicted_category"].value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    "category": counts.index,
    "count": counts.values,
    "percentage": [percentages[cat] for cat in counts.index]
})

print("\nMyposian token classification summary")
print("-------------------------------------")

for _, row in summary_df.iterrows():
    print(f"{row['category']}: {row['count']} tokens ({row['percentage']:.2f}%)")

# =========================
# Bar chart
# =========================

plt.figure(figsize=(9, 5))

colors = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6"
]

plt.bar(
    summary_df["category"],
    summary_df["percentage"],
    color=colors[:len(summary_df)]
)

plt.title("Predicted resemblance of Myposian tokens")
plt.xlabel("Predicted category")
plt.ylabel("Percentage of Myposian tokens")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# =========================
# Examples per category
# =========================

print("\nExamples per predicted category")
print("-------------------------------")

for category in summary_df["category"]:
    examples = (
        myposian_tokens_df[myposian_tokens_df["predicted_category"] == category]
        .sort_values("confidence", ascending=False)
        [["token", "confidence"]]
        .drop_duplicates("token")
        .head(10)
    )

    print(f"\n{category}")
    print("-" * len(category))

    if len(examples) == 0:
        print("No examples.")
    else:
        for _, row in examples.iterrows():
            print(f"{row['token']} ({row['confidence']:.2f})")

# =========================
# Save outputs
# =========================

myposian_tokens_df.to_csv(OUTPUT_DETAILED, index=False, encoding="utf-8-sig")
summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

print("\nSaved files")
print("-----------")
print(OUTPUT_DETAILED)
print(OUTPUT_SUMMARY)