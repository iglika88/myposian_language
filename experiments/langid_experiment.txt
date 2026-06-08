# Install if needed:
# !pip install langid

import pandas as pd
import langid
from collections import Counter
import matplotlib.pyplot as plt

# =========================
# User settings
# =========================

# Change this path if your dictionary is stored elsewhere
DICTIONARY = "/kaggle/input/datasets/iglikastoupak/myposian-language/myposian_dictionary.csv"

# Change this if the Myposian text column has a different name
TEXT_COLUMN = "myposian_entry"

# =========================
# Language code → language name
# =========================

LANGUAGE_NAMES = {
    "en": "English",
    "el": "Greek",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "pl": "Polish",
    "ru": "Russian",
    "bg": "Bulgarian",
    "uk": "Ukrainian",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "yi": "Yiddish",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "hi": "Hindi",
    "id": "Indonesian",
    "cs": "Czech",
    "sw": "Swahili",
    "mg": "Malagasy",
    "eo": "Esperanto",
    "eu": "Basque",
    "mt": "Maltese",
    "et": "Estonian",
    "sl": "Slovenian",
    "tl": "Tagalog",
    "ga": "Irish",
    "ms": "Malay",
    "hr": "Croatian",
    "unknown": "Unknown",
}

# =========================
# Load data
# =========================

df = pd.read_csv(DICTIONARY)

if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{TEXT_COLUMN}' not found. Available columns are: {list(df.columns)}"
    )

# =========================
# Detect language
# =========================

def detect_language(text):
    if pd.isna(text) or str(text).strip() == "":
        return "unknown", "Unknown", None

    lang_code, score = langid.classify(str(text))
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

    return lang_code, lang_name, score


detected = df[TEXT_COLUMN].apply(detect_language)

df["detected_language_code"] = detected.apply(lambda x: x[0])
df["detected_language"] = detected.apply(lambda x: x[1])
df["langid_score"] = detected.apply(lambda x: x[2])

# =========================
# Print statistics
# =========================

language_counts = Counter(df["detected_language"])

print("Language identification statistics")
print("----------------------------------")

for language, count in language_counts.most_common():
    print(f"{language} ({count})")

# =========================
# Bar chart: top 5 languages
# =========================

top_5 = language_counts.most_common(5)
languages = [item[0] for item in top_5]
counts = [item[1] for item in top_5]

plt.figure(figsize=(8, 5))
plt.bar(languages, counts)
plt.title("Top 5 languages identified by langid")
plt.xlabel("Detected language")
plt.ylabel("Number of Myposian entries")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# =========================
# Optional: save detailed output
# =========================

output_file = "myposian_dictionary_with_langid.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print()
print(f"Detailed results saved to: {output_file}")

try:
    display(df)
except NameError:
    pass