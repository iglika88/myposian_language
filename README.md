# The Myposian Language

This project contains the first compiled dictionary of the *Myposian* language, an imaginary language spoken by the character **Balki Bartokomous** from the American sitcom *Perfect Strangers* (1986–1993). The Myposian language is associated with the fictitious Greek island of **Mypos**.

## Files

### 📖 Dictionary

- [myposian_dictionary.csv](myposian_dictionary.csv)

This file contains a dictionary of the Myposian language.

| Column | Description |
|----------|----------|
| `myposian_entry` | Myposian word, expression, proper noun, or longer text |
| `meaning` | English translation or explanation |
| `reference` | Season(s) and episode(s) where the item was encountered |
| `type` | `word/expression`, `long text`, or `proper noun` |


### Experiments 

- [experiments/langid_experiment.py](experiments/langid_experiment.py)

  This file contains code that seeks to identify Myposian dictionary entries as one of the languages supported by the Python tool langid.

- [experiments/classifier_experiment.py](experiments/classifier_experiment.py)

This file contains code to train a simple word-based classification model on several relevant languages and verify which one of them Myposian words will me classified as. 



## Sources

The dictionary was compiled from:

- Episode scripts of *Perfect Strangers* available at: https://subslikescript.com/series/Perfect_Strangers-90501
- Manual additions, corrections, translations, and annotations by the author

## Notes

The dictionary is incomplete and may contain errors. Please feel free to edit. 
