---
language:
- lv
base_model:
- FacebookAI/xlm-roberta-large
license: cc-by-sa-4.0
datasets:
- universal_dependencies
metrics:
- accuracy
- uas
- las
---

# Latvian SpaCy Model: lv_roberta_large

## Hugging Face Model Repo:
➡️ https://huggingface.co/JesseHuang922/lv_roberta_large/tree/main

---

## Overview

This is a **spaCy transformer-based pipeline for Latvian**, built with the **XLM-RoBERTa-large backbone**.  

It includes the following components:  

- **Transformer** (XLM-RoBERTa-large)
- **Tagger**
- **Morphologizer**
- **Parser**
- **Sentence Segmenter (senter)**
- **Lemmatizer**
- (Note: Transformer component internally uses a `tok2vec` listener)

**Model type:** Transformer pipeline (XLM-RoBERTa-large backbone)  
**Language:** Latvian (lv)  
**Recommended hardware:** CPU for small-scale use, GPU recommended for faster training  

---

## Training Data

The model was trained on the **Latvian UD Treebank v2.16**, which is derived from the **Latvian Treebank (LVTB)** created at the University of Latvia, Institute of Mathematics and Computer Science, Artificial Intelligence Laboratory (AI Lab).  

- **Dataset source:** [UD Latvian LVTB](https://github.com/UniversalDependencies/UD_Latvian-LVTB)  
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)  
- **Data splits:**  
  - Train: 15,055 sentences  
  - Dev: 2,080 sentences  
  - Test: 2,396 sentences  

---

## Acknowledgements

- Thanks to the **University of Latvia, AI Lab**, and all contributors of the **Latvian UD Treebank**.  
- Model development supported by [LazyBomb.SIA].  
- Inspired by the **spaCy ecosystem** and training framework.  
- The Latvian UD Treebank was developed with support from multiple grants, including:  
  - European Regional Development Fund (Grant No. 1.1.1.1/16/A/219, 1.1.1.2/VIAA/1/16/188)  
  - State Research Programme "National Identity"  
  - State Research Programme "Digital Resources for the Humanities" (Grant No. VPP-IZM-DH-2020/1-0001)  
  - State Research Programme "Research on Modern Latvian Language and Development of Language Technology" (Grant No. VPP-LETONIKA-2021/1-0006)  

---

## License

This model is released under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to:
- **Share** — copy and redistribute the material in any medium or format, for any purpose, even commercially.  
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.  

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.  
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.  

---

## References

- Pretkalniņa, L., Rituma, L., Saulīte, B., et al. (2016–2025). Universal Dependencies Latvian Treebank (LVTB).  
- Grūzītis, N., Znotiņš, A., Nešpore-Bērzkalne, G., Paikens, P., et al. (2018). Creation of a Balanced State-of-the-Art Multilayer Corpus for NLU. *LREC 2018*.  
- Pretkalniņa, L., Rituma, L., Saulīte, B. (2016). Universal Dependency Treebank for Latvian: A Pilot. *Baltic Perspective Workshop*.  

---

---

## Usage
You can either:

1. **Download the model directly from the Hugging Face Hub**  
   Using `huggingface_hub.snapshot_download`, the model files will be automatically fetched and cached locally.

      ```python
      import spacy
      from huggingface_hub import snapshot_download
      
      # Load the pipeline
      model_dir = snapshot_download(repo_id="JesseHuang922/lv_roberta_large", repo_type="model")
      nlp = spacy.load(model_dir)
      ```

2. **Install from the pre-built wheel package**  
   Download the wheel file (**lv_roberta_large-1.0.0-py3-none-any.whl**) and install it into your virtual environment with:

       ```bash
       pip install lv_roberta_large-1.0.0-py3-none-any.whl
       
---

## Dependencies

The following Python packages are required to run the Latvian XLM-RoBERTa spaCy pipeline:

| Package                | Minimum Version | Notes                                                                                  | 
| ---------------------- | --------------- | -------------------------------------------------------------------------------------- | 
| **spaCy**              | 3.8.7           | Main NLP framework                                                          | 
| **spacy-transformers** | 1.3.9           | Integrates spaCy with Hugging Face Transformers  | 
| **transformers**       | 4.49.0          | Hugging Face Transformers library                       | 
| **torch**              | 2.8.0           | PyTorch backend for transformers                           | 
| **tokenizers**         | 0.21.4          | Fast tokenizer support                                                        | 
| **safetensors**        | 0.6.2           | Secure tensor storage for transformer weights                      | 
| **huggingface-hub**    | 0.34.4          | Download and manage the model files from the Hugging Face Hub      |


```python
import spacy
import numpy as np
from huggingface_hub import snapshot_download

# Load the pipeline
model_dir = snapshot_download(repo_id="JesseHuang922/lv_roberta_large", repo_type="model")
nlp = spacy.load(model_dir)

# Example text
text = """Baltijas jūras nosaukums ir devis nosaukumu baltu valodām un Baltijas valstīm.
Terminu "Baltijas jūra" (Mare Balticum) pirmoreiz lietoja vācu hronists Brēmenes Ādams 11. gadsimtā."""

# Process text
doc = nlp(text)

# ------------------------
# Tokenization 
# ------------------------
print("Tokens:")
print([token.text for token in doc])

# ------------------------
# Lemmatization
# ------------------------
print("Lemmas:")
print([token.lemma_ for token in doc])

# ------------------------
# Part-of-Speech Tagging
# ------------------------
print("POS tags:")
for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")

# ------------------------
# Morphological Features
# ------------------------
print("Morphological features:")
for token in doc:
    print(f"{token.text}: {token.morph}")

# ------------------------
# Dependency Parsing
# ------------------------
print("Dependency parsing:")
for token in doc:
    print(f"{token.text} <--{token.dep_}-- {token.head.text}")

# ------------------------
# Sentence Segmentation
# ------------------------
print("Sentences:")
for sent in doc.sents:
    print(sent.text)

# ------------------------
# Check Pipeline Components
# ------------------------
print("Pipeline components:")
print(nlp.pipe_names)

# Transformer vectors
vectors = np.vstack([token.vector for token in doc])
print("Token vectors shape:", vectors.shape)
