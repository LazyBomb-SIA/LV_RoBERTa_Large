# Latvian SpaCy Model: lv_roberta_large
拉脱维亚语 SpaCy 模型：lv_roberta_large

# Acknowledgements
致谢

Thanks to the University of Latvia, AI Lab, and all contributors of the Latvian UD Treebank.  
感谢拉脱维亚大学人工智能实验室，以及拉脱维亚 UD Treebank 所有贡献者。

Model development supported by [LazyBomb.SIA].  
模型开发得到 [LazyBomb.SIA] 支持。

Inspired by the spaCy ecosystem and training framework.  
灵感来自 spaCy 生态系统及其训练框架。

---

## Overview
## 模型概览

This is a **spaCy transformer-based pipeline for Latvian**, built with the **XLM-RoBERTa-large backbone**.  
这是一个 **基于 Transformer 的拉脱维亚语 spaCy 流水线**，底层采用 **XLM-RoBERTa-large 预训练模型**。

It includes the following components:  
包含以下组件：

- **Transformer** / Transformer 编码器 (XLM-RoBERTa-large)
- **Tagger** / 词性标注器
- **Morphologizer** / 形态分析器
- **Parser** / 依存句法分析器
- **Sentence Segmenter (senter)** / 分句器
- **Lemmatizer** / 词形还原器

**Model type / 模型类型:** Transformer pipeline (XLM-RoBERTa-large backbone)  
**Language / 语言:** Latvian (lv)  
**Recommended hardware / 推荐硬件:** CPU for small-scale use, GPU recommended for faster training  
小规模使用可用 CPU，建议使用 GPU 提高训练速度  

---

## Training Data
## 训练数据

The model was trained on the **Latvian UD Treebank v2.16**, which is derived from the **Latvian Treebank (LVTB)** created at the University of Latvia, AI Lab.  
本模型在 **拉脱维亚 UD Treebank v2.16** 上训练，该数据集由拉脱维亚大学 AI 实验室创建的 **LVTB** 转换而来。

- **License / 许可:** CC BY-SA 4.0 (Attribution-ShareAlike) / 署名-相同方式共享 4.0 国际  
- **Data splits / 数据划分:**  
  - Train / 训练集: 15055 sentences / 句子  
  - Dev / 验证集: 2080 sentences / 句子  
  - Test / 测试集: 2396 sentences / 句子  

**References / 参考文献:**  
Pretkalniņa, L., Rituma, L., Saulīte, B., et al. (2016–2018). Various publications on LVTB and UD Treebank for Latvian.  
Pretkalniņa, L., Rituma, L., Saulīte, B. 等（2016–2018）。关于拉脱维亚 LVTB 与 UD Treebank 的多篇论文。

> ⚠️ Users of this model must comply with the original CC BY-SA 4.0 license.  
> ⚠️ 使用本模型的用户必须遵守原始 CC BY-SA 4.0 许可协议。

---

## Usage
## 使用方法

```python
import spacy
import numpy as np

# Load the pipeline
# 加载模型流水线
model_dir = snapshot_download(repo_id="JesseHuang922/lv_roberta_large", repo_type="model")
nlp = spacy.load(model_dir)

# Example text
# 示例文本
text = """Baltijas jūras nosaukums ir devis nosaukumu baltu valodām un Baltijas valstīm.
Terminu "Baltijas jūra" (Mare Balticum) pirmoreiz lietoja vācu hronists Brēmenes Ādams 11. gadsimtā."""

# Process text
# 处理文本
doc = nlp(text)

# ------------------------
# Tokenization / 分词
# ------------------------
print("Tokens / 分词结果:")
print([token.text for token in doc])

# ------------------------
# Lemmatization / 词形还原
# ------------------------
print("Lemmas / 词形还原结果:")
print([token.lemma_ for token in doc])

# ------------------------
# Part-of-Speech Tagging / 词性标注
# ------------------------
print("POS tags / 词性标注:")
for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")

# ------------------------
# Morphological Features / 形态特征
# ------------------------
print("Morphological features / 形态特征:")
for token in doc:
    print(f"{token.text}: {token.morph}")

# ------------------------
# Dependency Parsing / 依存句法分析
# ------------------------
print("Dependency parsing / 依存句法分析:")
for token in doc:
    print(f"{token.text} <--{token.dep_}-- {token.head.text}")

# ------------------------
# Sentence Segmentation / 分句
# ------------------------
print("Sentences / 分句结果:")
for sent in doc.sents:
    print(sent.text)

# ------------------------
# 查看流水线组件
# ------------------------
print("Pipeline components / 流水线组件:")
print(nlp.pipe_names)

# Transformer vectors
# Transformer 向量表示
vectors = np.vstack([token.vector for token in doc])
print("Token vectors shape / Token 向量维度:", vectors.shape)
