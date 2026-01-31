# Fake News Detection Using Machine Learning & Deep Learning

## 🧠 Research Context
Fake news spreads rapidly and can manipulate public opinion. Numerous models exist to detect fake news, but few studies evaluate how well models trained on one dataset generalize to unseen news sources.

This project benchmarks multiple detection methods and tests their robustness on **out-of-distribution data**.

---

## 🎯 Research Questions
1. Which models perform best on unseen (out-of-distribution) news data?
2. Do deep learning approaches generalize better than classical machine learning?
3. What are the main patterns of errors (false positives vs false negatives)?

---

## 📊 Datasets
- **Ti-CNN dataset:** ~20,000 labeled news articles with text and metadata.
- **Unseen/OOD data:**  
  – Real news scraped from *Time Magazine*  
  – Fake news scraped from *The Onion*

---

## 🔬 Methodology

### 🧪 Preprocessing
- Text normalization (lowercasing, removing noise)
- Tokenization
- Stop word removal

### 📈 Feature Extraction
- TF-IDF vectors
- Word2Vec embeddings

### 🧠 Models Implemented
**Machine Learning:**
- Logistic Regression
- Support Vector Machine
- Passive-Aggressive
- Random Forest (optional)

**Deep Learning:**
- Simple CNN
- BiLSTM
- Word2Vec + BiLSTM + CNN
- Transformer baseline (BERT)

---

## 📌 Experimental Setup

- Train/test split validation  
- Cross-validation (k-fold)  
- Evaluation metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score

**Benchmarking on unseen data is the research focus.**

---

## 📈 Results Summary

| Model | Accuracy (Train/Test) | OOD Accuracy |
|-------|----------------------|---------------|
| SVM | 0.87 | 0.75 |
| Logistic Regression | 0.85 | 0.72 |
| BiLSTM | 0.89 | 0.78 |
| **Word2Vec + BiLSTM + CNN** | **0.92** | **0.86** |
| BERT | 0.91 | 0.79 |

> **Highest OOD generalization:** *Word2Vec + BiLSTM + CNN*

---

## 🧠 Key Findings
- Deep learning models outperform classical ML on unseen sources.
- Hybrid models using pretrained embeddings generalize best.
- Transformer models (BERT) had strong training performance but weaker OOD results in this setup.

---

## ⚠️ Limitations
- Source distribution differences may bias performance.
- Additional tuning could improve transformer OOD performance.
- Dataset labeling inconsistencies.

---

## 🔭 Future Work
- Evaluate multilingual datasets
- Add explainability (LIME/SHAP)
- Test robustness to adversarial examples
