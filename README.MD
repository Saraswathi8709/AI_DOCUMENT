# ⚖️ Enterprise AI Document Risk & Compliance Classifier

An automated machine learning governance platform engineered to parse text clauses, evaluate operational policy compliance, and isolate regulatory legal vulnerabilities dynamically.

## 🚀 Architectural Features
- **TF-IDF Semantic Tokenizer:** Transforms long-form contract strings into numerical vectors based on text importance frequencies.
- **Multi-Class Evaluation Module:** Groups documents accurately into 4 critical tiers: Low Risk, Medium Risk, High Risk, or Policy Violation.
- **Explainability Keyword Trace:** Uses token tagging arrays to isolate text triggers causing compliance exceptions.
- **Confidence Gatekeeper Layer:** Automatically routes documents to a manual Human Review Queue if prediction probabilities drop below safe parameters.

## 🛠️ Tech Stack & Engine Tools
- **Core Architecture:** Python 3.10+
- **Machine Learning Core:** Scikit-Learn (Logistic Regression Engine)
- **Data Structuring Panel:** Pandas Data Analytics System
- **Interface Rendering:** Streamlit Layout Engine

## ⚙️ Quick Installation Guide
1. Launch local terminal workspace and install base packages:
   ```bash
   pip install streamlit pandas numpy scikit-learn
   ```
2. Initiate the platform dashboard:
   ```bash
   streamlit run classifier.py
   ```
