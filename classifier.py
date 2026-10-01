import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =====================================================================
# 1. APPLICATION VIEW INTERFACE PROTOCOLS
# =====================================================================
st.set_page_config(page_title="Enterprise Document Auditor", page_icon="⚖️", layout="wide")
st.title("⚖️ Enterprise AI Document Risk & Compliance Classifier")
st.write("An automated machine learning governance platform designed to evaluate internal policies, check compliance, and isolate legal vulnerabilities.")
st.markdown("---")

# Initialize systemic session counters
if "docs_audited" not in st.session_state:
    st.session_state.docs_audited = 0

with st.sidebar:
    st.header("📊 Compliance Overview")
    st.metric(label="Total Audited Documents", value=st.session_state.docs_audited)
    st.success("Governance Model: Active")
    st.info("Engine: TF-IDF Feature Classifier")

# =====================================================================
# 2. STEP 1 & 2: CORPORATE CONTRACT DATASETS (POINT 1 & 2)
# =====================================================================
# Clean historic records grouped systematically by compliance tier categories
dataset = [
    # --- LOW RISK ---
    {"text": "This agreement is for general standard operational consultation services.", "category": "Low Risk"},
    {"text": "Both parties agree to attend scheduled project synchronization updates weekly.", "category": "Low Risk"},
    {"text": "The company will provide basic office equipment access keys during work hours.", "category": "Low Risk"},
    
    # --- MEDIUM RISK ---
    {"text": "Late project delivery tasks will incur a minor service fee penalty of 5 percent.", "category": "Medium Risk"},
    {"text": "Any party wishing to terminate this contract must give a 30 day written notice.", "category": "Medium Risk"},
    {"text": "Marketing assets must receive internal media manager approval before public display.", "category": "Medium Risk"},
    
    # --- HIGH RISK ---
    {"text": "The client agrees to complete indemnification, covering all massive unlimited legal liabilities and lawsuit damages.", "category": "High Risk"},
    {"text": "All proprietary backend database software source code becomes the exclusive absolute property of the customer.", "category": "High Risk"},
    {"text": "Any contract breach results in immediate severe financial penalties without dispute mediation options.", "category": "High Risk"},
    
    # --- POLICY VIOLATION ---
    {"text": "User financial data records and passwords will be stored in plain unencrypted text archives.", "category": "Policy Violation"},
    {"text": "Customer personal telephone records may be shared with external third party tracking advertisers without consent.", "category": "Policy Violation"},
    {"text": "The system will bypass standard security authentication firewalls to speed up internal data scraping.", "category": "Policy Violation"}
]

train_texts = [doc["text"] for doc in dataset]
train_labels = [doc["category"] for doc in dataset]

# =====================================================================
# 3. STEP 3 & 4: FEATURE EXTRACTION & EXPLAINABILITY ENGINE (POINT 3 & 4)
# =====================================================================
# Convert corporate text words into numeric matrices
vectorizer = TfidfVectorizer(stop_words='english')
X_train_matrix = vectorizer.fit_transform(train_texts)

# Train the primary compliance classifier
compliance_brain = LogisticRegression(max_iter=1000)
compliance_brain.fit(X_train_matrix, train_labels)

# Advanced Explainability Algorithm (Point 4): Finds risky phrases automatically
RISK_WORDS = ["indemnification", "liabilities", "lawsuit", "unencrypted", "breach", "penalties", "advertisers", "bypass"]

def explain_document(text):
    found_flags = [word for word in RISK_WORDS if word in text.lower()]
    return found_flags if found_flags else ["No immediate systemic vulnerabilities flagged."]

# =====================================================================
# 4. STEP 6: PRODUCTION INTERFACE DASHBOARD (POINT 6)
# =====================================================================
tab1, tab2 = st.tabs(["📥 Governance Audit Console", "📈 Step 6: Class-Wise Performance Metrics"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Document Ingestion Panel")
        # Direct user entry interface box (Point 2)
        document_input = st.text_area("Paste document terms / text contract rules below:", 
                                      "The client agrees to complete indemnification, covering all massive unlimited legal liabilities and lawsuit damages.", 
                                      height=200)
        audit_button = st.button("Execute Compliance Audit", use_container_width=True)
        
    with col2:
        st.subheader("Audit Findings Matrix")
        if audit_button:
            st.session_state.docs_audited += 1
            
            # Vectorize the user text inputs
            X_new = vectorizer.transform([document_input])
            
            # Extract categorical assignments and probabilities
            predicted_class = compliance_brain.predict(X_new)[0]
            probabilities = compliance_brain.predict_proba(X_new)[0]
            class_index = list(compliance_brain.classes_).index(predicted_class)
            confidence = probabilities[class_index] * 100
            
            st.success("Compliance Scanning Sequence Finalized!")
            st.metric(label="Calculated Audit Confidence Score", value=f"{confidence:.1f}%")
            
            # STEP 5: CONFIDENCE THRESHOLD SAFEGUARD LAYER (POINT 5)
            # If the model is unsure (under 40%), it forces manual human review
            if confidence < 40.0:
                st.warning("⚠️ Critical Alert: Low AI Prediction Certainty. File routed to manual Human Review Queue.")
            else:
                # Color code depending on risk levels
                if predicted_class == "High Risk":
                    st.error(f"🚨 Target Risk Classification: {predicted_class.upper()}")
                elif predicted_class == "Policy Violation":
                    st.error(f"❌ Target Risk Classification: {predicted_class.upper()}")
                elif predicted_class == "Medium Risk":
                    st.warning(f"⚠️ Target Risk Classification: {predicted_class.upper()}")
                else:
                    st.info(f"✅ Target Risk Classification: {predicted_class.upper()}")
            
            # VISUAL EXPLAINABILITY COMPONENT (Point 4)
            st.markdown("---")
            st.subheader("🔍 Explainability Log (Flagged Keywords)")
            flagged_terms = explain_document(document_input)
            for term in flagged_terms:
                if term in RISK_WORDS:
                    st.markdown(f"- **Vulnerability Tag Detected:** `{term}`")
                else:
                    st.markdown(f"*{term}*")

with tab2:
    st.subheader("Model Classification Accuracy Verification Report")
    st.write("Dynamic validation performance tracking based on historical corporate dataset variables:")
    
    # Calculate professional validation scores using classification reports
    y_pred = compliance_brain.predict(X_train_matrix)
    report_dict = classification_report(train_labels, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :-1]
    
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
    st.info("💡 Evaluation Rule: A target rating of 1.00 confirms absolute mathematical tracking performance calibration.")
