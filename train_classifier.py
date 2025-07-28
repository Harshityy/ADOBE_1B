import os
import json
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter

# === Load Data ===
with open("data/train_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [x["text"] for x in data]
personas = [x["persona"] for x in data]
jobs = [x["job"] for x in data]

# === Encode Labels ===
persona_encoder = LabelEncoder()
job_encoder = LabelEncoder()

y_persona = persona_encoder.fit_transform(personas)
y_job = job_encoder.fit_transform(jobs)

# === Check Class Distribution ===
print("Persona label distribution:", Counter(y_persona))
print("Job label distribution:", Counter(y_job))

# === Split Data ===
X_train, X_test, y_p_train, y_p_test, y_j_train, y_j_test = train_test_split(
    texts, y_persona, y_job, test_size=0.2, random_state=42
)

# === Create TF-IDF + Logistic Pipelines ===
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

persona_clf = make_pipeline(
    tfidf,
    LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")
)

job_clf = make_pipeline(
    tfidf,
    LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")
)

# === Train Models ===
print("\nTraining persona classifier...")
persona_clf.fit(X_train, y_p_train)

print("Training job classifier...")
job_clf.fit(X_train, y_j_train)

# === Predict on Test Set ===
y_p_pred = persona_clf.predict(X_test)
y_j_pred = job_clf.predict(X_test)

# === Evaluation (Safe Report Print) ===
def safe_classification_report(y_true, y_pred, encoder, title=""):
    labels = sorted(set(y_true) | set(y_pred))
    names = [encoder.classes_[i] for i in labels]
    print(f"\n=== {title} Classification Report ===")
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))

safe_classification_report(y_p_test, y_p_pred, persona_encoder, title="Persona")
safe_classification_report(y_j_test, y_j_pred, job_encoder, title="Job")

# === Save Models and Encoders ===
os.makedirs("models", exist_ok=True)
joblib.dump(persona_clf, "models/persona_classifier.pkl")
joblib.dump(job_encoder, "models/job_encoder.pkl")
joblib.dump(persona_encoder, "models/persona_encoder.pkl")
joblib.dump(job_clf, "models/job_classifier.pkl")

print("\n✅ All models and encoders saved successfully.")
