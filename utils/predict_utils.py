import os
import joblib
import re

MODELS_DIR = "models"

# Load the pipeline models (TF-IDF vectorizer is inside these)
persona_model = joblib.load(os.path.join(MODELS_DIR, "persona_classifier.pkl"))
job_model = joblib.load(os.path.join(MODELS_DIR, "job_classifier.pkl"))

# Label encoders
persona_encoder = joblib.load(os.path.join(MODELS_DIR, "persona_encoder.pkl"))
job_encoder = joblib.load(os.path.join(MODELS_DIR, "job_encoder.pkl"))

def preprocess(text):
    return re.sub(r"[^\w\s]", " ", text.lower())

def predict_persona_and_job(text):
    cleaned = preprocess(text)

    # Directly pass cleaned text to the pipeline
    persona_pred = persona_model.predict([cleaned])[0]
    job_pred = job_model.predict([cleaned])[0]

    # Decode labels
    persona = persona_encoder.inverse_transform([persona_pred])[0]
    job = job_encoder.inverse_transform([job_pred])[0]
    
    return persona, job
