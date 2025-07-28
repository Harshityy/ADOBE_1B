import os
import json
from datetime import datetime
from utils.pdf_parser import extract_blocks_from_pdfs
from utils.embedding_utils import load_encoder, encode_texts, compute_similarity
from utils.ranking import rank_sections, extract_top_sentences
from utils.json_writer import write_output_json
from sentence_transformers import SentenceTransformer
import joblib

INPUT_FOLDER = "data/input_pdf"
OUTPUT_JSON = "output/challenge1b_output.json"
MODEL_DIR = "models"

# Load encoder and classifiers
encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Light, CPU-friendly, offline-ready
persona_model = joblib.load(os.path.join(MODEL_DIR, "persona_model.pkl"))
job_model = joblib.load(os.path.join(MODEL_DIR, "job_model.pkl"))
persona_encoder = joblib.load(os.path.join(MODEL_DIR, "persona_encoder.pkl"))
job_encoder = joblib.load(os.path.join(MODEL_DIR, "job_encoder.pkl"))

def predict_persona_and_job(text):
    embedding = encoder.encode([text], normalize_embeddings=True)
    persona_pred = persona_model.predict(embedding)
    job_pred = job_model.predict(embedding)
    persona = persona_encoder.inverse_transform(persona_pred)[0]
    job = job_encoder.inverse_transform(job_pred)[0]
    return persona, job

def main():
    # Step 1: Extract PDF content
    blocks = extract_blocks_from_pdfs(INPUT_FOLDER)
    full_text = " ".join([blk['text'] for blk in blocks])

    # Step 2: Predict persona & job
    persona, job = predict_persona_and_job(full_text)

    # Step 3: Encode section blocks
    section_texts = [blk['text'] for blk in blocks]
    section_embeddings = encoder.encode(section_texts, normalize_embeddings=True)

    # Step 4: Create query based on predicted intent
    query = f"{persona} wants to {job}"
    query_embedding = encoder.encode([query], normalize_embeddings=True)

    # Step 5: Rank sections & extract top sentences
    top_sections = rank_sections(query_embedding, section_embeddings, blocks)
    top_sentences = extract_top_sentences(top_sections, query_embedding, encoder)

    # Step 6: Format final output
    output = {
        "documents": [os.path.basename(p) for p in blocks[0]['source_paths']] if blocks else [],
        "persona": persona,
        "job": job,
        "top_sections": top_sentences,
        "processing_timestamp": datetime.now().isoformat()
    }

    write_output_json(output, OUTPUT_JSON)

if __name__ == "__main__":
    main()
