import os
import json
from datetime import datetime
from utils.pdf_parser import extract_blocks_from_pdfs
from utils.embedding_utils import load_encoder, encode_texts, compute_similarity
from utils.ranking import rank_sections, extract_top_sentences
from utils.json_writer import write_output_json
from utils.predict_utils import predict_persona_and_job

INPUT_FOLDER = "data/input_pdf"
OUTPUT_JSON = "output/challenge1b_output.json"

def run_pipeline():
    # Get all PDF files in the input folder
    input_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdf")]

    if not input_files:
        print("❌ No PDF files found in the input folder.")
        return

    print("🔍 Extracting content from PDFs...")
    documents = extract_blocks_from_pdfs(INPUT_FOLDER, input_files)

    combined_text = " ".join(
        block["text"] for doc in documents for block in doc["blocks"]
    )

    predicted_persona, predicted_job = predict_persona_and_job(combined_text)
    print(f"🔍 Predicted Persona: {predicted_persona}")
    print(f"📌 Predicted Job: {predicted_job}")

    print("🧠 Loading embedding model...")
    encoder = load_encoder("models")

    persona_job_text = f"{predicted_persona}. Task: {predicted_job}"
    print(f"🔢 Embedding persona+job: {persona_job_text}")
    query_vec = encode_texts([persona_job_text], encoder)[0]

    print("📊 Scoring sections...")
    section_scores = []
    for doc in documents:
        for block in doc["blocks"]:
            block_vec = encode_texts([block["text"]], encoder)[0]
            sim_score = compute_similarity(query_vec, block_vec)
            section_scores.append({
                "document": doc["name"],
                "page": block["page"],
                "text": block["text"],
                "similarity": sim_score
            })

    print("🏅 Ranking top sections...")
    top_sections = rank_sections(section_scores, top_k=5)

    print("🔬 Extracting sub-section analysis...")
    top_subsections = extract_top_sentences(top_sections, query_vec, encoder, top_n=5)

    print("📝 Writing JSON output...")
    output = write_output_json(
        documents=input_files,
        persona=predicted_persona,
        job=predicted_job,
        top_sections=top_sections,
        top_sentences=top_subsections,
        timestamp=str(datetime.now())
    )

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Output written to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_pipeline()
