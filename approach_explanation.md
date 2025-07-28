 Approach Explanation – Adobe Round 1B: Persona-Driven Document Intelligence
🔍 Problem Statement
Given a set of documents (PDFs) and a target persona with a corresponding job-to-be-done, the goal is to intelligently extract and prioritize the most relevant sections and subsections based on the persona’s intent. The solution must work in a CPU-only environment, with constraints on model size (≤1GB) and runtime (≤60 seconds) per document.

✅ Solution Overview
Our approach consists of the following stages:

1. Text Extraction from PDFs
We use PyMuPDF (fitz) via the custom utility pdf_parser.py to extract:

Text content

Page number

Section-level chunking (based on visual features like font size, boldness, position – re-used from Round 1A when relevant)

2. Persona & Job Prediction
From the training data (train_data.json), we train two separate classifiers:

persona_classifier.pkl: Predicts the most probable persona

job_classifier.pkl: Predicts the job-to-be-done

Both classifiers are TF-IDF + Logistic Regression pipelines trained offline and saved in the models/ directory.

Prediction happens via predict_utils.py during runtime using the document text content.

3. Semantic Embedding of Chunks
We use a Sentence-BERT (SBERT) based model (model.safetensors, size <1GB) to convert each extracted chunk into a dense vector.

The persona and job prediction are also encoded into a dense query embedding.

This is handled via embedding_utils.py.

4. Chunk Ranking & Relevance Scoring
Using cosine similarity, we compare all chunk embeddings to the combined persona+job embedding.

Chunks are ranked in descending order of relevance.

The most relevant chunks are selected (based on a configurable threshold or top-k), and passed for final formatting.

5. Structured Output Generation
Final ranked results are written to challenge1b_output.json via json_writer.py.

Each entry contains:

json
Copy
Edit
{
  "document": "<document_name>",
  "persona": "<predicted_persona>",
  "job": "<predicted_job>",
  "relevant_sections": [
      { "page": <page_num>, "text": "<chunk_text>" },
      ...
  ]
}
🧠 Model Choices & Constraints
Component	Choice	Reason
Persona/Job Classifier	TF-IDF + Logistic Regression	Fast, small-sized, CPU-efficient
Embedding Model	Distil SBERT (Quantized .safetensors)	Semantically strong + <1GB for CPU-only inference
Chunk Ranking	Cosine Similarity	Simple and effective vector matching

⚙️ Execution Pipeline
main.py is the entrypoint.

It loads all PDFs from input_pdf/.

Calls:

Text extraction → Prediction → Embedding → Ranking → Output writing

Final output written in output/challenge1b_output.json.

📦 Directory Structure Summary
lua
Copy
Edit
ADOBE_ROUND1B/
├── data/
│   └── train_data.json
├── model/
│   ├── models/ (pkl + SBERT weights)
│   └── tokenizer + config files
├── utils/
│   ├── pdf_parser.py
│   ├── embedding_utils.py
│   ├── predict_utils.py
│   ├── json_writer.py
│   └── ranking.py
├── input_pdf/
├── output/
├── main.py
├── train_classifier.py
├── predict.py
├── model_creation.py
├── Dockerfile
├── requirements.txt
└── README.md
🧪 Training Summary
200+ labeled training samples

Balanced class distribution between personas and jobs

LabelEncoder used to encode targets

Models trained in under 20s on CPU, saved using joblib

🚀 Performance
Average inference time per document: ~25–30 seconds

Memory usage: ~600MB (model + runtime)

Output quality: Highly relevant sections per persona-job intent

