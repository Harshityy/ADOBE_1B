Adobe Round 1B — Persona-Driven Document Intelligence
This repository contains the complete pipeline for solving Adobe Round 1B challenge:
"Extract and prioritize sections from PDFs based on a given persona and job-to-be-done."

📌 Works offline, fully CPU-compatible, model size under 1GB, and supports end-to-end prediction: from PDF input → persona/job classification → relevant section extraction.

📁 Folder Structure
bash
Copy
Edit
ADOBE_ROUND1B/
│
├── data/
│   ├── input_pdf/                     # Raw PDF inputs
│   └── train_data.json               # Labeled data for training
│
├── model/
│   ├── models/
│   │   ├── persona_classifier.pkl
│   │   ├── job_classifier.pkl
│   │   ├── persona_encoder.pkl
│   │   ├── job_encoder.pkl
│   │   └── model.safetensors         # SentenceTransformer model weights
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── ...
│
├── output/
│   └── challenge1b_output.json       # Final structured output
│
├── utils/
│   ├── embedding_utils.py
│   ├── json_writer.py
│   ├── pdf_parser.py
│   ├── predict_utils.py
│   └── ranking.py
│
├── train_classifier.py              # Classifier training script
├── model_creation.py                # Embedding model loading + sentence scoring
├── predict.py                       # Final inference pipeline
├── main.py                          # Run end-to-end pipeline
├── Dockerfile                       # For offline/CPU Docker deployment
├── requirements.txt                 # All dependencies
├── approach_explanation.md          # Detailed write-up of approach
└── README.md                        # You're here!
🚀 Pipeline Overview
PDF → Text
Uses pdf_parser.py to extract clean text blocks from raw PDFs.

Persona + Job Classification
train_classifier.py trains classifiers using TF-IDF + Logistic Regression to predict:

persona: Who is reading this doc?

job: What is the user trying to achieve?

Embedding & Section Ranking

Converts document sections to embeddings using a SentenceTransformer model.

Compares cosine similarity with a reference query derived from (persona + job).

Ranks sections by semantic relevance.

Final Output
Writes top-ranked, relevant sections to challenge1b_output.json in structured JSON format.

🧪 Training Instructions
Train persona/job classifiers:

bash
Copy
Edit
python train_classifier.py
If needed, update the embedding model in model_creation.py
Currently using a pre-downloaded model for offline use.

🧠 Inference Instructions
bash
Copy
Edit
python main.py --input data/input_pdf/sample.pdf
This:

Extracts text from the PDF.

Classifies persona and job.

Embeds and ranks sections.

Outputs final result to output/challenge1b_output.json

📦 Docker (CPU-Only, Offline)
To run in a constrained environment:

bash
Copy
Edit
docker build -t adobe1b .
docker run --rm -v $(pwd):/app adobe1b
✅ Output Format
json
Copy
Edit
{
  "persona": "Undergraduate Chemistry Student",
  "job": "Understand reaction mechanisms",
  "sections": [
    {
      "title": "Reaction Kinetics",
      "content": "Reaction kinetics is the study of the speed of chemical reactions...",
      "score": 0.92
    },
    ...
  ]
}
💡 Features
✅ CPU-only model support (no GPU needed)

✅ < 1GB total model size

✅ Fast inference (< 60 seconds)

✅ Clean codebase with utils separated

✅ Easy to extend for other personas/jobs

📚 Tech Stack
Python 3.10+

scikit-learn (for classification)

sentence-transformers (for semantic embeddings)

PyMuPDF (for PDF parsing)

joblib, numpy, json, argparse

Docker (for offline containerized runs)

📖 approach_explanation.md
For detailed method, reasoning, model selection, ranking logic, limitations, etc., check:
approach_explanation.md

🙋 Author
Built with ❤️ by Harshit Srivastava
For: Adobe Document Intelligence — Challenge Round 1B