# utils/json_writer.py
from datetime import datetime

def write_output_json(documents, persona, job, top_sections, top_sentences, timestamp):
    return {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": timestamp
        },
        "extracted_sections": [
            {
                "document": section["document"],
                "page_number": section["page"],
                "section_title": section["text"],
                "importance_rank": i + 1
            }
            for i, section in enumerate(top_sections)
        ],
        "sub_section_analysis": [
            {
                "document": sentence["document"],
                "refined_text": sentence["refined_text"],
                "page_number": sentence["page_number"]
            }
            for sentence in top_sentences
        ]
    }
