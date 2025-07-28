import os
import fitz  # PyMuPDF

def extract_blocks_from_pdfs(folder_path, input_files):
    documents = []

    for filename in input_files:
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            print(f"❌ Skipping missing file: {file_path}")
            continue

        doc = fitz.open(file_path)
        blocks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                blocks.append({
                    "page": page_num + 1,
                    "text": text.strip()
                })

        documents.append({
            "name": filename,
            "blocks": blocks
        })

    return documents
