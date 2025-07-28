# utils/ranking.py
import nltk
from utils.embedding_utils import encode_texts, compute_similarity

def rank_sections(section_scores, top_k=5):
    return sorted(section_scores, key=lambda x: -x["similarity"])[:top_k]

def extract_top_sentences(top_sections, query_vec, model, top_n=5):
    results = []
    for sec in top_sections:
        sentences = nltk.sent_tokenize(sec["text"])
        embeddings = encode_texts(sentences, model)
        scores = [compute_similarity(query_vec, emb) for emb in embeddings]

        best_sentences = sorted(
            zip(sentences, scores),
            key=lambda x: -x[1]
        )[:top_n]

        for sent, score in best_sentences:
            results.append({
                "document": sec["document"],
                "refined_text": sent,
                "page_number": sec["page"]
            })

    return results
