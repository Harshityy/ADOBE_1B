# utils/embedding_utils.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_encoder(model_path):
    return SentenceTransformer(model_path)

def encode_texts(text_list, model):
    return model.encode(text_list, convert_to_numpy=True)

def compute_similarity(vec1, vec2):
    return float(cosine_similarity([vec1], [vec2])[0][0])
