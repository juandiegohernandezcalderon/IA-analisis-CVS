# utils.py
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calcula la similitud del coseno entre dos vectores.
    Los vectores deben ser listas o arrays numpy.
    """
    if not vec1 or not vec2:
        return 0.0 # O maneja el error como prefieras si un vector está vacío

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # Evitar división por cero si algún vector es cero

    return dot_product / (norm_vec1 * norm_vec2)