import math

def compute_vector_similarity(query_emb, mem_emb) -> float:
    """Cosine similarity. Returns 0.0 for missing/empty/zero-norm inputs."""
    if query_emb is None or mem_emb is None:
        return 0.0
    if len(query_emb) == 0 or len(mem_emb) == 0:
        return 0.0
    if len(query_emb) != len(mem_emb):
        return 0.0

    dot = sum(a * b for a, b in zip(query_emb, mem_emb))
    norm_a = math.sqrt(sum(a * a for a in query_emb))
    norm_b = math.sqrt(sum(b * b for b in mem_emb))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
