import math
import pandas as pd
from src.preprocessing import clean_text
from src.vectorizer import to_tfidf_vector, compute_idf_from_docs

def cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0


def _mean_vectors(vectors):
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    n = float(len(vectors))
    return [x / n for x in out]


def build_label_centroids(df, vocab):
    """Tạo centroid TF-IDF cho từng nhãn để dự đoán ổn định hơn."""
    tokenized_docs = [clean_text(t) for t in df['text']]
    idf_dict = compute_idf_from_docs(tokenized_docs, vocab)

    label_to_vectors = {}
    for tokens, (_, row) in zip(tokenized_docs, df.iterrows()):
        vec = to_tfidf_vector(tokens, vocab, idf_dict)
        label_to_vectors.setdefault(row['label'], []).append(vec)

    centroids = {label: _mean_vectors(vs) for label, vs in label_to_vectors.items()}
    return idf_dict, centroids


def build_label_centroids_pmi(df, vocab, pmi_pairs: set[tuple[str, str]] | None = None):
    """Centroid theo nhãn, token hoá có thể dùng PMI phrases."""
    tokenized_docs = [
        clean_text(
            t,
            word_ngram_max=1,
            use_pmi_phrases=bool(pmi_pairs),
            pmi_pairs=pmi_pairs,
        )
        for t in df['text']
    ]
    idf_dict = compute_idf_from_docs(tokenized_docs, vocab)

    label_to_vectors = {}
    for tokens, (_, row) in zip(tokenized_docs, df.iterrows()):
        vec = to_tfidf_vector(tokens, vocab, idf_dict)
        label_to_vectors.setdefault(row['label'], []).append(vec)

    centroids = {label: _mean_vectors(vs) for label, vs in label_to_vectors.items()}
    return idf_dict, centroids

def predict(input_text, df, vocab, pmi_pairs: set[tuple[str, str]] | None = None):
    # Huấn luyện nhanh centroid theo nhãn
    if pmi_pairs:
        idf_dict, centroids = build_label_centroids_pmi(df, vocab, pmi_pairs=pmi_pairs)
        tokens_in = clean_text(
            input_text,
            word_ngram_max=1,
            use_pmi_phrases=True,
            pmi_pairs=pmi_pairs,
        )
    else:
        idf_dict, centroids = build_label_centroids(df, vocab)
        tokens_in = clean_text(input_text)
    vec_in = to_tfidf_vector(tokens_in, vocab, idf_dict)

    results = []
    for label, vec_centroid in centroids.items():
        sim = cosine_sim(vec_in, vec_centroid)
        results.append({"label": label, "score": sim})

    df_res = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    best_match = df_res.iloc[0] if not df_res.empty else pd.Series({"label": "Khác", "score": 0})
    return best_match['label'], float(best_match['score']), df_res
