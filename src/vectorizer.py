import math
from collections import Counter
from src.preprocessing import clean_text


def compute_pmi_pairs(
    df,
    *,
    use_stopwords: bool = True,
    min_count: int = 2,
    pmi_threshold: float = 3.0,
):
    """Chọn các cặp từ (bigram) có PMI cao từ dữ liệu huấn luyện.

    PMI(w1,w2) = log2( P(w1 w2) / (P(w1)*P(w2)) )
    - min_count: số lần xuất hiện tối thiểu của bigram trong corpus
    - pmi_threshold: ngưỡng PMI để coi là cụm từ có nghĩa
    """
    tokenized_docs = [
        clean_text(t, use_stopwords=use_stopwords, word_ngram_max=1, use_pmi_phrases=False)
        for t in df["text"]
    ]

    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[tuple[str, str]] = Counter()
    total_unigrams = 0
    total_bigrams = 0

    for tokens in tokenized_docs:
        tokens = list(tokens)
        unigram_counts.update(tokens)
        total_unigrams += len(tokens)
        for i in range(len(tokens) - 1):
            bigram_counts[(tokens[i], tokens[i + 1])] += 1
        total_bigrams += max(len(tokens) - 1, 0)

    if total_unigrams == 0 or total_bigrams == 0:
        return set()

    pairs: set[tuple[str, str]] = set()
    for (w1, w2), c12 in bigram_counts.items():
        if c12 < min_count:
            continue

        p12 = c12 / total_bigrams
        p1 = unigram_counts[w1] / total_unigrams
        p2 = unigram_counts[w2] / total_unigrams
        denom = p1 * p2
        if denom <= 0:
            continue

        pmi = math.log2(p12 / denom)
        if pmi >= pmi_threshold:
            pairs.add((w1, w2))

    return pairs


def compute_pmi_debug(
    df,
    pairs: list[tuple[str, str]],
    *,
    use_stopwords: bool = True,
    min_count: int = 2,
    pmi_threshold: float = 3.0,
):
    """Tính PMI chi tiết cho một danh sách cặp (w1,w2) để giải thích (tính tay).

    Trả về:
    - totals: tổng unigram/bigram trong corpus
    - rows: list dict gồm counts, P(), PMI, và cờ selected theo (min_count, pmi_threshold)
    """
    tokenized_docs = [
        clean_text(t, use_stopwords=use_stopwords, word_ngram_max=1, use_pmi_phrases=False)
        for t in df["text"]
    ]

    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[tuple[str, str]] = Counter()
    total_unigrams = 0
    total_bigrams = 0

    for tokens in tokenized_docs:
        tokens = list(tokens)
        unigram_counts.update(tokens)
        total_unigrams += len(tokens)
        for i in range(len(tokens) - 1):
            bigram_counts[(tokens[i], tokens[i + 1])] += 1
        total_bigrams += max(len(tokens) - 1, 0)

    rows = []
    for w1, w2 in pairs:
        c1 = int(unigram_counts.get(w1, 0))
        c2 = int(unigram_counts.get(w2, 0))
        c12 = int(bigram_counts.get((w1, w2), 0))

        p1 = (c1 / total_unigrams) if total_unigrams else 0.0
        p2 = (c2 / total_unigrams) if total_unigrams else 0.0
        p12 = (c12 / total_bigrams) if total_bigrams else 0.0

        denom = p1 * p2
        pmi = math.log2(p12 / denom) if (p12 > 0 and denom > 0) else float("-inf")
        selected = (c12 >= min_count) and (pmi != float("-inf")) and (pmi >= pmi_threshold)

        rows.append(
            {
                "w1": w1,
                "w2": w2,
                "bigram": f"{w1}_{w2}",
                "c(w1)": c1,
                "c(w2)": c2,
                "c(w1,w2)": c12,
                "total_unigrams": int(total_unigrams),
                "total_bigrams": int(total_bigrams),
                "P(w1)": p1,
                "P(w2)": p2,
                "P(w1,w2)": p12,
                "PMI": pmi,
                "min_count": int(min_count),
                "threshold": float(pmi_threshold),
                "selected": bool(selected),
            }
        )

    return {
        "total_unigrams": int(total_unigrams),
        "total_bigrams": int(total_bigrams),
        "rows": rows,
    }


def build_vocab_and_pmi(
    df,
    *,
    use_stopwords: bool = True,
    min_count: int = 2,
    pmi_threshold: float = 3.0,
):
    """Build vocab (unigram + PMI-selected bigram phrases) và trả về pmi_pairs."""
    pmi_pairs = compute_pmi_pairs(
        df,
        use_stopwords=use_stopwords,
        min_count=min_count,
        pmi_threshold=pmi_threshold,
    )

    vocab = set()
    for text in df["text"]:
        vocab.update(
            clean_text(
                text,
                use_stopwords=use_stopwords,
                word_ngram_max=1,
                use_pmi_phrases=True,
                pmi_pairs=pmi_pairs,
            )
        )
    return sorted(list(vocab)), pmi_pairs

def build_vocab(
    df,
    *,
    use_stopwords: bool = True,
    word_ngram_max: int = 2,
    use_char_ngrams: bool = False,
    char_ngram_range: tuple[int, int] = (3, 5),
    # PMI phrase options (nếu có pmi_pairs, nên đặt word_ngram_max=1 để tránh bigram đại trà)
    use_pmi_phrases: bool = False,
    pmi_pairs: set[tuple[str, str]] | None = None,
):
    """Xây dựng vocabulary từ dữ liệu huấn luyện.

    Mặc định gồm unigram + bigram và có lọc stopwords.
    """
    vocab = set()
    for text in df['text']:
        vocab.update(
            clean_text(
                text,
                use_stopwords=use_stopwords,
                word_ngram_max=word_ngram_max,
                use_pmi_phrases=use_pmi_phrases,
                pmi_pairs=pmi_pairs,
                use_char_ngrams=use_char_ngrams,
                char_ngram_range=char_ngram_range,
            )
        )
    return sorted(list(vocab))

def compute_tf(tokens):
    """Tính Term Frequency (TF)"""
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0: return {}
    return {word: count/total for word, count in counts.items()}

def compute_idf_from_docs(tokenized_docs, vocab):
    """Tính IDF từ danh sách văn bản đã tokenize (tối ưu hơn)."""
    N = len(tokenized_docs)
    if N == 0:
        return {word: 0.0 for word in vocab}

    doc_sets = [set(tokens) for tokens in tokenized_docs]
    idf = {}
    for word in vocab:
        df_count = sum(1 for s in doc_sets if word in s)
        idf[word] = math.log((N + 1) / (df_count + 1)) + 1
    return idf

def compute_idf(
    df,
    vocab,
    *,
    use_stopwords: bool = True,
    use_pmi_phrases: bool = False,
    pmi_pairs: set[tuple[str, str]] | None = None,
):
    """Tính Inverse Document Frequency (IDF)"""
    tokenized_docs = [
        clean_text(
            t,
            use_stopwords=use_stopwords,
            word_ngram_max=1 if use_pmi_phrases else 2,
            use_pmi_phrases=use_pmi_phrases,
            pmi_pairs=pmi_pairs,
        )
        for t in df['text']
    ]
    return compute_idf_from_docs(tokenized_docs, vocab)

def to_tfidf_vector(tokens, vocab, idf_dict):
    """Biểu diễn văn bản dưới dạng vector TF-IDF"""
    tf = compute_tf(tokens)
    vector = []
    for word in vocab:
        # TF-IDF = TF * IDF
        val = tf.get(word, 0) * idf_dict.get(word, 0)
        vector.append(val)
    return vector
