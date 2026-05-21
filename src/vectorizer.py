import math
from collections import Counter
from src.preprocessing import clean_text

def build_vocab(df):
    vocab = set()
    for text in df['text']:
        vocab.update(clean_text(text))
    return sorted(list(vocab))

def compute_tf(tokens):
    """Tính Term Frequency (TF)"""
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0: return {}
    return {word: count/total for word, count in counts.items()}

def compute_idf(df, vocab):
    """Tính Inverse Document Frequency (IDF)"""
    N = len(df)
    idf = {}
    for word in vocab:
        # Đếm số văn bản chứa từ 'word'
        count = sum(1 for text in df['text'] if word in clean_text(text))
        # Công thức log cơ bản (thêm 1 để tránh chia cho 0)
        idf[word] = math.log(N / (1 + count)) + 1
    return idf

def to_tfidf_vector(tokens, vocab, idf_dict):
    """Biểu diễn văn bản dưới dạng vector TF-IDF"""
    tf = compute_tf(tokens)
    vector = []
    for word in vocab:
        # TF-IDF = TF * IDF
        val = tf.get(word, 0) * idf_dict.get(word, 0)
        vector.append(val)
    return vector
