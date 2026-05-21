import math
import pandas as pd
from src.preprocessing import clean_text
from src.vectorizer import to_tfidf_vector, compute_idf

def cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0

def predict(input_text, df, vocab):
    # Tính IDF cho toàn bộ tập dữ liệu huấn luyện
    idf_dict = compute_idf(df, vocab)
    
    # Biểu diễn văn bản đầu vào dưới dạng vector TF-IDF
    tokens_in = clean_text(input_text)
    vec_in = to_tfidf_vector(tokens_in, vocab, idf_dict)
    
    results = []
    
    for _, row in df.iterrows():
        # Biểu diễn văn bản mẫu dưới dạng vector TF-IDF
        tokens_train = clean_text(row['text'])
        vec_train = to_tfidf_vector(tokens_train, vocab, idf_dict)
        
        sim = cosine_sim(vec_in, vec_train)
        results.append({"label": row['label'], "score": sim})
        
    df_res = pd.DataFrame(results).groupby("label")["score"].max().reset_index()
    best_match = df_res.loc[df_res['score'].idxmax()]
    
    return best_match['label'], best_match['score'], df_res
