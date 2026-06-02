import streamlit as st
from src.data_manager import load_data
from src.preprocessing import clean_text
from src.vectorizer import build_vocab_and_pmi
from src.classifier import predict

def render_classify_page():
    st.markdown("<h1>🔍 Hệ thống Phân loại Thông minh</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style='color: #6B7280; font-size: 1.1em;'>
            Nhập yêu cầu của bạn bên dưới, trí tuệ nhân tạo sẽ tự động phân tích và đưa ra phòng ban xử lý phù hợp nhất.
        </p>
    """, unsafe_allow_html=True)
    
    df = load_data()
    if df.empty:
        st.warning("⚠️ **Dữ liệu huấn luyện đang trống!** Vui lòng chuyển sang tab 'Quản lý dữ liệu' để bắt đầu.")
        return

    # Thông số PMI (tương hỗ)
    pmi_min_count = 2
    pmi_threshold = 3.0
    vocab, pmi_pairs = build_vocab_and_pmi(df, min_count=pmi_min_count, pmi_threshold=pmi_threshold)
    
    # Hero Section for Input
    with st.container():
        st.markdown("### ✍️ Nhập thắc mắc của bạn")
        input_q = st.text_area(
            "Câu hỏi của bạn:", 
            placeholder="Ví dụ: Em muốn hỏi về thủ tục xin bảng điểm hoặc lịch thi học kỳ tới...",
            height=120,
            label_visibility="collapsed"
        )
        
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            clicked = st.button("🚀 Phân tích ngay", use_container_width=True)

    if clicked and input_q:
        label, score, df_res = predict(input_q, df, vocab, pmi_pairs=pmi_pairs)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔮 Kết quả Dự đoán")
        
        # Fancy result card using st.info/success with custom styling
        if score > 0.15:
            st.success(f"""
                ### ✅ Dự đoán: **{label}**
                Độ tin cậy: **{score:.2%}**
            """)
            
            # Nhận xét ngắn gọn (Phần 2 yêu cầu)
            st.markdown(f"**💡 Nhận xét:** Yêu cầu này có tính tương đồng cao với nhóm **{label}**. Bạn nên liên hệ bộ phận tương ứng để được hỗ trợ.")
        elif score > 0:
            st.warning(f"""
                ### ⚠️ Dự đoán: **{label}** (Độ tin cậy thấp)
                Độ tin cậy: **{score:.2%}**
            """)
            st.markdown("**💡 Nhận xét:** Kết quả có độ tin cậy chưa cao. Vui lòng kiểm tra lại cách đặt câu hỏi hoặc bổ sung thêm dữ liệu mẫu.")
        else:
            st.error("### ❌ Không thể xác định\nHệ thống không tìm thấy sự tương đồng với dữ liệu mẫu nào.")

        # Interactive Analysis Tabs
        tab1, tab2 = st.tabs(["📊 Biểu đồ so sánh", "🧬 Quy trình xử lý NLP"])
        
        with tab1:
            st.markdown("#### Tương quan giữa các chủ đề")
            st.bar_chart(df_res.set_index("label"), color="#3B82F6", use_container_width=True)
            
        with tab2:
            st.info("💡 Phần này hiển thị cách hệ thống thực hiện theo các tiêu chí chấm đồ án.")
            
            # Step 1: Preprocessing
            st.markdown("##### 1️⃣ Tiền xử lý (Criteria 3.2.2)")
            from src.preprocessing import to_lowercase, remove_punctuation_and_whitespace, count_words, explain_clean_text

            # Token hoá đúng pipeline PMI (unigram + PMI phrases)
            tokens = clean_text(
                input_q,
                word_ngram_max=1,
                use_pmi_phrases=True,
                pmi_pairs=pmi_pairs,
            )

            explain = explain_clean_text(
                input_q,
                word_ngram_max=1,
                use_stopwords=True,
                use_pmi_phrases=True,
                pmi_pairs=pmi_pairs,
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Viết thường:**")
                st.code(to_lowercase(input_q))
            with c2:
                st.write("**Xóa dấu câu & Khoảng trắng:**")
                st.code(remove_punctuation_and_whitespace(input_q))
            with c3:
                st.write("**Đếm số từ:**")
                st.metric("Tổng số từ", count_words(input_q))

            with st.expander("📌 Xem chi tiết tách từ (từng bước như tính tay)"):
                st.markdown("**Bước 1 — Lowercase**")
                st.code(explain["lower"])

                st.markdown("**Bước 2 — Chuẩn hoá số** (mọi số → `<num>`) ")
                st.code(explain["normalized_numbers"])

                st.markdown("**Bước 3 — Xoá dấu câu & chuẩn hoá khoảng trắng**")
                st.code(explain["removed_punctuation"])

                st.markdown("**Bước 4 — Tokenize (tách theo khoảng trắng)**")
                st.write(explain["tokens_raw"])

                st.markdown("**Bước 5 — Loại stopwords**")
                st.write(
                    {
                        "stopwords_count": explain["stopwords_count"],
                        "removed_stopwords": explain["removed_stopwords"],
                        "tokens_no_stopwords": explain["tokens_no_stopwords"],
                    }
                )

                st.markdown("**Bước 6 — Ghép cụm từ theo PMI (tương hỗ)**")
                st.write(
                    {
                        "pmi_matched_pairs": explain["pmi_matched_pairs"],
                        "pmi_added": explain["pmi_added"],
                        "pmi_merge_steps": explain.get("pmi_merge_steps", []),
                        "tokens_after_pmi": explain["tokens_after_pmi"],
                    }
                )

                st.markdown("**Cách tính PMI (độ tương hỗ) như tính tay**")
                st.latex(r"PMI(w_1,w_2) = \log_2\left(\frac{P(w_1,w_2)}{P(w_1)P(w_2)}\right)")
                st.caption(
                    "Trong code: P(w1,w2) tính theo tần suất bigram; P(w1), P(w2) tính theo tần suất unigram trong toàn bộ corpus. "
                    "Chọn phrase nếu c(w1,w2) >= min_count và PMI >= threshold."
                )

                st.write({"min_count": pmi_min_count, "threshold": pmi_threshold, "num_selected_pairs": len(pmi_pairs)})

                if explain["pmi_matched_pairs"]:
                    from src.vectorizer import compute_pmi_debug
                    import pandas as pd

                    debug = compute_pmi_debug(
                        df,
                        pairs=list(explain["pmi_matched_pairs"]),
                        min_count=pmi_min_count,
                        pmi_threshold=pmi_threshold,
                    )
                    rows = debug["rows"]
                    df_pmi = pd.DataFrame(rows)

                    # format PMI -inf -> None để dễ nhìn
                    def _fmt_pmi(x):
                        if x == float("-inf"):
                            return None
                        return float(x)

                    df_pmi["PMI"] = df_pmi["PMI"].map(_fmt_pmi)
                    st.dataframe(
                        df_pmi[[
                            "bigram",
                            "c(w1)",
                            "c(w2)",
                            "c(w1,w2)",
                            "P(w1)",
                            "P(w2)",
                            "P(w1,w2)",
                            "PMI",
                            "selected",
                        ]].sort_values(["selected", "PMI"], ascending=[False, False]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Giải thích cột: c(w1,w2)=số lần 2 từ đứng cạnh nhau trong corpus; "
                        "P(w1,w2)=c(w1,w2)/total_bigrams; P(w1)=c(w1)/total_unigrams."
                    )
                else:
                    st.warning("Không có cặp (w1,w2) nào trong câu khớp với danh sách PMI đã chọn.")

                st.markdown("**Kết quả tokens cuối cùng (dùng để tính TF‑IDF)**")
                st.write(explain["tokens_final"])

            # Step 2: Representation
            st.markdown("##### 2️⃣ Biểu diễn TF-IDF (Criteria 3.2.3)")
            from src.vectorizer import to_tfidf_vector, compute_idf, compute_tf
            idf_dict = compute_idf(df, vocab, use_pmi_phrases=True, pmi_pairs=pmi_pairs)
            vec = to_tfidf_vector(tokens, vocab, idf_dict)
            with st.expander("Xem Vector TF-IDF"):
                st.write(f"Vector {len(vocab)} chiều (Đã tính toán trọng số quan trọng của từ):")
                st.code(str([round(v, 4) for v in vec]))
                st.caption("TF-IDF = TF (Tần suất từ) * IDF (Tần suất nghịch của văn bản).")

            # Hiển thị phép tính chi tiết cho từng từ trong câu nhập
            with st.expander("Xem phép tính theo từng từ (TF, IDF, TF-IDF)"):
                import pandas as pd

                # Tokenize toàn bộ docs theo đúng pipeline để tính df_count chính xác
                tokenized_docs = [
                    clean_text(
                        t,
                        word_ngram_max=1,
                        use_pmi_phrases=True,
                        pmi_pairs=pmi_pairs,
                    )
                    for t in df["text"]
                ]
                N = len(tokenized_docs)
                doc_sets = [set(toks) for toks in tokenized_docs]

                tf_dict = compute_tf(tokens)
                token_counts = {}
                total_tokens = len(tokens)
                for t in tokens:
                    token_counts[t] = token_counts.get(t, 0) + 1

                rows = []
                for term, tf_val in sorted(tf_dict.items(), key=lambda x: (-x[1], x[0])):
                    idf_val = float(idf_dict.get(term, 0.0))
                    df_count = sum(1 for s in doc_sets if term in s)
                    rows.append(
                        {
                            "Từ/Cụm từ": term,
                            "Count": int(token_counts.get(term, 0)),
                            "Total": int(total_tokens),
                            "TF": float(tf_val),
                            "DF": int(df_count),
                            "N": int(N),
                            "IDF": idf_val,
                            "TF-IDF": float(tf_val) * idf_val,
                            "Công thức IDF": "log((N+1)/(DF+1)) + 1",
                        }
                    )

                if rows:
                    df_calc = pd.DataFrame(rows).sort_values("TF-IDF", ascending=False).reset_index(drop=True)
                    st.dataframe(df_calc, use_container_width=True, hide_index=True)
                    st.caption(
                        "Ghi chú: Token có thể là cụm từ dạng word1_word2 (do PMI ghép và bỏ từ đơn). "
                        "Bảng này chỉ liệt kê các token xuất hiện trong câu hỏi đầu vào (TF > 0)."
                    )
                else:
                    st.warning("Không có token nào sau tiền xử lý (có thể do stopwords lọc hết).")

            # Step 3: Probability/Scores
            st.markdown("##### 3️⃣ Tính toán điểm (Criteria 3.2.4)")
            st.write("Sử dụng thuật toán **Cosine Similarity** (Toán học cơ bản):")
            st.dataframe(
                df_res.sort_values("score", ascending=False).rename(columns={"label": "Nhóm chủ đề", "score": "Điểm tương đồng"}),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("🧮 Xem phép tính Cosine Similarity (tính tay theo từng nhãn)"):
                import math
                import pandas as pd
                from src.classifier import build_label_centroids_pmi
                from src.vectorizer import to_tfidf_vector

                # Dùng lại IDF theo đúng cách classifier tính (từ tokenized_docs PMI)
                idf_centroid, centroids = build_label_centroids_pmi(df, vocab, pmi_pairs=pmi_pairs)
                vec_in = to_tfidf_vector(tokens, vocab, idf_centroid)

                # Chuẩn bị map token -> index để lấy nhanh trọng số
                vocab_index = {w: i for i, w in enumerate(vocab)}
                present_terms = [t for t in dict.fromkeys(tokens) if t in vocab_index]

                mag_in = math.sqrt(sum(x * x for x in vec_in))
                st.markdown(
                    "Cosine(v,q) = (v·q) / (||v|| * ||q||), với v là centroid từng nhãn, q là câu hỏi."
                )
                st.write({"||q||": mag_in, "num_terms_in_query": len(present_terms)})

                # Tính chi tiết theo từng nhãn: dot product chỉ cần các term có trong query
                detail_rows = []
                for label_name, vec_centroid in centroids.items():
                    mag_c = math.sqrt(sum(x * x for x in vec_centroid))
                    dot = 0.0
                    for term in present_terms:
                        i = vocab_index[term]
                        dot += vec_in[i] * vec_centroid[i]
                    cosine = dot / (mag_in * mag_c) if mag_in * mag_c > 0 else 0.0
                    detail_rows.append(
                        {
                            "Nhóm": label_name,
                            "Dot(q·v)": dot,
                            "||q||": mag_in,
                            "||v||": mag_c,
                            "Cosine": cosine,
                        }
                    )

                df_cos = pd.DataFrame(detail_rows).sort_values("Cosine", ascending=False).reset_index(drop=True)
                st.dataframe(df_cos, use_container_width=True, hide_index=True)

                # Hiển thị breakdown theo nhãn dự đoán (top-1)
                best_label = df_cos.iloc[0]["Nhóm"] if not df_cos.empty else None
                if best_label is not None:
                    st.markdown(f"**Breakdown theo từng token cho nhãn dự đoán: {best_label}**")
                    vec_best = centroids[best_label]
                    rows = []
                    for term in present_terms:
                        i = vocab_index[term]
                        qv = vec_in[i]
                        cv = vec_best[i]
                        rows.append(
                            {
                                "Token": term,
                                "q_TFIDF": qv,
                                "centroid_TFIDF": cv,
                                "product": qv * cv,
                            }
                        )
                    df_break = pd.DataFrame(rows).sort_values("product", ascending=False).reset_index(drop=True)
                    st.dataframe(df_break, use_container_width=True, hide_index=True)
                    st.caption(
                        "Ghi chú: dot product = tổng(product) trên các token có trong câu hỏi. "
                        "||q|| và ||v|| tính trên toàn bộ chiều của vector." 
                    )
