import streamlit as st
from src.data_manager import load_data
from src.preprocessing import clean_text
from src.vectorizer import build_vocab
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

    vocab = build_vocab(df)
    
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
        label, score, df_res = predict(input_q, df, vocab)
        
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
            from src.preprocessing import to_lowercase, remove_punctuation_and_whitespace, count_words
            tokens = clean_text(input_q)
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

            # Step 2: Representation
            st.markdown("##### 2️⃣ Biểu diễn TF-IDF (Criteria 3.2.3)")
            from src.vectorizer import to_tfidf_vector, compute_idf
            idf_dict = compute_idf(df, vocab)
            vec = to_tfidf_vector(tokens, vocab, idf_dict)
            with st.expander("Xem Vector TF-IDF"):
                st.write(f"Vector {len(vocab)} chiều (Đã tính toán trọng số quan trọng của từ):")
                st.code(str([round(v, 4) for v in vec]))
                st.caption("TF-IDF = TF (Tần suất từ) * IDF (Tần suất nghịch của văn bản).")

            # Step 3: Probability/Scores
            st.markdown("##### 3️⃣ Tính toán điểm (Criteria 3.2.4)")
            st.write("Sử dụng thuật toán **Cosine Similarity** (Toán học cơ bản):")
            st.dataframe(
                df_res.sort_values("score", ascending=False).rename(columns={"label": "Nhóm chủ đề", "score": "Điểm tương đồng"}),
                use_container_width=True,
                hide_index=True
            )
