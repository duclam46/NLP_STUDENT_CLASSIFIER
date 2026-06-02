import streamlit as st
import pandas as pd
from src.data_manager import load_data, save_data, load_stopwords_df, save_stopwords_df

def render_manage_data_page():
    st.markdown("<h1>⚙️ Trung tâm Quản lý Dữ liệu</h1>", unsafe_allow_html=True)
    
    df = load_data()
    
    # Overview Stats in a row
    st.markdown("### 📈 Tổng quan hệ thống")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Tổng mẫu tin", len(df), delta=None)
    with s2:
        num_labels = df['label'].nunique() if not df.empty else 0
        st.metric("Số lượng nhãn", num_labels)
    with s3:
        from src.preprocessing import count_words
        avg_len = df['text'].apply(count_words).mean() if not df.empty else 0
        st.metric("Độ dài TB", f"{avg_len:.1f} từ")
    with s4:
        st.metric("Trạng thái", "Sẵn sàng" if not df.empty else "Cần nạp dữ liệu")

    # Keyword Search Feature (Criteria Phần 1)
    st.markdown("---")
    st.markdown("### 🔍 Tìm kiếm từ khóa (Criteria Phần 1)")
    keyword = st.text_input("Nhập từ khóa cần tìm trong tập dữ liệu:", placeholder="Ví dụ: học phí, lịch học...")
    if keyword:
        from src.preprocessing import clean_text
        keyword_clean = keyword.lower().strip()
        # Hàm tìm từ khóa (Criteria yêu cầu viết hàm này)
        def find_by_keyword(dataframe, kw):
            return dataframe[dataframe['text'].str.lower().str.contains(kw)]
        
        search_results = find_by_keyword(df, keyword_clean)
        if not search_results.empty:
            st.write(f"Tìm thấy **{len(search_results)}** mẫu tin chứa từ khóa `{keyword}`:")
            st.dataframe(search_results, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Không tìm thấy mẫu tin nào chứa từ khóa `{keyword}`.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Use Tabs for different management styles
    tab_edit, tab_batch, tab_stopwords = st.tabs([
        "📑 Chỉnh sửa trực tiếp",
        "📥 Nhập hàng loạt (Batch)",
        "🧾 Stopwords",
    ])
    
    with tab_edit:
        st.markdown("#### 🛠 Danh sách dữ liệu huấn luyện")
        st.info("💡 Bạn có thể sửa trực tiếp trên bảng hoặc thêm dòng mới ở cuối.")
        
        edited_df = st.data_editor(
            df, 
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "text": st.column_config.TextColumn(
                    "💬 Câu hỏi mẫu", 
                    help="Nhập các câu hỏi thực tế của sinh viên",
                    required=True,
                    width="large"
                ),
                "label": st.column_config.SelectboxColumn(
                    "🏷️ Nhãn phân loại",
                    help="Nhóm chủ đề",
                    options=["Lịch học", "Điểm số", "Học phí", "Tài khoản", "Thủ tục hành chính", "Khác"],
                    required=True
                )
            },
            key="data_editor_main"
        )

        # Hiển thị số từ cho từng văn bản (yêu cầu Phần 1)
        from src.preprocessing import count_words
        preview_df = edited_df.copy()
        preview_df["text"] = preview_df["text"].fillna("")
        preview_df["word_count"] = preview_df["text"].apply(count_words)
        st.markdown("#### 🔢 Số từ trong từng mẫu")
        st.dataframe(
            preview_df[["text", "label", "word_count"]].rename(
                columns={"text": "Câu hỏi mẫu", "label": "Nhãn", "word_count": "Số từ"}
            ),
            use_container_width=True,
            hide_index=True
        )
        
        col_save, _ = st.columns([1, 4])
        with col_save:
            if st.button("💾 Lưu tất cả thay đổi", use_container_width=True):
                save_data(edited_df)
                st.success("Hệ thống đã cập nhật dữ liệu mới!")
                st.balloons()
                st.rerun()

    with tab_batch:
        st.markdown("#### 📥 Nhập nhanh dữ liệu hàng loạt")
        st.write("Dán danh sách câu hỏi theo định dạng: `Nội dung câu hỏi | Tên nhãn`")
        
        batch_text = st.text_area(
            "Nhập dữ liệu tại đây:",
            placeholder="Ví dụ:\nĐăng ký học phần ở đâu? | Lịch học\nXem điểm thi ở đâu? | Điểm số",
            height=250,
            label_visibility="collapsed"
        )
        
        col_import, _ = st.columns([1, 3])
        with col_import:
            if st.button("➕ Thực hiện Trích xuất", use_container_width=True):
                if batch_text:
                    new_rows = []
                    lines = batch_text.strip().split("\n")
                    for line in lines:
                        if "|" in line:
                            txt, lbl = line.split("|", 1)
                            new_rows.append({"text": txt.strip(), "label": lbl.strip()})
                    
                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        updated_df = pd.concat([df, new_df], ignore_index=True)
                        save_data(updated_df)
                        st.success(f"🚀 Thành công! Đã thêm {len(new_rows)} mẫu tin vào hệ thống.")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Định dạng không hợp lệ. Vui lòng kiểm tra dấu gạch đứng `|`.")
                else:
                    st.warning("Vui lòng dán dữ liệu vào ô trên.")

    with tab_stopwords:
        st.markdown("#### 🧾 Quản lý Stopwords")
        st.caption("Stopwords lưu ở dạng CSV (cột `word`). Có thể thêm/xoá dòng giống dữ liệu huấn luyện.")

        sw_df = load_stopwords_df()
        sw_df = sw_df[["word"]].copy() if "word" in sw_df.columns else pd.DataFrame({"word": []})
        sw_df["word"] = sw_df["word"].fillna("").astype(str)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Tổng stopwords", int((sw_df["word"].str.strip() != "").sum()))
        with m2:
            st.metric("Trạng thái", "Sẵn sàng")

        edited_sw = st.data_editor(
            sw_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "word": st.column_config.TextColumn(
                    "Stopword",
                    help="Nhập 1 từ/dạng token mỗi dòng (ví dụ: 'là', 'của', 'như')",
                    required=True,
                )
            },
            key="stopwords_editor",
        )

        col_norm, col_save_sw, _ = st.columns([1, 1, 3])
        with col_norm:
            if st.button("🧹 Chuẩn hóa", use_container_width=True):
                normalized = (
                    edited_sw["word"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )
                normalized = sorted({w for w in normalized.tolist() if w})
                st.session_state["_stopwords_df"] = pd.DataFrame({"word": normalized})
                st.rerun()

        if "_stopwords_df" in st.session_state:
            edited_sw = st.session_state.pop("_stopwords_df")
            st.info("Đã chuẩn hóa stopwords (lower + unique + sort). Bấm Lưu để ghi ra file.")
            st.dataframe(edited_sw, use_container_width=True, hide_index=True)

        with col_save_sw:
            if st.button("💾 Lưu stopwords", use_container_width=True):
                out = edited_sw.copy()
                out["word"] = out["word"].fillna("").astype(str).str.strip().str.lower()
                out = out[out["word"] != ""]
                out = out.drop_duplicates(subset=["word"]).sort_values("word").reset_index(drop=True)
                save_stopwords_df(out)
                from src.preprocessing import reset_stopwords_cache
                reset_stopwords_cache()
                st.success("Đã lưu stopwords CSV và áp dụng cho tiền xử lý!")
