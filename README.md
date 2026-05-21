# 🎓 Hệ thống Phân loại Câu hỏi Sinh viên (NLP Student Classifier)

Đồ án môn học Xử lý ngôn ngữ tự nhiên: Xây dựng hệ thống phân loại câu hỏi của sinh viên sử dụng phương pháp thống kê cơ bản.

## 🌟 Tính năng
- Tiền xử lý văn bản tiếng Việt (loại bỏ dấu câu, viết thường, loại bỏ từ dừng).
- Biểu diễn văn bản dưới dạng vector Bag-of-Words (BoW).
- Phân loại bằng cách tính độ tương đồng Cosine (Cosine Similarity) với tập dữ liệu huấn luyện.
- Giao diện trực quan với Streamlit.

## 📂 Cấu trúc thư mục
```text
NLP_STUDENT_CLASSIFIER/
├── .streamlit/                # Cấu hình giao diện Streamlit
├── data/                      # Quản lý dữ liệu
│   └── training_data.csv      # Cơ sở dữ liệu câu hỏi mẫu
├── src/                       # Mã nguồn xử lý logic
│   ├── preprocessing.py       # Tiền xử lý văn bản
│   ├── vectorizer.py          # Biểu diễn văn bản (BoW)
│   └── classifier.py          # Thuật toán phân loại (Cosine Sim)
├── ui/                        # Giao diện người dùng
│   ├── classify_page.py       # Trang dự đoán/phân loại
│   └── manage_data_page.py    # Trang quản lý dữ liệu
├── app.py                     # Entry point của ứng dụng
├── requirements.txt           # Danh sách thư viện
└── README.md                  # Tài liệu hướng dẫn
```

## 🚀 Hướng dẫn cài đặt và chạy

### 1. Cài đặt thư viện
Yêu cầu Python 3.8+.
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
 python -m streamlit run app.py

## 🛠 Công nghệ sử dụng
- **Ngôn ngữ:** Python
- **Giao diện:** Streamlit
- **Xử lý dữ liệu:** Pandas, Regex
- **Thuật toán:** Bag-of-Words, Cosine Similarity


## Dạng nhập văn bản vào 
Làm sao để đăng ký học phần? | Lịch học
Bao giờ có điểm thi kết thúc học phần? | Điểm số
Em muốn nộp học phí qua ngân hàng nào? | Học phí
