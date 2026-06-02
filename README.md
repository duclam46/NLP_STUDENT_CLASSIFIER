# 🎓 Hệ thống Phân loại Câu hỏi Sinh viên (NLP Student Classifier)

Đồ án môn học Xử lý ngôn ngữ tự nhiên: xây dựng hệ thống phân loại câu hỏi của sinh viên bằng các kỹ thuật thống kê cơ bản.

Trọng tâm của dự án là một pipeline rõ ràng và “giải thích được”:
**Tiền xử lý → (tùy chọn) ghép cụm từ bằng PMI → TF‑IDF → centroid theo nhãn → Cosine Similarity**.

---

## 0) Mình đã sử dụng những phương pháp nào để làm hệ thống này?

Mục tiêu của đồ án là **phân loại câu hỏi sinh viên** bằng các kỹ thuật thống kê cơ bản, dễ “giải thích” và có thể trình bày lại theo từng bước. Vì vậy, mình chọn các phương pháp sau (đều đã được cài đặt/triển khai trong code):

### 0.1. Tiền xử lý văn bản tiếng Việt (rule-based)
Mình xử lý văn bản theo hướng đơn giản, ổn định:
- Chuẩn hoá Unicode (NFKC) để hạn chế lỗi ký tự đặc biệt
- Chuyển về chữ thường
- Chuẩn hoá số về token `<num>` để giảm nhiễu
- Loại dấu câu/ký tự lạ và chuẩn hoá khoảng trắng
- Tách token bằng khoảng trắng (`split()`)
- Loại stopwords từ file CSV (tự quản lý trong UI)

Lý do chọn: dữ liệu ngắn (câu hỏi), cần pipeline rõ ràng và dễ mô tả, không phụ thuộc thư viện NLP nặng.

### 0.2. Tạo cụm từ (phrase) bằng PMI thay vì bigram “đại trà”
Thay vì sinh mọi bigram có thể có (dễ nhiễu), mình dùng **PMI** để chọn các cặp từ đi cùng nhau “có nghĩa” trong dữ liệu huấn luyện (ví dụ: `học_phí`, `lịch_học`).

Lý do chọn: giúp biểu diễn cụm từ tốt hơn khi dataset nhỏ và câu hỏi hay có “cụm cố định”.

### 0.3. Biểu diễn văn bản bằng TF‑IDF (tự cài đặt)
Mình dùng TF‑IDF để biến danh sách token thành vector số:
- TF phản ánh mức độ xuất hiện trong câu
- IDF giảm trọng số các token quá phổ biến trong toàn bộ tập huấn luyện

Lý do chọn: đơn giản, hiệu quả cho bài toán phân loại chủ đề với dữ liệu dạng câu ngắn.

### 0.4. Phân loại bằng Cosine Similarity + centroid theo nhãn
Thay vì so sánh với từng câu mẫu, mình tính **centroid vector TF‑IDF** cho mỗi nhãn (trung bình các vector của nhãn đó), rồi so input bằng **Cosine Similarity**.

Lý do chọn: ổn định hơn với dữ liệu ít, dễ giải thích, không cần huấn luyện mô hình ML phức tạp.

### 0.5. Xây dựng giao diện và quy trình nhập dữ liệu
Mình dùng Streamlit để:
- Nhập câu hỏi và xem kết quả phân loại + bảng điểm theo từng nhãn
- Hiển thị “quy trình xử lý NLP” (giải thích từng bước tiền xử lý, TF‑IDF, cosine)
- Cho phép quản lý dữ liệu huấn luyện và stopwords trực tiếp trong UI (sửa bảng/batch import)

### 0.6. Kiểm tra nhanh (smoke test) và sinh thêm dữ liệu
- Có smoke test chạy end‑to‑end để đảm bảo pipeline hoạt động
- Có script sinh thêm dữ liệu theo template để tăng độ đa dạng câu hỏi

---

## 1) Tổng quan nhanh

### Các thành phần chính
- **Tiền xử lý**: [src/preprocessing.py](src/preprocessing.py)
- **Vector hoá TF‑IDF**: [src/vectorizer.py](src/vectorizer.py)
- **Phân loại (centroid + cosine)**: [src/classifier.py](src/classifier.py)
- **Quản lý dữ liệu (CSV)**: [src/data_manager.py](src/data_manager.py)
- **Giao diện Streamlit**: [app.py](app.py), [ui/classify_page.py](ui/classify_page.py), [ui/manage_data_page.py](ui/manage_data_page.py)

### Sơ đồ pipeline
```mermaid
flowchart LR
  A[training_data.csv
  (text,label)] --> B[Tiền xử lý
  clean_text()]
  B --> C[Chọn cụm từ bằng PMI
  (từ corpus)]
  C --> D[Xây vocab
  (unigram + phrase)]
  D --> E[TF-IDF
  (TF * IDF)]
  E --> F[Centroid theo nhãn
  (mean vector)]
  G[Câu hỏi mới] --> B
  B --> H[TF-IDF câu hỏi]
  H --> I[Cosine(q, centroid_label)]
  I --> J[Nhãn có điểm cao nhất]
```

---

## 2) Cài đặt & chạy ứng dụng

### Yêu cầu
- Python **3.8+**

### Cài thư viện
```bash
pip install -r requirements.txt
```

### Chạy Streamlit
```bash
python -m streamlit run app.py
```

Nếu chạy trên Windows và gặp lỗi font/Unicode khi in ra console, ưu tiên chạy trong VS Code Terminal (PowerShell) hoặc Windows Terminal.

---

## 3) Dữ liệu đầu vào (CSV) — bắt buộc đúng schema

### 3.1. Dữ liệu huấn luyện: data/training_data.csv
File [data/training_data.csv](data/training_data.csv) phải có **2 cột**:

| Cột | Ý nghĩa |
|---|---|
| `text` | Câu hỏi mẫu (văn bản tiếng Việt) |
| `label` | Nhãn/chủ đề (ví dụ: Lịch học, Điểm số, Học phí, …) |

Ví dụ (CSV):
```csv
text,label
Làm sao để đăng ký học phần,Lịch học
Bao giờ có điểm thi kết thúc học phần,Điểm số
Em muốn nộp học phí qua ngân hàng nào,Học phí
```

Gợi ý: dữ liệu càng đa dạng (nhiều cách diễn đạt) thì kết quả càng ổn định.

### 3.2. Stopwords: data/stopwords.csv
File [data/stopwords.csv](data/stopwords.csv) là CSV có cột `word`.

| Cột | Ý nghĩa |
|---|---|
| `word` | 1 stopword mỗi dòng (ví dụ: “là”, “của”, “như”, …) |

Stopwords sẽ được **lowercase** và lọc trong bước tiền xử lý.

---

## 4) Tiền xử lý văn bản (Preprocessing) — làm gì, theo thứ tự nào?

Tất cả logic tập trung ở hàm `clean_text()` trong [src/preprocessing.py](src/preprocessing.py).

### 4.1. Pipeline chuẩn (đúng với app)
Trong giao diện phân loại, hệ thống đang dùng chế độ:
- **unigram + ghép cụm từ bằng PMI** (không dùng bigram đại trà)

Các bước (theo đúng thứ tự):

1. **Lowercase**: chuyển về chữ thường
2. **Chuẩn hoá số**: mọi số được thay bằng token chung `<num>`
	- Ví dụ: “học phí 2025” → “học phí `<num>`”
3. **Chuẩn hoá Unicode + xoá dấu câu + chuẩn hoá khoảng trắng**
4. **Tokenize**: tách từ đơn giản bằng khoảng trắng (`split()`)
5. **Loại stopwords** (nếu bật)
6. **Ghép cụm từ theo PMI** (nếu có danh sách cặp từ PMI)
	- Quy tắc greedy trái → phải:
	  - Nếu `(w_i, w_{i+1})` nằm trong `pmi_pairs` → tạo token `w_i_w_{i+1}` và bỏ qua `w_{i+1}`
	  - Nếu không → giữ `w_i`
7. **Kết quả**: danh sách token cuối cùng dùng để tính TF‑IDF

### 4.2. Tuỳ chọn khác (có trong code)
Ngoài chế độ PMI, `clean_text()` còn hỗ trợ:
- **Word n‑grams đại trà**: thêm token dạng `w1_w2`… theo `word_ngram_max` (mặc định 2 = unigram + bigram)
- **Char n‑grams**: tạo n‑grams ký tự (mặc định 3–5) — nên cân nhắc vì dễ “nổ vocab” khi dữ liệu lớn

Lưu ý: app đang ưu tiên PMI vì “chọn lọc” bigram có nghĩa hơn so với bigram tạo đại trà.

---

## 5) Ghép cụm từ bằng PMI (Pointwise Mutual Information)

Phần này nằm trong [src/vectorizer.py](src/vectorizer.py) với các hàm `compute_pmi_pairs()` / `build_vocab_and_pmi()`.

### 5.1. PMI là gì?
PMI cho bigram $(w_1, w_2)$:
$$
PMI(w_1,w_2) = \log_2\left(\frac{P(w_1,w_2)}{P(w_1)P(w_2)}\right)
$$

Trong code:
- $P(w_1,w_2)$ ước lượng bằng tần suất 2 từ đứng cạnh nhau (bigram) trong toàn corpus
- $P(w_1)$, $P(w_2)$ ước lượng bằng tần suất unigram trong toàn corpus

### 5.2. Tiêu chí chọn phrase
Trong UI phân loại ([ui/classify_page.py](ui/classify_page.py)) đang dùng:
- `min_count = 2`
- `pmi_threshold = 3.0`

Chỉ các cặp có `c(w1,w2) >= min_count` và `PMI >= threshold` mới được chọn vào `pmi_pairs`.

---

## 6) Vector hoá TF‑IDF (tự cài đặt)

Phần TF‑IDF nằm trong [src/vectorizer.py](src/vectorizer.py).

### 6.1. TF (Term Frequency)
Với danh sách token của một văn bản:
$$
TF(t) = \frac{\text{count}(t)}{\text{total_tokens}}
$$

### 6.2. IDF (Inverse Document Frequency)
IDF được tính theo công thức (có smoothing):
$$
IDF(t) = \log\left(\frac{N + 1}{DF(t) + 1}\right) + 1
$$

Trong đó:
- $N$ = số văn bản huấn luyện
- $DF(t)$ = số văn bản có chứa token $t$

### 6.3. TF‑IDF
$$
TF\text{-}IDF(t) = TF(t) \times IDF(t)
$$

Vector TF‑IDF có số chiều bằng kích thước vocabulary (`len(vocab)`).

---

## 7) Phân loại: centroid theo nhãn + Cosine Similarity

Thuật toán nằm trong [src/classifier.py](src/classifier.py).

### 7.1. Vì sao dùng centroid theo nhãn?
Thay vì so input với từng câu mẫu, ta tính **centroid** cho mỗi nhãn (trung bình vector TF‑IDF của các câu thuộc nhãn đó). Cách này thường ổn định hơn khi dữ liệu ít.

Với nhãn $y$ có các vector $v_1, v_2, ..., v_k$:
$$
centroid_y = \frac{1}{k}\sum_{i=1}^k v_i
$$

### 7.2. Cosine Similarity
Với vector câu hỏi $q$ và centroid $c$:
$$
\cos(q,c) = \frac{q \cdot c}{\|q\|\,\|c\|}
$$

Hệ thống tính cosine cho mọi nhãn và chọn nhãn có điểm cao nhất.

### 7.3. Diễn giải “độ tin cậy” trong UI
Trong UI phân loại, hệ thống hiển thị:
- **Tốt** nếu `score > 0.15`
- **Thấp** nếu `0 < score <= 0.15`
- **Không xác định** nếu `score == 0`

Các ngưỡng này là heuristic để trình bày; có thể thay đổi tuỳ dữ liệu.

---

## 8) Cách sử dụng giao diện (Streamlit)

### 8.1. Trang “🔍 Phân loại câu hỏi”
1. Nhập 1 câu hỏi tự do
2. Bấm **“Phân tích ngay”**
3. Xem:
	- Nhãn dự đoán + điểm tương đồng
	- Biểu đồ so sánh điểm giữa các nhãn
	- Tab “Quy trình xử lý NLP” để xem từng bước tiền xử lý, TF‑IDF, cosine

### 8.2. Trang “⚙️ Quản lý dữ liệu”
Gồm 3 tab:

1. **Chỉnh sửa trực tiếp**: sửa bảng `text,label` và bấm Lưu
2. **Nhập hàng loạt (Batch)**: dán nhiều dòng theo định dạng:
	```text
	Nội dung câu hỏi | Tên nhãn
	```
	Ví dụ:
	```text
	Làm sao để đăng ký học phần? | Lịch học
	Bao giờ có điểm thi kết thúc học phần? | Điểm số
	Em muốn nộp học phí qua ngân hàng nào? | Học phí
	```
3. **Stopwords**: thêm/xoá stopwords và bấm Lưu để áp dụng cho tiền xử lý

---

## 9) Script mở rộng dữ liệu (tạo thêm câu mẫu)

Có sẵn script [scripts/expand_training_data.py](scripts/expand_training_data.py) để sinh thêm dữ liệu theo template.

Ví dụ thêm 200 dòng vào `data/training_data.csv`:
```bash
python scripts/expand_training_data.py --add 200
```

Tham số:
- `--csv` (mặc định `data/training_data.csv`)
- `--add` số dòng muốn thêm
- `--seed` để tái lập kết quả sinh

---

## 10) Kiểm tra nhanh (smoke test)

Chạy kiểm thử nhanh end‑to‑end:
```bash
python tests/smoke_test.py
```

Smoke test sẽ kiểm tra:
- Stopwords có được load và không rỗng
- Hành vi unigram+bigram mặc định của `clean_text()`
- Dự đoán end‑to‑end trên một vài câu mẫu

---

## 11) Ghi chú triển khai

- Dự án cố tình **không dùng thư viện ML nặng** (scikit‑learn, fastText, …) để đáp ứng tiêu chí “tự cài đặt” và dễ giải thích.
- Vì TF‑IDF/centroid được “tính nhanh” từ CSV khi dự đoán, nên khi dữ liệu tăng lớn bạn có thể cân nhắc cache mô hình (ngoài phạm vi đồ án hiện tại).
