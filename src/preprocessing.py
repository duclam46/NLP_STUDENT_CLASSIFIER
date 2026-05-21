import re
import string

def to_lowercase(text):
    """Chuẩn hóa về chữ thường"""
    return str(text).lower()

def remove_punctuation_and_whitespace(text):
    """Loại bỏ dấu câu và khoảng trắng thừa"""
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Loại bỏ khoảng trắng thừa (giữa và 2 đầu)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Tách từ (Tokenize)"""
    return text.split()

def count_words(text):
    """Đếm số từ trong văn bản"""
    return len(tokenize(text))

def clean_text(text):
    """Hàm tổng hợp để xử lý toàn bộ các bước"""
    text = to_lowercase(text)
    text = remove_punctuation_and_whitespace(text)
    return tokenize(text)
