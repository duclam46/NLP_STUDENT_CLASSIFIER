import re
import string
import unicodedata
from pathlib import Path
import pandas as pd

_STOPWORDS_CACHE = None


def _project_root() -> Path:
    # src/preprocessing.py -> project root
    return Path(__file__).resolve().parents[1]


def load_stopwords(stopwords_path: str | None = None) -> set[str]:
    """Load stopwords from CSV (data/stopwords.csv) or a custom path."""
    global _STOPWORDS_CACHE
    if stopwords_path is None and _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    if stopwords_path:
        path = Path(stopwords_path)
        if not path.exists():
            words: set[str] = set()
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, encoding="utf-8")
            col = "word" if "word" in df.columns else df.columns[0]
            words = set(str(x).strip().lower() for x in df[col].fillna("") if str(x).strip())
        else:
            words = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                w = line.strip().lower()
                if w:
                    words.add(w)
    else:
        from src.data_manager import load_stopwords_df
        df = load_stopwords_df()
        words = set(str(x).strip().lower() for x in df["word"].fillna("") if str(x).strip())

    if stopwords_path is None:
        _STOPWORDS_CACHE = words
    return words


def reset_stopwords_cache():
    """Reset cache stopwords (dùng sau khi cập nhật file stopwords từ UI)."""
    global _STOPWORDS_CACHE
    _STOPWORDS_CACHE = None


def normalize_unicode(text):
    """Chuẩn hóa Unicode (NFKC) để xử lý dấu câu/ký tự đặc biệt ổn định hơn."""
    return unicodedata.normalize("NFKC", str(text))

def to_lowercase(text):
    """Chuẩn hóa về chữ thường"""
    return str(text).lower()

def remove_punctuation_and_whitespace(text):
    """Loại bỏ dấu câu và khoảng trắng thừa"""
    text = normalize_unicode(text)

    # Loại bỏ dấu câu ASCII cơ bản
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Loại bỏ các ký tự không phải chữ/số/khoảng trắng (bao gồm nhiều dấu câu Unicode)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Loại bỏ khoảng trắng thừa (giữa và 2 đầu)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_numbers(text):
    """Chuẩn hóa chuỗi số về một token chung để giảm nhiễu."""
    text = str(text)
    return re.sub(r"\d+([\.,]\d+)?", " <num> ", text)

def tokenize(text):
    """Tách từ (Tokenize)"""
    return text.split()

def count_words(text):
    """Đếm số từ trong văn bản"""
    return len(tokenize(text))


def add_word_ngrams(tokens, n_max=2):
    """Thêm word n-grams (ví dụ bigram) theo dạng token1_token2."""
    tokens = list(tokens)
    if n_max <= 1:
        return tokens

    enriched = list(tokens)
    for n in range(2, n_max + 1):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            enriched.append("_".join(tokens[i : i + n]))
    return enriched


def add_pmi_phrases(tokens, pmi_pairs: set[tuple[str, str]]):
    """Thêm phrase (bigram) theo danh sách cặp từ được chọn bằng PMI.

    Ví dụ: nếu ('học','phí') nằm trong pmi_pairs thì thêm token 'học_phí'.
    """
    tokens = list(tokens)
    if len(tokens) < 2:
        return tokens

    out = list(tokens)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in pmi_pairs:
            out.append(f"{pair[0]}_{pair[1]}")
    return out


def merge_pmi_phrases(tokens, pmi_pairs: set[tuple[str, str]]):
    """Ghép cụm từ theo PMI và loại bỏ từ đơn (strict segmentation).

    Quy tắc greedy trái->phải:
    - Nếu (w_i, w_{i+1}) nằm trong pmi_pairs: output w_i_w_{i+1} và bỏ qua i+1.
    - Ngược lại: output w_i.

    Ví dụ: ['lịch','học','môn','ai'] + {('lịch','học'),('môn','ai')} -> ['lịch_học','môn_ai']
    """
    tokens = list(tokens)
    out = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in pmi_pairs:
            out.append(f"{tokens[i]}_{tokens[i + 1]}")
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def char_ngrams(text, n_min=3, n_max=5):
    """Tạo char n-grams từ văn bản đã chuẩn hóa (không dùng thư viện NLP).

    Lưu ý: chỉ nên bật khi dữ liệu nhỏ để tránh nổ vocab.
    """
    text = to_lowercase(text)
    text = normalize_numbers(text)
    text = remove_punctuation_and_whitespace(text)
    if not text:
        return []

    # Giữ ranh giới từ bằng ký tự '_' thay vì bỏ hết khoảng trắng
    s = text.replace(" ", "_")
    grams = []
    for n in range(n_min, n_max + 1):
        if len(s) < n:
            continue
        grams.extend(s[i : i + n] for i in range(len(s) - n + 1))
    return grams

def clean_text(
    text,
    *,
    use_stopwords: bool = True,
    stopwords: set[str] | None = None,
    word_ngram_max: int = 2,
    use_pmi_phrases: bool = False,
    pmi_pairs: set[tuple[str, str]] | None = None,
    use_char_ngrams: bool = False,
    char_ngram_range: tuple[int, int] = (3, 5),
):
    """Hàm tổng hợp để xử lý toàn bộ các bước.

    Mặc định: lower + chuẩn hóa số + bỏ dấu câu/whitespace + tokenize + bỏ stopwords + thêm word bigram.
    """
    text = to_lowercase(text)
    text = normalize_numbers(text)
    text = remove_punctuation_and_whitespace(text)
    tokens = tokenize(text)

    if use_stopwords:
        sw = stopwords if stopwords is not None else load_stopwords()
        if sw:
            tokens = [t for t in tokens if t not in sw]

    # PMI phrases (bigram) được chọn bằng tương hỗ thông tin
    # Chế độ strict: ghép cụm từ và bỏ từ đơn
    if use_pmi_phrases and pmi_pairs:
        tokens = merge_pmi_phrases(tokens, pmi_pairs)

    # Word n-grams đại trà (unigram+bigram/...) - giữ lại để tương thích
    tokens = add_word_ngrams(tokens, n_max=word_ngram_max)

    if use_char_ngrams:
        n_min, n_max = char_ngram_range
        tokens.extend(char_ngrams(text, n_min=n_min, n_max=n_max))

    return tokens


def explain_clean_text(
    text,
    *,
    use_stopwords: bool = True,
    stopwords: set[str] | None = None,
    word_ngram_max: int = 2,
    use_pmi_phrases: bool = False,
    pmi_pairs: set[tuple[str, str]] | None = None,
):
    """Giải thích chi tiết từng bước tiền xử lý.

    Trả về dict chứa các trạng thái trung gian để UI có thể hiển thị "tính tay".
    Không bao gồm char n-grams (vì dài và khó đọc).
    """
    original = str(text)
    lower = to_lowercase(original)
    normalized_numbers = normalize_numbers(lower)
    removed_punct = remove_punctuation_and_whitespace(normalized_numbers)
    tokens_raw = tokenize(removed_punct)

    stopwords_used: set[str] = set()
    if use_stopwords:
        stopwords_used = stopwords if stopwords is not None else load_stopwords()

    tokens_no_stopwords = list(tokens_raw)
    removed_stopwords = []
    if use_stopwords and stopwords_used:
        tokens_no_stopwords = [t for t in tokens_raw if t not in stopwords_used]
        removed_stopwords = [t for t in tokens_raw if t in stopwords_used]

    tokens_after_pmi = list(tokens_no_stopwords)
    pmi_added = []
    pmi_matched_pairs = []
    pmi_merge_steps = []
    if use_pmi_phrases and pmi_pairs:
        # Greedy merge để đúng kết quả tokens_final
        i = 0
        while i < len(tokens_no_stopwords):
            if i < len(tokens_no_stopwords) - 1:
                pair = (tokens_no_stopwords[i], tokens_no_stopwords[i + 1])
                if pair in pmi_pairs:
                    pmi_matched_pairs.append(pair)
                    merged = f"{pair[0]}_{pair[1]}"
                    pmi_added.append(merged)
                    pmi_merge_steps.append({"i": i, "pair": pair, "merged": merged})
                    i += 2
                    continue
            i += 1
        tokens_after_pmi = merge_pmi_phrases(tokens_no_stopwords, pmi_pairs)

    tokens_after_ngrams = add_word_ngrams(tokens_after_pmi, n_max=word_ngram_max)

    return {
        "original": original,
        "lower": lower,
        "normalized_numbers": normalized_numbers,
        "removed_punctuation": removed_punct,
        "tokens_raw": tokens_raw,
        "use_stopwords": use_stopwords,
        "stopwords_count": len(stopwords_used),
        "removed_stopwords": removed_stopwords,
        "tokens_no_stopwords": tokens_no_stopwords,
        "use_pmi_phrases": use_pmi_phrases,
        "pmi_matched_pairs": pmi_matched_pairs,
        "pmi_added": pmi_added,
        "pmi_merge_steps": pmi_merge_steps,
        "tokens_after_pmi": tokens_after_pmi,
        "word_ngram_max": word_ngram_max,
        "tokens_final": tokens_after_ngrams,
    }
