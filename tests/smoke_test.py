from __future__ import annotations

import sys
from dataclasses import dataclass

import pandas as pd

from src.preprocessing import clean_text, load_stopwords, reset_stopwords_cache
from src.vectorizer import build_vocab
from src.classifier import predict


@dataclass(frozen=True)
class Case:
    text: str
    expected_label: str
    min_score: float = 0.05


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def test_stopwords_loaded() -> None:
    reset_stopwords_cache()
    sw = load_stopwords()
    assert isinstance(sw, set)
    assert len(sw) > 0, "Stopwords set is empty. Check data/stopwords.csv"


def test_unigram_bigram_behavior() -> None:
    # Ensure bigram generation happens (default word_ngram_max=2)
    tokens = clean_text("học phí")
    assert "học" in tokens
    assert "phí" in tokens
    assert "học_phí" in tokens, "Expected bigram 'học_phí' not found in tokens"


def test_end_to_end_prediction() -> None:
    df = pd.read_csv("data/training_data.csv")
    assert not df.empty, "training_data.csv is empty"
    assert {"text", "label"}.issubset(df.columns)

    vocab = build_vocab(df)
    assert len(vocab) > 0

    cases = [
        Case("Hạn nộp học phí là khi nào?", "Học phí", 0.05),
        Case("Bao giờ có điểm thi kết thúc học phần?", "Điểm số", 0.05),
        Case("Khi nào có lịch học môn AI?", "Lịch học", 0.05),
        Case("Quên mật khẩu đăng nhập portal", "Tài khoản", 0.05),
    ]

    for c in cases:
        label, score, df_res = predict(c.text, df, vocab)
        assert not df_res.empty
        assert isinstance(label, str)
        assert score >= 0.0
        # With small datasets, scores fluctuate; enforce weak constraints.
        assert score >= c.min_score, f"Score too low for '{c.text}': {score}"
        assert (
            label == c.expected_label
        ), f"Expected '{c.expected_label}' for '{c.text}', got '{label}' (score={score})"


def main() -> int:
    _ensure_utf8_stdout()

    tests = [
        ("stopwords_loaded", test_stopwords_loaded),
        ("unigram_bigram", test_unigram_bigram_behavior),
        ("end_to_end", test_end_to_end_prediction),
    ]

    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")

    if failed:
        print(f"\n{failed} test(s) failed")
        return 1

    print("\nAll smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
