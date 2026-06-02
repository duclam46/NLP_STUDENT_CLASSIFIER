import pandas as pd
import os
from pathlib import Path

DATA_FILE = "data/training_data.csv"
STOPWORDS_FILE = "data/stopwords.csv"

def load_data():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=["text", "label"])
        df.to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE)

def save_data(df):
    df.to_csv(DATA_FILE, index=False)


def load_stopwords_df() -> pd.DataFrame:
    """Load stopwords from CSV."""
    path = Path(STOPWORDS_FILE)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["word"]).to_csv(path, index=False, encoding="utf-8")

    df = pd.read_csv(path, encoding="utf-8")
    if "word" not in df.columns:
        df = pd.DataFrame({"word": []})
    df["word"] = df["word"].fillna("").astype(str)
    return df


def save_stopwords_df(df: pd.DataFrame):
    path = Path(STOPWORDS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "word" not in out.columns:
        out = pd.DataFrame({"word": []})
    out["word"] = out["word"].fillna("").astype(str)
    out.to_csv(path, index=False, encoding="utf-8")
