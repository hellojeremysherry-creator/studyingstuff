from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

LEGAL_STOP = {
    "court", "case", "held", "holding", "rule", "rules", "element", "elements",
    "thus", "therefore", "because", "example", "examples",
    "note", "notes", "outline", "see", "cf", "id", "supra",
    "ca", "california", "mpc", "common", "law",
}
STOPWORDS |= LEGAL_STOP

MD_JUNK_RE = re.compile(r"(```.*?```|`[^`]+`|\!\[[^\]]*\]\([^)]+\)|\[[^\]]+\]\([^)]+\))", re.S)
NON_WORD_RE = re.compile(r"[^a-zA-Z']+")

def load_text(topic_dir: Path, include_casebook: bool) -> str:
    paths = list(topic_dir.rglob("*.md"))
    if include_casebook:
        paths += list(topic_dir.rglob("*.txt"))
    if not include_casebook:
        paths = [p for p in paths if "casebook_text" not in p.parts]
    chunks = [p.read_text(encoding="utf-8", errors="ignore") for p in sorted(paths)]
    return "\n".join(chunks)

def tokenize(text: str) -> list[str]:
    text = MD_JUNK_RE.sub(" ", text)
    text = text.replace("#", " ").replace("*", " ").replace("_", " ")
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    toks = []
    for w in text.split():
        w = w.strip("'")
        if len(w) < 3:
            continue
        if w in STOPWORDS:
            continue
        toks.append(w)
    return toks

def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))

def bar_plot(items: list[tuple[str, float]], title: str, out_path: Path):
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(12, 7))
    plt.barh(list(reversed(labels)), list(reversed(values)))
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Analytics for a crimlaw topic folder.")
    ap.add_argument("--crimlaw-root", default="crimlaw")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--include-casebook", action="store_true")
    ap.add_argument("--topn", type=int, default=25)
    args = ap.parse_args()

    crimlaw_root = Path(args.crimlaw_root)
    topic_dir = crimlaw_root / args.topic
    if not topic_dir.exists():
        raise FileNotFoundError(topic_dir)

    text = load_text(topic_dir, args.include_casebook)
    tokens = tokenize(text)

    out_dir = Path("studytools/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top terms
    term_counts = Counter(tokens).most_common(args.topn)
    bar_plot(term_counts, f"Top terms — {args.topic}", out_dir / f"{args.topic}_top_terms.png")

    # Top bigrams
    bi = [" ".join(b) for b in bigrams(tokens)]
    bigram_counts = Counter(bi).most_common(args.topn)
    bar_plot(bigram_counts, f"Top bigrams — {args.topic}", out_dir / f"{args.topic}_top_bigrams.png")

    # TF-IDF keywords (single-doc TF-IDF still helpful for “distinctive-ish” terms)
    # Better version later: compare across multiple topic folders.
    vec = TfidfVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform([text])
    scores = X.toarray()[0]
    feats = vec.get_feature_names_out()
    top_idx = scores.argsort()[-args.topn:][::-1]
    tfidf_top = [(feats[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    df = pd.DataFrame(tfidf_top, columns=["term", "tfidf"])
    df_path = out_dir / f"{args.topic}_tfidf.csv"
    df.to_csv(df_path, index=False)
    print(f"Saved: {df_path}")

    print("\nTop TF-IDF terms:")
    for t, s in tfidf_top[:15]:
        print(f"{t:>30}  {s:.4f}")

if __name__ == "__main__":
    main()
