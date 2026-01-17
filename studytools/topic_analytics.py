from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import Counter
from typing import Iterable

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


def _parse_exts(exts_csv: str) -> tuple[str, ...]:
    raw = [e.strip().lower() for e in exts_csv.split(",") if e.strip()]
    out = []
    for e in raw:
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return tuple(dict.fromkeys(out))


def _iter_files(base: Path, exts: tuple[str, ...], exclude_dirs: set[str]) -> Iterable[Path]:
    for p in base.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in exts:
            continue
        if any(part in exclude_dirs for part in p.parts):
            continue
        yield p


def collect_documents(root: Path, topic: str, exts: tuple[str, ...], exclude_dirs: set[str]) -> list[str]:
    target = (root / topic).resolve()

    paths: list[Path] = []

    if target.is_file():
        if target.suffix.lower() not in exts:
            raise ValueError(f"File extension not in {exts}: {target}")
        paths = [target]

    elif target.is_dir():
        for p in target.rglob("*"):
            if p.is_dir():
                continue
            if p.suffix.lower() not in exts:
                continue
            if any(part in exclude_dirs for part in p.parts):
                continue
            paths.append(p)

    else:
        raise FileNotFoundError(f"Target not found: {target}")

    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No matching files found for: {target}")

    docs: list[str] = []
    for p in paths:
        docs.append(p.read_text(encoding="utf-8", errors="ignore"))
    return docs



def tokenize(text: str) -> list[str]:
    text = MD_JUNK_RE.sub(" ", text)
    text = text.replace("#", " ").replace("*", " ").replace("_", " ")
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)

    toks: list[str] = []
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


def _slug(s: str) -> str:
    s = s.strip().replace("\\", "/")
    s = s.replace("/", "__")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s or "topic"


def main():
    ap = argparse.ArgumentParser(description="Analytics for a doctrine/topic folder.")
    ap.add_argument("--doctrine", default="crimlaw", help="Top-level folder name (default: crimlaw)")
    ap.add_argument("--root", default=None, help="Override root path (if set, ignores --doctrine)")
    ap.add_argument("--topic", required=True, help="Folder path under root (e.g., homicide OR R2D OR 'UCC FULL')")

    ap.add_argument("--exts", default="md,txt", help="Comma-separated extensions to include (default: md,txt)")
    ap.add_argument(
        "--include-casebook",
        action="store_true",
        help="Do not exclude 'casebook_text' directory (kept for backward compatibility)",
    )
    ap.add_argument("--exclude-dirs", default=None, help="Comma-separated dir names to exclude (optional)")
    ap.add_argument("--topn", type=int, default=25)
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(args.doctrine)
    exts = _parse_exts(args.exts)

    exclude_dirs = set()
    if not args.include_casebook:
        exclude_dirs.add("casebook_text")
    if args.exclude_dirs:
        exclude_dirs |= {d.strip() for d in args.exclude_dirs.split(",") if d.strip()}

    docs = collect_documents(root=root, topic=args.topic, exts=exts, exclude_dirs=exclude_dirs)
    text = "\n".join(docs)  # for token counts/bigrams
    tokens = tokenize(text)

    out_dir = Path("studytools/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    topic_slug = _slug(args.topic)
    doctrine_slug = _slug(str(root))
    prefix = f"{doctrine_slug}__{topic_slug}"

    # Top terms
    term_counts = Counter(tokens).most_common(args.topn)
    bar_plot(term_counts, f"Top terms — {args.topic}", out_dir / f"{prefix}_top_terms.png")

    # Top bigrams
    bi = [" ".join(b) for b in bigrams(tokens)]
    bigram_counts = Counter(bi).most_common(args.topn)
    bar_plot(bigram_counts, f"Top bigrams — {args.topic}", out_dir / f"{prefix}_top_bigrams.png")

    # TF-IDF keywords across documents (folder = many docs, file = 1 doc)
    min_df = 1 if len(docs) == 1 else 2

    vec = TfidfVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=min_df,
    )

    X = vec.fit_transform(docs)
    scores = X.mean(axis=0).A1  # average tf-idf across docs
    feats = vec.get_feature_names_out()

    top_idx = scores.argsort()[-args.topn:][::-1]
    tfidf_top = [(feats[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    df = pd.DataFrame(tfidf_top, columns=["term", "tfidf"])
    df_path = out_dir / f"{prefix}_tfidf.csv"
    df.to_csv(df_path, index=False)
    print(f"Saved: {df_path}")

    print("\nTop TF-IDF terms:")
    for t, s in tfidf_top[:15]:
        print(f"{t:>30}  {s:.4f}")


if __name__ == "__main__":
    main()
