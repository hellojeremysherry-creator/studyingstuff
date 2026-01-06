from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    # fallback if nltk stopwords not available
    STOPWORDS = set()

# Add "legal-ish" and outline-ish stopwords so the cloud is more meaningful
LEGAL_STOP = {
    "court", "case", "held", "holding", "rule", "rules", "element", "elements",
    "thus", "therefore", "because", "example", "examples",
    "note", "notes", "outline", "see", "cf", "id", "supra",
    "ca", "california", "mpc", "common", "law",
    "section", "sections", "p", "pp",
}
STOPWORDS |= LEGAL_STOP

MD_JUNK_RE = re.compile(r"(```.*?```|`[^`]+`|\!\[[^\]]*\]\([^)]+\)|\[[^\]]+\]\([^)]+\))", re.S)
NON_WORD_RE = re.compile(r"[^a-zA-Z']+")

def read_topic_text(crimlaw_root: Path, topic: str, include_casebook: bool) -> str:
    topic_dir = crimlaw_root / topic
    if not topic_dir.exists():
        raise FileNotFoundError(f"Topic folder not found: {topic_dir}")

    paths = list(topic_dir.rglob("*.md"))
    if include_casebook:
        paths += list(topic_dir.rglob("*.txt"))

    # If topic dir has no md/txt, still allow user to point directly to folder names
    if not paths:
        raise FileNotFoundError(f"No .md (or .txt) files found under: {topic_dir}")

    chunks = []
    for p in sorted(paths):
        # Skip huge casebook dumps unless explicitly included
        if (not include_casebook) and ("casebook_text" in p.parts):
            continue
        chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)

def tokenize(text: str) -> list[str]:
    # Strip markdown blocks/links/images/code
    text = MD_JUNK_RE.sub(" ", text)
    # Remove headings/bullets emphasis clutter
    text = text.replace("#", " ").replace("*", " ").replace("_", " ")
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)

    tokens = []
    for w in text.split():
        if len(w) < 3:
            continue
        if w in STOPWORDS:
            continue
        # collapse possessives and simple apostrophes
        w = w.strip("'")
        if not w or w in STOPWORDS:
            continue
        tokens.append(w)
    return tokens

def main():
    ap = argparse.ArgumentParser(description="Generate a word cloud from a crimlaw topic folder.")
    ap.add_argument("--crimlaw-root", default="crimlaw", help="Path to crimlaw directory (default: crimlaw)")
    ap.add_argument("--topic", required=True, help="Topic folder name under crimlaw (e.g., homicide, attempt)")
    ap.add_argument("--include-casebook", action="store_true", help="Include .txt/casebook_text if present")
    ap.add_argument("--max-words", type=int, default=200, help="Max words in cloud")
    ap.add_argument("--out", default=None, help="Output PNG path (default: studytools/out/<topic>_wordcloud.png)")
    args = ap.parse_args()

    crimlaw_root = Path(args.crimlaw_root)
    text = read_topic_text(crimlaw_root, args.topic, args.include_casebook)
    tokens = tokenize(text)

    freqs = Counter(tokens)
    if not freqs:
        raise RuntimeError("No tokens after filtering. Try --include-casebook or reduce stopwords.")

    out_path = Path(args.out) if args.out else Path("studytools/out") / f"{args.topic}_wordcloud.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wc = WordCloud(width=1600, height=900, background_color="white", max_words=args.max_words)
    wc.generate_from_frequencies(freqs)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")
    print("Top 20 terms:")
    for w, c in freqs.most_common(20):
        print(f"{w:>18}  {c}")

if __name__ == "__main__":
    main()
