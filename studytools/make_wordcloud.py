from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import Counter
from typing import Iterable

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


def _parse_exts(exts_csv: str) -> tuple[str, ...]:
    # "md,txt" -> (".md", ".txt")
    raw = [e.strip().lower() for e in exts_csv.split(",") if e.strip()]
    out = []
    for e in raw:
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return tuple(dict.fromkeys(out))  # de-dupe, preserve order


def _iter_files(base: Path, exts: tuple[str, ...], exclude_dirs: set[str]) -> Iterable[Path]:
    for p in base.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in exts:
            continue
        if any(part in exclude_dirs for part in p.parts):
            continue
        yield p


def read_target_text(root: Path, topic: str, exts: tuple[str, ...], exclude_dirs: set[str]) -> str:
    """
    Reads all matching files under root/topic (topic can include subfolders, e.g. 'R2D' or 'UCC FULL').
    """
    target_dir = (root / topic).resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Target folder not found: {target_dir}")

    paths = sorted(_iter_files(target_dir, exts=exts, exclude_dirs=exclude_dirs))
    if not paths:
        raise FileNotFoundError(f"No files with {exts} found under: {target_dir}")

    chunks: list[str] = []
    for p in paths:
        chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def tokenize(text: str) -> list[str]:
    # Strip markdown blocks/links/images/code
    text = MD_JUNK_RE.sub(" ", text)
    # Remove headings/bullets emphasis clutter
    text = text.replace("#", " ").replace("*", " ").replace("_", " ")
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)

    tokens: list[str] = []
    for w in text.split():
        if len(w) < 3:
            continue
        w = w.strip("'")
        if not w:
            continue
        if w in STOPWORDS:
            continue
        tokens.append(w)
    return tokens


def _slug(s: str) -> str:
    # nice filenames even if topic has spaces/slashes
    s = s.strip().replace("\\", "/")
    s = s.replace("/", "__")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s or "topic"


def main():
    ap = argparse.ArgumentParser(description="Generate a word cloud from a doctrine/topic folder.")
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
    ap.add_argument("--max-words", type=int, default=200, help="Max words in cloud")
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: studytools/out/<doctrine>__<topic>_wordcloud.png)",
    )
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(args.doctrine)

    exts = _parse_exts(args.exts)

    exclude_dirs = set()
    if not args.include_casebook:
        exclude_dirs.add("casebook_text")
    if args.exclude_dirs:
        exclude_dirs |= {d.strip() for d in args.exclude_dirs.split(",") if d.strip()}

    text = read_target_text(root=root, topic=args.topic, exts=exts, exclude_dirs=exclude_dirs)
    tokens = tokenize(text)

    freqs = Counter(tokens)
    if not freqs:
        raise RuntimeError("No tokens after filtering. Try widening --exts or reducing stopwords.")

    topic_slug = _slug(args.topic)
    doctrine_slug = _slug(str(root))
    out_path = Path(args.out) if args.out else Path("studytools/out") / f"{doctrine_slug}__{topic_slug}_wordcloud.png"
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
