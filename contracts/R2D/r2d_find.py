#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

R2D_DIR = Path(__file__).resolve().parent

# Light synonym expansion so "define an offer" still hits §24-type language.
SYNONYMS = {
    "offer": ["offer", "offeree", "offeror", "bargain", "proposal", "manifestation of willingness"],
    "objective": ["objective", "reasonable", "external", "manifestation", "undisclosed intention"],
    "acceptance": ["acceptance", "accept", "assent", "power of acceptance"],
}

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","that","this","is","are","be","must","it","as"
}

def tokenize(q: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_']+", q.lower())
    return [w for w in words if w not in STOPWORDS]

def expand_terms(tokens: list[str]) -> list[str]:
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in SYNONYMS:
            expanded.extend(SYNONYMS[t])
    # de-dup preserving order
    seen = set()
    out = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def score_text(text: str, terms: list[str]) -> int:
    # crude but effective: count term occurrences (case-insensitive)
    lower = text.lower()
    score = 0
    for term in terms:
        score += lower.count(term.lower())
    return score

def best_snippets(text: str, terms: list[str], max_snips: int = 3, context: int = 90):
    lower = text.lower()
    hits = []
    for term in terms:
        t = term.lower()
        start = 0
        while True:
            idx = lower.find(t, start)
            if idx == -1:
                break
            hits.append(idx)
            start = idx + max(1, len(t))
    hits = sorted(set(hits))[: max_snips]
    snippets = []
    for idx in hits:
        a = max(0, idx - context)
        b = min(len(text), idx + context)
        snip = text[a:b].replace("\n", " ")
        snippets.append("…" + snip.strip() + "…")
    return snippets

def main():
    ap = argparse.ArgumentParser(description="Search contracts/R2D .txt files by natural query.")
    ap.add_argument("query", nargs="+", help='e.g. "define an offer"')
    ap.add_argument("--top", type=int, default=10, help="how many files to show")
    ap.add_argument("--minscore", type=int, default=1, help="minimum score to include")
    args = ap.parse_args()

    query = " ".join(args.query)
    tokens = tokenize(query)
    terms = expand_terms(tokens)

    results = []
    for p in sorted(R2D_DIR.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        s = score_text(text, terms)
        if s >= args.minscore:
            results.append((s, p, text))

    results.sort(key=lambda x: x[0], reverse=True)

    if not results:
        print("No matches. Try different wording or lower --minscore.")
        return

    for rank, (s, p, text) in enumerate(results[: args.top], start=1):
        print(f"\n#{rank}  score={s}  file={p.name}")
        for snip in best_snippets(text, terms):
            print("   " + snip)

if __name__ == "__main__":
    main()
