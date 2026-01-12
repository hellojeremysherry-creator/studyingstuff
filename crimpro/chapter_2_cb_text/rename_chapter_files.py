#!/usr/bin/env python3
import os
import re
import sys
import argparse
import subprocess
import unicodedata
from pathlib import Path

def slugify(s: str) -> str:
    # Normalize unicode and strip odd characters
    s = unicodedata.normalize("NFKD", s)

    # Remove quotes (your sample has trailing quotes)
    s = s.replace('"', '').replace("’", "'").replace("‘", "'")

    # Convert separators/punctuation to spaces
    s = re.sub(r"[(){}\[\]]", " ", s)
    s = re.sub(r"[–—-]", " ", s)          # hyphen-like chars
    s = re.sub(r"[.,:;!?]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Lower + snake_case
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)     # keep alnum + spaces only
    s = re.sub(r"\s+", "_", s).strip("_")
    return s

def make_new_name(old_name: str) -> str:
    """
    Tries to parse names like:
      Chapter 2 - SEARCH AND SEIZURE (B. WHAT IS A SEARCH).json
    Produces:
      chapter_02_search_and_seizure_b_what_is_a_search.json
    """
    # Strip any accidental trailing quotes
    old_name = old_name.strip().strip('"')

    p = Path(old_name)
    stem = p.stem
    ext = p.suffix.lower()

    # Extract chapter number if present
    # Matches "Chapter 2" or "CHAPTER 12"
    chap_match = re.search(r"\bchapter\s+(\d+)\b", stem, flags=re.IGNORECASE)
    chap_num = chap_match.group(1) if chap_match else None

    # Extract section letter if present inside parentheses like "(B. ...)" or "(B ...)"
    sec_match = re.search(r"\(([A-H])\s*[\.\)]", stem, flags=re.IGNORECASE)
    sec_letter = sec_match.group(1).upper() if sec_match else None

    # Build a cleaned base name (remove the word "Chapter X" but keep the rest)
    base = stem
    base = re.sub(r"\bchapter\s+\d+\b", "", base, flags=re.IGNORECASE).strip()
    base = base.strip("- ").strip()

    # If there's a "(B. ...)" keep its content but remove parentheses
    base = base.replace("(", " ").replace(")", " ")

    base_slug = slugify(base)

    parts = []
    if chap_num is not None:
        parts.append(f"chapter_{int(chap_num):02d}")
    parts.append(base_slug)

    # Sometimes base_slug already includes the letter; but if not, prefix it.
    if sec_letter and not re.search(rf"(^|_)({sec_letter.lower()})(_|$)", base_slug):
        parts.append(sec_letter.lower())

    new_stem = "_".join([p for p in parts if p])
    new_stem = re.sub(r"_+", "_", new_stem).strip("_")

    return f"{new_stem}{ext}"

def is_git_repo(path: Path) -> bool:
    try:
        subprocess.run(["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def git_mv(src: Path, dst: Path):
    subprocess.run(["git", "mv", str(src), str(dst)], check=True)

def main():
    ap = argparse.ArgumentParser(description="Batch rename chapter files to machine-friendly names.")
    ap.add_argument("directory", nargs="?", default=".", help="Directory containing files")
    ap.add_argument("--dry-run", action="store_true", help="Print changes without renaming")
    ap.add_argument("--use-os-rename", action="store_true",
                    help="Force os.rename even if this is a git repo (default uses git mv)")
    args = ap.parse_args()

    root = Path(args.directory).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    use_git = (not args.use_os_rename) and is_git_repo(root)

    # Only rename .txt/.json (adjust if you want)
    files = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".json"}])

    mapping = []
    for f in files:
        new_name = make_new_name(f.name)
        if new_name != f.name:
            mapping.append((f, root / new_name))

    if not mapping:
        print("No files matched / nothing to rename.")
        return

    # Detect collisions
    targets = [dst.name for _, dst in mapping]
    if len(set(targets)) != len(targets):
        print("ERROR: Renaming would cause filename collisions. Fix rules or rename manually.", file=sys.stderr)
        # Show duplicates
        seen = {}
        for src, dst in mapping:
            seen.setdefault(dst.name, []).append(src.name)
        for name, srcs in seen.items():
            if len(srcs) > 1:
                print(f"  Collision -> {name}: {srcs}", file=sys.stderr)
        sys.exit(2)

    for src, dst in mapping:
        print(f"{src.name}  ->  {dst.name}")

    if args.dry_run:
        print("\n(dry-run: no changes made)")
        return

    for src, dst in mapping:
        if use_git:
            git_mv(src, dst)
        else:
            os.rename(src, dst)

    print(f"\nDone. Renamed {len(mapping)} files using {'git mv' if use_git else 'os.rename'}.")

if __name__ == "__main__":
    main()
