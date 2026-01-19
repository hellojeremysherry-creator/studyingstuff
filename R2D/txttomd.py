from pathlib import Path
import re

CASE_DIR = Path(".")

def txt_to_md(text: str) -> str:
    lines = text.splitlines()
    out = []

    for line in lines:
        stripped = line.strip()

        # Preserve blank lines
        if not stripped:
            out.append("")
            continue

        # ALL CAPS → H2
        if stripped.isupper() and len(stripped) > 3:
            out.append(f"## {stripped.title()}")
            continue

        # Line ending with colon → H3
        if stripped.endswith(":") and len(stripped.split()) <= 12:
            out.append(f"### {stripped[:-1]}")
            continue

        # Bullet normalization
        if re.match(r"^[*-]\s+", stripped):
            out.append(f"- {stripped[2:].strip()}")
            continue

        out.append(line)

    return "\n".join(out).strip() + "\n"

def main():
    txt_files = sorted(CASE_DIR.glob("*.txt"))
    if not txt_files:
        print("No TXT files found.")
        return

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8")
            md_text = txt_to_md(text)

            md_file = txt_file.with_suffix(".md")
            md_file.write_text(md_text, encoding="utf-8")

            print(f"Converted: {txt_file.name} -> {md_file.name}")
        except Exception as e:
            print(f"FAILED: {txt_file.name} ({e})")

    print("Done.")

if __name__ == "__main__":
    main()
