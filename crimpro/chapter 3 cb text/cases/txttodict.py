from pathlib import Path

CASE_DIR = Path(".")

cases = []   # list of dictionaries

for txt_file in CASE_DIR.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")

    case = {
        "case_name": txt_file.stem,
        "full_text": text,
        "paragraphs": [p.strip() for p in text.split("\n\n") if p.strip()],
    }

    cases.append(case)

print(f"Loaded {len(cases)} cases")

# Safe forever:
print(cases[0].get("case_name"))
