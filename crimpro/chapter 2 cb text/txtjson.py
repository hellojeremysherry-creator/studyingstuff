import json
from pathlib import Path

# Current folder (where you run the script)
CASE_DIR = Path(".")

for txt_file in CASE_DIR.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")

    data = {
        # Use the filename (without .txt) as case_name by default
        "case_name": txt_file.stem,
        "full_text": text,
        "paragraphs": [p.strip() for p in text.split("\n\n") if p.strip()],
    }

    json_file = txt_file.with_suffix(".json")

    # Write JSON (overwrite if it already exists)
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Converted: {txt_file.name} -> {json_file.name}")

print("Done.")
