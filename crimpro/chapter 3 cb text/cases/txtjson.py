import json
from pathlib import Path

# Load text
text = Path("Mapp v. Ohio_367 U.S. 643.txt").read_text(encoding="utf-8")

data = {
    "case_name": "Mapp v. Ohio",
    "citation": "367 U.S. 643 (1961)",
    "court": "Supreme Court of the United States",
    "year": 1961,
    "full_text": text,
    "paragraphs": [p.strip() for p in text.split("\n\n") if p.strip()]
}

# Write JSON
with open("Mapp_v_Ohio_367_US_643.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
