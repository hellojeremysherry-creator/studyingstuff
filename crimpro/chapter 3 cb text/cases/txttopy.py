from pathlib import Path
import re

CASE_DIR = Path(".")

def slugify_module_name(name: str) -> str:
    """
    Turn a filename stem into a valid Python module name:
    - replace non-alphanumerics with underscores
    - ensure it doesn't start with a digit
    - collapse multiple underscores
    """
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "case"
    if s[0].isdigit():
        s = f"case_{s}"
    return s

for txt_file in CASE_DIR.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")

    data = {
        "case_name": txt_file.stem,
        "full_text": text,
        "paragraphs": [p.strip() for p in text.split("\n\n") if p.strip()],
    }

    module_name = slugify_module_name(txt_file.stem)
    py_file = CASE_DIR / f"{module_name}.py"

    # If you don't want overwrites, uncomment:
    # if py_file.exists():
    #     print(f"Skipping (exists): {py_file.name}")
    #     continue

    py_contents = (
        "# Auto-generated from a .txt case file\n"
        "# Do not edit manually unless you know what you're doing.\n\n"
        "CASE = "
        + repr(data)
        + "\n"
    )

    py_file.write_text(py_contents, encoding="utf-8")
    print(f"Converted: {txt_file.name} -> {py_file.name}")

print("Done.")
