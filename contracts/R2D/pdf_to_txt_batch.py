from pathlib import Path
import fitz  # PyMuPDF

CASE_DIR = Path(".")  # folder where script is run (or where it lives if you prefer)

def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # "text" gives a reasonable reading order for most PDFs
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()

def main():
    pdf_files = sorted(CASE_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in this folder.")
        return

    for pdf_file in pdf_files:
        try:
            text = pdf_to_text(pdf_file)

            txt_file = pdf_file.with_suffix(".txt")
            txt_file.write_text(text, encoding="utf-8")

            print(f"Converted: {pdf_file.name} -> {txt_file.name} ({len(text)} chars)")
        except Exception as e:
            print(f"FAILED: {pdf_file.name} ({e})")

    print("Done.")

if __name__ == "__main__":
    main()
