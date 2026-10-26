import sys
from pathlib import Path

from docling.document_converter import DocumentConverter

input_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "./cvs")
output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "./mds")
output_dir.mkdir(parents=True, exist_ok=True)

pdfs = sorted(input_dir.glob("*.pdf"))
if not pdfs:
    sys.exit(f"No PDF files found in {input_dir}")

converter = DocumentConverter()

for pdf in pdfs:
    result = converter.convert(pdf)
    doc = result.document

    out = output_dir / f"{pdf.stem}.md"
    out.write_text(doc.export_to_markdown(), encoding="utf-8")
    print(f"{pdf.name} -> {out}")
