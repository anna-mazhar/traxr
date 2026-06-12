"""Generate the tiny real artifacts in tests/fixtures/ (build-plan line 410).

Run from the repo root with the dev venv:

    python scripts/generate_test_fixtures.py

The outputs are committed; this script exists for provenance/regeneration.
"""

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

CSV_CONTENT = """\
name,date,amount,quantity,price
Widget A,2023-01-15,1200,5,19.99
Widget B,2023-02-20,3400,12,7.50
Gadget C,2023-03-05,560,3,129.00
Gadget D,2023-04-11,7800,40,2.25
Doohickey E,2023-05-30,910,7,45.10
"""

TXT_CONTENT = """\
Quarterly Operations Report

Revenue grew to 45,000 dollars in Q3, up 12% from the prior quarter. \
The team shipped 1,250 units before 10/15/2023. Alice Johnson led the rollout.

Costs were held at 32,500 dollars despite supplier churn. \
The new vendor signed on 09/01/2023. Bob Smith negotiated the contract.

Headcount reached 48 by the end of the quarter. \
Attrition stayed below 5% for the third consecutive period.

Inventory turnover improved to 8.4 cycles. Warehouse capacity sits at 78% \
utilization, with 2,300 pallets on hand as of 11/01/2023.

The outlook for Q4 projects 52,000 dollars in revenue. Carol Davis presented \
the forecast to the board on 11/20/2023.
"""

MD_CONTENT = """\
# Quarterly Operations Report

Revenue grew to 45,000 dollars in Q3, up 12% from the prior quarter. \
The team shipped 1,250 units before 10/15/2023. Alice Johnson led the rollout.

## Costs

Costs were held at 32,500 dollars despite supplier churn. \
The new vendor signed on 09/01/2023. Bob Smith negotiated the contract.

## People

Headcount reached 48 by the end of the quarter. \
Attrition stayed below 5% for the third consecutive period.

## Outlook

The outlook for Q4 projects 52,000 dollars in revenue. Carol Davis presented \
the forecast to the board on 11/20/2023.
"""

PDF_PAGES = [
    [
        "PAGEONE Quarterly Operations Report for fiscal year 2023.",
        "Revenue grew to 45,000 dollars in Q3, up 12% from Q2. "
        "The team shipped 1,250 units before 10/15/2023. Alice Johnson led the rollout.",
        "Costs were held at 32,500 dollars despite supplier churn since 09/01/2023.",
    ],
    [
        "PAGETWO People and staffing summary follows below.",
        "Headcount reached 48 by quarter end. Attrition stayed below 5 percent. "
        "Bob Smith negotiated the renewal contract on 08/12/2023.",
        "Training budget consumed was 9,800 dollars across 36 sessions.",
    ],
    [
        "PAGETHREE Inventory and logistics overview for the period.",
        "Inventory turnover improved to 8.4 cycles with 2,300 pallets on hand. "
        "Warehouse utilization sits at 78 percent as of 11/01/2023.",
        "Freight costs averaged 415 dollars per shipment over 612 shipments.",
    ],
    [
        "PAGEFOUR Outlook and board summary conclude this report.",
        "Q4 projects 52,000 dollars in revenue. Carol Davis presented the "
        "forecast to the board on 11/20/2023.",
        "The board approved a 6,500 dollar contingency reserve for Q4.",
    ],
]


def make_xlsx(path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    rows = [line.split(",") for line in CSV_CONTENT.strip().splitlines()]
    for row in rows:
        typed_row = []
        for value in row:
            try:
                typed_row.append(float(value) if "." in value else int(value))
            except ValueError:
                typed_row.append(value)
        ws.append(typed_row)
    wb.save(path)
    wb.close()


def make_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    for blocks in PDF_PAGES:
        page = doc.new_page()  # A4 portrait
        y = 72
        for text in blocks:
            rect = fitz.Rect(72, y, 523, y + 120)
            page.insert_textbox(rect, text, fontname="helv", fontsize=11)
            y += 160  # separated rects -> separate text blocks
    doc.save(path)
    doc.close()


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    # Valid artifacts
    (FIXTURES / "sample.csv").write_text(CSV_CONTENT, encoding="utf-8")
    (FIXTURES / "sample.txt").write_text(TXT_CONTENT, encoding="utf-8")
    (FIXTURES / "sample.md").write_text(MD_CONTENT, encoding="utf-8")
    make_xlsx(FIXTURES / "sample.xlsx")
    make_pdf(FIXTURES / "sample.pdf")

    # Negative variants
    (FIXTURES / "empty.csv").write_bytes(b"")
    (FIXTURES / "empty.txt").write_bytes(b"")
    # Right magic, unparseable body -> InvalidArtifactError
    (FIXTURES / "corrupt.pdf").write_bytes(b"%PDF-1.7\n" + b"\xde\xad\xbe\xef" * 64)
    xlsx_bytes = (FIXTURES / "sample.xlsx").read_bytes()
    (FIXTURES / "truncated.xlsx").write_bytes(xlsx_bytes[:64])
    pdf_bytes = (FIXTURES / "sample.pdf").read_bytes()
    (FIXTURES / "truncated.pdf").write_bytes(pdf_bytes[: len(pdf_bytes) // 10])
    # Wrong extension: a *different* format's magic -> ModalityMismatchError
    (FIXTURES / "wrong_extension.csv").write_bytes(pdf_bytes)
    (FIXTURES / "wrong_extension.pdf").write_bytes(xlsx_bytes)
    # Unsupported modalities (extension check fires before content is read)
    (FIXTURES / "sample.docx").write_bytes(b"PK\x03\x04 not a real docx")
    (FIXTURES / "sample.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 16  # PNG signature + stub body
    )

    print(f"Fixtures written to {FIXTURES}")


if __name__ == "__main__":
    main()
