from __future__ import annotations

import argparse
from pathlib import Path

import fitz


PAGES = [
    (
        "ACME CORPORATE TRAVEL POLICY",
        "FLIGHT AND GROUND TRANSPORT",
        "Employees must book economy-class flights at least 14 calendar days before departure whenever practical. "
        "The company reimburses one standard checked bag. Business-class travel requires written approval from a department director. "
        "Employees should use approved rail or taxi services for ground transport and retain every receipt. "
        "Personal vehicle mileage is reimbursed at INR 12 per kilometre.",
    ),
    (
        "ACME CORPORATE TRAVEL POLICY",
        "HOTEL AND MEAL ALLOWANCES",
        "Hotel accommodation is reimbursed up to INR 6,000 per night, excluding taxes. "
        "Any hotel booking above this limit requires written approval before the trip. "
        "The daily meal allowance is INR 1,200 per employee. Alcohol, minibar purchases, and personal entertainment are not reimbursable. "
        "Employees sharing a room must submit one itemised hotel invoice.",
    ),
    (
        "ACME CORPORATE TRAVEL POLICY",
        "EXPENSE CLAIM SUBMISSION",
        "Employees must submit travel expense claims within 10 calendar days after completing the trip. "
        "Each claim must include itemised receipts, the business purpose, travel dates, and the relevant project code. "
        "Managers should review claims within five working days. Finance may reject incomplete claims and request additional evidence. "
        "Claims submitted after 30 calendar days require approval from the finance controller.",
    ),
]


def create_corpus(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for title, heading, body in PAGES:
        page = document.new_page(width=595, height=842)
        page.insert_text((48, 54), title, fontsize=11, fontname="helv")
        page.draw_line((48, 68), (547, 68), color=(0.2, 0.3, 0.5), width=1)
        page.insert_text((48, 110), heading, fontsize=18, fontname="helv")
        page.insert_textbox(
            fitz.Rect(48, 145, 545, 500),
            body,
            fontsize=12,
            fontname="helv",
            lineheight=1.5,
        )
        page.insert_text((48, 790), "Synthetic benchmark document · Not a real company policy", fontsize=8)
    document.save(output_path)
    document.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic DocuMind retrieval benchmark PDF.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/sample_travel_policy.pdf"))
    args = parser.parse_args()
    output = create_corpus(args.output.resolve())
    print(f"Created synthetic benchmark PDF: {output}")


if __name__ == "__main__":
    main()
