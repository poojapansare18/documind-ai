from __future__ import annotations

import argparse
from pathlib import Path

import fitz


PAGES = [
    ("FLIGHT BOOKING", "Employees should reserve economy-class flights at least 14 calendar days before departure whenever practical. Business-class tickets require written approval from a department director. Changes to approved itineraries must be recorded in the travel portal."),
    ("GROUND TRANSPORT", "Employees should use approved rail operators or licensed taxis for business journeys. Personal vehicle use is reimbursed at INR 12 per kilometre when the route and business purpose are recorded. Parking fines and traffic penalties are not reimbursable."),
    ("HOTEL ACCOMMODATION", "Hotel accommodation is reimbursed up to INR 6,000 per night, excluding taxes. A booking above this nightly ceiling requires written approval before travel. Employees must submit an itemised hotel invoice showing the guest name and dates of stay."),
    ("MEAL ALLOWANCE", "The daily meal allowance is INR 1,200 per employee for an approved business trip. Alcohol, minibar purchases, and personal entertainment are excluded. Meals already provided by a conference or hotel must be deducted from the daily allowance."),
    ("EXPENSE SUBMISSION", "Employees must submit travel expense claims within 10 calendar days after completing the trip. Every claim must include itemised receipts, the business purpose, travel dates, and the relevant project code. Incomplete claims may be returned for correction."),
    ("LATE CLAIM APPROVAL", "Claims submitted more than 30 calendar days after the trip require written approval from the finance controller. A manager cannot waive this requirement. The employee must explain the delay and attach all original supporting receipts."),
    ("TRAVEL INSURANCE", "The company provides emergency medical and travel insurance for employees travelling on approved international assignments. Employees should carry the assistance telephone number and policy reference. Personal holiday extensions are not covered by the company policy."),
    ("ADVANCE PAYMENTS", "Employees may request a travel advance at least seven working days before departure. The request must state the destination, travel dates, estimated costs, and project code. Unused advance funds must be returned within five working days after the trip."),
    ("CANCELLATIONS", "If a business trip is cancelled, employees must notify the travel desk and their manager as soon as possible. Refundable tickets and hotel reservations should be cancelled promptly. Cancellation fees caused by personal changes are not reimbursable."),
    ("FOREIGN CURRENCY", "International expenses must be reported in the original transaction currency and converted to Indian rupees using the card statement exchange rate. If no card statement is available, Finance will apply the published exchange rate for the transaction date. Currency conversion fees must be shown separately."),
    ("CLIENT ENTERTAINMENT", "Client meals require advance written approval from the department head. The expense record must identify the client organisation, attendees, business purpose, and date. Gifts, personal hospitality, and entertainment for family members are not reimbursable."),
    ("DOCUMENT RETENTION", "Employees must retain original travel receipts and supporting invoices for at least 12 months after reimbursement. Finance may request additional evidence during an audit. Digital copies must be clear, complete, and readable.")
]


def create_corpus(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for heading, body in PAGES:
        page = document.new_page(width=595, height=842)
        page.insert_text((48, 54), "ACME CORPORATE TRAVEL POLICY", fontsize=11, fontname="helv")
        page.draw_line((48, 68), (547, 68), color=(0.2, 0.3, 0.5), width=1)
        page.insert_text((48, 110), heading, fontsize=18, fontname="helv")
        page.insert_textbox(fitz.Rect(48, 145, 545, 500), body, fontsize=12, fontname="helv", lineheight=1.5)
        page.insert_text((48, 790), "Synthetic benchmark document - Not a real company policy", fontsize=8)
    document.save(output_path)
    document.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic DocuMind retrieval benchmark PDF.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/sample_travel_policy_v2.pdf"))
    args = parser.parse_args()
    output = create_corpus(args.output.resolve())
    print(f"Created synthetic benchmark PDF: {output}")


if __name__ == "__main__":
    main()
