from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf as fitz


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


# Version 3 deliberately spans three policy domains and several page layouts.
# All organisations and policy text below are fictional benchmark material.
DIVERSE_PAGES = [
    ("NORTHSTAR TRAVEL POLICY", "BOOKING AND APPROVAL", "Standard economy travel should be booked at least 14 calendar days before departure. Any business-class itinerary requires written approval from the employee's executive vice president before ticketing.", "paragraph"),
    ("NORTHSTAR TRAVEL POLICY", "RAIL AND LOCAL TRANSPORT", "For journeys under 400 kilometres, employees should consider rail before air travel. Local journeys must use licensed taxis or the approved mobility provider. Unapproved ride-share services are not reimbursable.", "paragraph"),
    ("NORTHSTAR TRAVEL POLICY", "LODGING LIMITS", "Accommodation ceilings are stated per night, before applicable taxes. Tier 1 cities: INR 7,500. Tier 2 cities: INR 5,500. Other locations: INR 4,000. A higher rate requires written regional-director approval before booking.", "table"),
    ("NORTHSTAR TRAVEL POLICY", "TRAVEL ADVANCES", "An employee may request an advance no earlier than 15 working days and no later than 5 working days before departure. The request must include a cost estimate and project code. Unused funds must be returned within 7 working days after the trip.", "paragraph"),
    ("NORTHSTAR TRAVEL POLICY", "EXPENSE CLAIMS", "Expense claims must be filed within 12 calendar days after the trip ends. The submission must include itemised receipts, dates, business purpose, cost centre, and manager approval. Claims missing mandatory evidence are returned to the employee.", "paragraph"),
    ("NORTHSTAR TRAVEL POLICY", "LATE SUBMISSIONS", "A claim filed more than 30 calendar days after the trip requires an exception signed by the Finance Director. The line manager cannot approve this exception. The employee must attach a written explanation and original receipts.", "paragraph"),
    ("NORTHSTAR TRAVEL POLICY", "CANCELLATIONS", "When a trip is cancelled, the traveller must notify the travel desk and line manager within one business day. Refundable reservations must be cancelled immediately. A personal-change fee is not eligible for reimbursement.", "paragraph"),
    ("NORTHSTAR TRAVEL POLICY", "FOREIGN CURRENCY", "Report overseas expenses in the transaction currency. Convert to INR using the card issuer's posted settlement rate. If the card statement is unavailable, Finance uses the Reserve Bank reference rate for the transaction date. Show conversion fees as a separate line item.", "paragraph"),
    ("MERIDIAN PROCUREMENT STANDARD", "PURCHASE REQUESTS", "A purchase request must identify the business owner, cost centre, item specification, quantity, delivery location, and required-by date. A request is not a purchase order and does not authorise a supplier to begin work.", "paragraph"),
    ("MERIDIAN PROCUREMENT STANDARD", "APPROVAL THRESHOLDS", "Approval matrix (before tax): Up to INR 25,000 — department manager; INR 25,001 to INR 2,00,000 — department head; above INR 2,00,000 — finance controller and procurement director jointly.", "table"),
    ("MERIDIAN PROCUREMENT STANDARD", "COMPETITIVE QUOTES", "Purchases above INR 50,000 require at least three comparable written supplier quotations. If fewer than three suppliers can provide the item, the requester must attach a sole-source justification approved by the procurement director.", "paragraph"),
    ("MERIDIAN PROCUREMENT STANDARD", "SUPPLIER ONBOARDING", "Before the first order, Procurement must verify the supplier's tax registration, bank-account proof, legal business name, and signed code of conduct. Bank-detail changes require independent callback verification using a previously registered telephone number.", "paragraph"),
    ("MERIDIAN PROCUREMENT STANDARD", "CONFLICTS OF INTEREST", "Employees involved in supplier selection must disclose any personal, family, or financial relationship with a bidder before evaluating offers. The employee must withdraw from the evaluation until Compliance records a decision.", "paragraph"),
    ("MERIDIAN PROCUREMENT STANDARD", "GOODS RECEIPT", "The receiving employee must compare delivered quantities and visible condition with the purchase order. Record shortages or damage in the goods-received note within two business days. Do not confirm receipt for items that have not arrived.", "paragraph"),
    ("MERIDIAN PROCUREMENT STANDARD", "INVOICE MATCHING", "Accounts Payable releases payment only after a three-way match of the approved purchase order, goods-received note, and supplier invoice. A price or quantity mismatch places the invoice on hold pending buyer resolution.", "paragraph"),
    ("MERIDIAN PROCUREMENT STANDARD", "EMERGENCY PURCHASES", "Emergency purchases may proceed before the normal quotation process only when delay creates an immediate safety or operational risk. The business owner must document the reason within one business day and obtain retrospective approval within three business days.", "paragraph"),
    ("ORBIT INFORMATION SECURITY STANDARD", "ACCESS REQUESTS", "Access requests must name the system, business justification, requested role, and end date where access is temporary. The system owner approves access; a user's line manager alone cannot grant application permissions.", "paragraph"),
    ("ORBIT INFORMATION SECURITY STANDARD", "MULTI-FACTOR AUTHENTICATION", "Multi-factor authentication is mandatory for remote access, administrator accounts, and systems containing customer or employee personal data. Approved factors are a hardware security key or authenticator application; SMS is not an approved factor.", "paragraph"),
    ("ORBIT INFORMATION SECURITY STANDARD", "PASSWORDS AND SECRETS", "Human account passwords must contain at least 14 characters and must not be reused across corporate services. API keys and service credentials must be stored in the approved secrets vault, never in source code or shared documents.", "paragraph"),
    ("ORBIT INFORMATION SECURITY STANDARD", "DATA CLASSIFICATION", "Public: approved for unrestricted release. Internal: routine business information for staff. Confidential: commercial, financial, or personal information shared only with authorised roles. Restricted: highly sensitive data requiring named-user access and encryption.", "table"),
    ("ORBIT INFORMATION SECURITY STANDARD", "SECURITY INCIDENTS", "Suspected loss of a company device, credential disclosure, malware infection, or accidental data sharing must be reported to the Security Operations Centre within one hour of discovery. Preserve evidence and do not attempt to erase logs.", "paragraph"),
    ("ORBIT INFORMATION SECURITY STANDARD", "DATA RETENTION", "Customer support recordings are retained for 180 days unless a legal hold applies. Routine application logs are retained for 90 days. Legal, privacy, or security teams may issue a documented hold that suspends scheduled deletion.", "paragraph"),
    ("ORBIT INFORMATION SECURITY STANDARD", "VENDOR DATA ACCESS", "A vendor may access company data only under an executed data-processing agreement and a named business sponsor. Production access must be time-limited, approved by the system owner, and reviewed at least every 90 days.", "paragraph"),
    ("ORBIT INFORMATION SECURITY STANDARD", "BACKUPS AND RECOVERY", "Critical production databases are backed up every 4 hours. Backup copies must be encrypted and stored in a separate security account. The recovery objective is a maximum 8-hour data-loss window and restoration of service within 12 hours.", "paragraph"),
]


def create_diverse_corpus(output_path: Path) -> Path:
    """Create a 24-page, three-domain synthetic corpus with varied page layouts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open()
    for index, (document_title, heading, body, layout) in enumerate(DIVERSE_PAGES):
        page = pdf.new_page(width=595, height=842)
        page.insert_text((42, 42), document_title, fontsize=10, fontname="hebo")
        page.draw_line((42, 53), (553, 53), color=(0.25, 0.35, 0.5), width=0.8)
        page.insert_text((42, 91), heading, fontsize=16, fontname="hebo")
        if layout == "table":
            lines = body.split(". ")
            table_text = "\n".join(f"{line.strip()}." for line in lines if line.strip())
            page.insert_textbox(fitz.Rect(52, 135, 540, 410), table_text, fontsize=12, fontname="cour", lineheight=1.8)
            page.draw_rect(fitz.Rect(45, 125, 548, 420), color=(0.45, 0.5, 0.6), width=0.7)
        elif index % 3 == 1:
            page.insert_textbox(fitz.Rect(42, 135, 285, 500), body, fontsize=11, fontname="helv", lineheight=1.5)
            page.insert_textbox(fitz.Rect(310, 135, 550, 500), "CONTROL NOTES\n\nThis section is part of the same policy. Apply the stated approval, evidence, and record-keeping requirements together. If an exception is needed, retain the written decision with the transaction record.", fontsize=10, fontname="helv", lineheight=1.5)
        else:
            page.insert_textbox(fitz.Rect(48, 140, 545, 530), body, fontsize=12, fontname="helv", lineheight=1.55)
        page.insert_text((42, 790), f"Synthetic evaluation corpus - fictional policy - page {index + 1}", fontsize=8, fontname="helv")
    pdf.save(output_path)
    pdf.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic DocuMind retrieval benchmark PDF.")
    parser.add_argument("--output", type=Path, default=Path("evaluation/sample_travel_policy_v2.pdf"))
    args = parser.parse_args()
    output = create_corpus(args.output.resolve())
    print(f"Created synthetic benchmark PDF: {output}")


if __name__ == "__main__":
    main()
