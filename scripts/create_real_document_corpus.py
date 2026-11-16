from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

import pymupdf as fitz

# Official NIST-hosted derivative of SP 800-53 Rev. 5 controls.
# The source PDF contains the control catalogue only (not the full publication's
# background/implementation guidance); see the benchmark README for provenance.
SOURCE_URL = (
    "https://csrc.nist.gov/CSRC/media/Projects/risk-management/"
    "800-53%20Downloads/800-53r5/SP_800-53_v5_1-derived-OSCAL.pdf"
)
SOURCE_PDF_PAGES = [19, 24, 39, 69, 82, 100, 131, 143, 158, 163, 313]


def create_real_nist_corpus(output_path: Path, source_path: Path | None = None) -> Path:
    """Download NIST SP 800-53 Rev. 5 controls and extract cited source pages.

    Output page numbers are compact 1-based positions. SOURCE_PDF_PAGES records
    the corresponding original 1-based PDF pages in the official NIST source.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path is None:
        source_path = output_path.parent / "source_nist_sp800_53_rev5.pdf"
        if not source_path.exists():
            request = Request(SOURCE_URL, headers={"User-Agent": "DocuMindAI-benchmark/1.0"})
            with urlopen(request, timeout=60) as response:
                source_path.write_bytes(response.read())

    source = fitz.open(source_path)
    try:
        if len(source) < max(SOURCE_PDF_PAGES):
            raise ValueError(
                f"Expected at least {max(SOURCE_PDF_PAGES)} source pages; found {len(source)}"
            )
        output = fitz.open()
        try:
            for source_page_number in SOURCE_PDF_PAGES:
                output.insert_pdf(
                    source,
                    from_page=source_page_number - 1,
                    to_page=source_page_number - 1,
                )
            output.set_metadata({
                "title": "DocuMind Real-Document Retrieval Benchmark v4",
                "author": "DocuMind AI benchmark",
                "subject": "Selected pages from NIST SP 800-53 Rev. 5.1 controls",
                "keywords": "NIST, SP 800-53, security controls, retrieval benchmark",
            })
            output.save(output_path, garbage=4, deflate=True)
        finally:
            output.close()
    finally:
        source.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the real-document NIST v4 retrieval corpus."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/sample_nist_sp800_53_v4.pdf"),
    )
    parser.add_argument("--source-pdf", type=Path, default=None)
    args = parser.parse_args()
    output = create_real_nist_corpus(args.output.resolve(), args.source_pdf)
    print(f"Created real-document benchmark PDF: {output}")
    print(f"Selected original NIST PDF pages: {', '.join(map(str, SOURCE_PDF_PAGES))}")


if __name__ == "__main__":
    main()
