import fitz
from fastapi.testclient import TestClient

import app.main as main
from app.chunking import chunk_text
from app.generation import ABSTENTION, validate_answer_citations
from app.layout import detect_table_candidates, reconstruct_reading_order
from app.tables import _table_to_payload

client = TestClient(main.app)


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def setup_function():
    with main.connect_db() as connection:
        connection.execute("DELETE FROM document_pages")
        connection.execute("DELETE FROM documents")
        connection.commit()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_home_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "DocuMind" in response.text


def test_upload_and_summary():
    content = ("Document intelligence helps teams understand business information. " * 8).strip()
    response = client.post(
        "/documents/upload",
        files={"file": ("sample.pdf", make_pdf(content), "application/pdf")},
    )
    assert response.status_code == 201
    document_id = response.json()["document_id"]
    summary = client.get(f"/documents/{document_id}/summary")
    assert summary.status_code == 200
    assert summary.json()["summary"]
    assert summary.json()["method"] == "extractive"


def test_reject_non_pdf():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_search_returns_matching_document():
    content = ("Automation reduces manual invoice processing time. " * 8).strip()
    upload = client.post(
        "/documents/upload",
        files={"file": ("invoice.pdf", make_pdf(content), "application/pdf")},
    )
    assert upload.status_code == 201
    response = client.post("/documents/search", json={"query": "invoice automation"})
    assert response.status_code == 200
    assert response.json()[0]["filename"] == "invoice.pdf"


def test_summary_empty_input_is_safe():
    assert isinstance(main.extractive_summary(""), str)


def test_reject_invalid_pdf_bytes():
    response = client.post(
        "/documents/upload",
        files={"file": ("broken.pdf", b"not-a-pdf", "application/pdf")},
    )
    assert response.status_code == 415


def test_unknown_document_returns_404():
    response = client.get("/documents/does-not-exist/summary")
    assert response.status_code == 404


def test_documents_are_listed():
    upload = client.post(
        "/documents/upload",
        files={"file": ("listed.pdf", make_pdf("This document should appear in the document list."), "application/pdf")},
    )
    assert upload.status_code == 201
    response = client.get("/documents")
    assert response.status_code == 200
    assert any(item["document_id"] == upload.json()["document_id"] for item in response.json())


def test_document_can_be_deleted():
    upload = client.post(
        "/documents/upload",
        files={"file": ("delete-me.pdf", make_pdf("This document will be deleted."), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert client.get(f"/documents/{document_id}/summary").status_code == 404


def test_deleting_unknown_document_returns_404():
    response = client.delete("/documents/missing-document")
    assert response.status_code == 404


def test_search_rejects_stop_words_only():
    response = client.post("/documents/search", json={"query": "the and of"})
    assert response.status_code == 422


def test_upload_rejects_wrong_pdf_content_type():
    response = client.post(
        "/documents/upload",
        files={"file": ("wrong.pdf", b"%PDF-1.4", "text/plain")},
    )
    assert response.status_code == 415



def test_upload_persists_page_level_text():
    content = "Page-level extraction keeps source locations available for retrieval."
    upload = client.post(
        "/documents/upload",
        files={"file": ("pages.pdf", make_pdf(content), "application/pdf")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["document_id"]
    pages = client.get(f"/documents/{document_id}/pages")
    assert pages.status_code == 200
    assert len(pages.json()) == 1
    assert pages.json()[0]["page_number"] == 1
    assert content in pages.json()[0]["text"]


def test_document_detail_returns_metadata():
    upload = client.post(
        "/documents/upload",
        files={"file": ("detail.pdf", make_pdf("Document metadata is available."), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    detail = client.get(f"/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["filename"] == "detail.pdf"
    assert detail.json()["page_count"] == 1
    assert detail.json()["created_at"]


def test_search_results_include_page_number():
    upload = client.post(
        "/documents/upload",
        files={"file": ("source.pdf", make_pdf("Semantic retrieval needs traceable source references."), "application/pdf")},
    )
    response = client.post("/documents/search", json={"query": "traceable source"})
    assert response.status_code == 200
    assert response.json()[0]["document_id"] == upload.json()["document_id"]
    assert response.json()[0]["page_number"] == 1


def test_deleting_document_removes_its_pages():
    upload = client.post(
        "/documents/upload",
        files={"file": ("remove-pages.pdf", make_pdf("Page records should be removed with their document."), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    assert client.get(f"/documents/{document_id}/pages").json()
    assert client.delete(f"/documents/{document_id}").status_code == 200
    assert client.get(f"/documents/{document_id}/pages").status_code == 404



def test_upload_persists_layout_geometry():
    content = "Layout geometry preserves the position of text on a PDF page."
    upload = client.post(
        "/documents/upload",
        files={"file": ("layout.pdf", make_pdf(content), "application/pdf")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["document_id"]
    layout = client.get(f"/documents/{document_id}/layout")
    assert layout.status_code == 200
    assert len(layout.json()) == 1
    page = layout.json()[0]
    assert page["page_width"] > 0
    assert page["page_height"] > 0
    assert page["blocks"]
    block = page["blocks"][0]
    assert content in block["text"]
    assert block["lines"]
    assert block["lines"][0]["spans"]
    assert block["lines"][0]["spans"][0]["font_size"] > 0


def test_layout_endpoint_returns_404_for_unknown_document():
    response = client.get("/documents/unknown-document/layout")
    assert response.status_code == 404


def test_table_candidate_detection_uses_repeated_aligned_rows():
    words = []
    rows = [
        (10, [("Code", 10, 40), ("Title", 100, 135), ("Qty", 220, 245)]),
        (30, [("A01", 10, 30), ("AI Systems", 100, 155), ("12", 220, 232)]),
        (50, [("A02", 10, 30), ("Data Science", 100, 165), ("8", 220, 228)]),
    ]
    for line_no, (y, cells) in enumerate(rows):
        for word_no, (text, x0, x1) in enumerate(cells):
            words.append({
                "x0": x0, "y0": y, "x1": x1, "y1": y + 10,
                "text": text, "block_no": word_no, "line_no": 0, "word_no": word_no,
            })
    candidates = detect_table_candidates(words)
    assert len(candidates) == 1
    assert candidates[0]["row_count"] == 3
    assert candidates[0]["column_count"] == 3
    assert candidates[0]["rows"][1]["cells"][1]["text"] == "AI Systems"


def test_table_candidate_detection_ignores_single_paragraph_line():
    words = [
        {"x0": 10, "y0": 10, "x1": 35, "y1": 20, "text": "This", "block_no": 0, "line_no": 0, "word_no": 0},
        {"x0": 40, "y0": 10, "x1": 80, "y1": 20, "text": "is", "block_no": 0, "line_no": 0, "word_no": 1},
        {"x0": 85, "y0": 10, "x1": 130, "y1": 20, "text": "normal", "block_no": 0, "line_no": 0, "word_no": 2},
        {"x0": 135, "y0": 10, "x1": 175, "y1": 20, "text": "text", "block_no": 0, "line_no": 0, "word_no": 3},
    ]
    assert detect_table_candidates(words) == []


def test_document_tables_endpoint_returns_page_results():
    upload = client.post(
        "/documents/upload",
        files={"file": ("plain.pdf", make_pdf("This is a simple paragraph without a table."), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    response = client.get(f"/documents/{document_id}/tables")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["page_number"] == 1
    assert response.json()[0]["tables"] == []



def test_chunk_text_preserves_offsets_and_overlap():
    text = ("A useful sentence about document retrieval and source citations. " * 40).strip()
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    for item in chunks:
        assert item["text"] == text[item["start_char"]:item["end_char"]]
        assert item["end_char"] - item["start_char"] == len(item["text"])
        assert len(item["text"]) <= 300
    assert chunks[1]["start_char"] < chunks[0]["end_char"]


def test_chunk_text_handles_empty_and_short_text():
    assert chunk_text("") == []
    assert chunk_text("A short page.") == [{
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 13,
        "text": "A short page.",
    }]


def test_chunk_text_rejects_invalid_configuration():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("Some text", max_chars=50)
    with pytest.raises(ValueError):
        chunk_text("Some text", max_chars=100, overlap=100)


def test_upload_persists_page_aware_chunks():
    content = ("Chunk retrieval preserves source page references. " * 50).strip()
    upload = client.post(
        "/documents/upload",
        files={"file": ("chunked.pdf", make_pdf(content), "application/pdf")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["document_id"]
    response = client.get(f"/documents/{document_id}/chunks")
    assert response.status_code == 200
    chunks = response.json()
    assert chunks
    assert all(item["page_number"] == 1 for item in chunks)
    assert all(item["document_id"] == document_id for item in chunks)
    assert all(item["text"] for item in chunks)
    assert all(item["end_char"] - item["start_char"] == item["character_count"] for item in chunks)


def test_chunks_endpoint_returns_404_for_unknown_document():
    response = client.get("/documents/missing-document/chunks")
    assert response.status_code == 404


def test_deleting_document_cascades_to_chunks():
    content = ("Delete this document and its chunks together. " * 40).strip()
    upload = client.post(
        "/documents/upload",
        files={"file": ("delete-chunks.pdf", make_pdf(content), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    assert client.get(f"/documents/{document_id}/chunks").json()
    assert client.delete(f"/documents/{document_id}").status_code == 200
    with main.connect_db() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE document_id = ?",
            (document_id,),
        ).fetchone()[0]
    assert count == 0



def test_existing_page_records_are_backfilled_into_chunks():
    legacy_text = ("Legacy documents receive chunks during database migration. " * 30).strip()
    with main.connect_db() as connection:
        connection.execute(
            "INSERT INTO documents (document_id, filename, extracted_text, page_count, character_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("legacy-doc", "legacy.pdf", legacy_text, 1, len(legacy_text)),
        )
        connection.execute(
            "INSERT INTO document_pages (document_id, page_number, extracted_text, layout_json) "
            "VALUES (?, ?, ?, ?)",
            ("legacy-doc", 1, legacy_text, "{}"),
        )
        connection.commit()

    main.init_db()
    response = client.get("/documents/legacy-doc/chunks")
    assert response.status_code == 200
    assert response.json()
    assert response.json()[0]["text"] in legacy_text



def test_semantic_search_requires_indexing(monkeypatch):
    monkeypatch.setattr(main, "embed_query", lambda query: [1.0, 0.0])
    upload = client.post(
        "/documents/upload",
        files={"file": ("unindexed.pdf", make_pdf("Semantic retrieval needs stored embeddings."), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    response = client.post(
        "/documents/semantic-search",
        json={"query": "find stored embeddings", "document_id": document_id},
    )
    assert response.status_code == 409
    assert "index" in response.json()["detail"].lower()


def test_index_and_semantic_search_return_source_provenance(monkeypatch):
    invoice = client.post(
        "/documents/upload",
        files={"file": ("invoice.pdf", make_pdf("Invoice automation extracts supplier names and totals."), "application/pdf")},
    )
    garden = client.post(
        "/documents/upload",
        files={"file": ("garden.pdf", make_pdf("Gardening requires soil, sunlight, and regular watering."), "application/pdf")},
    )
    invoice_id = invoice.json()["document_id"]
    garden_id = garden.json()["document_id"]

    def fake_passage_embeddings(texts):
        return [[1.0, 0.0] if "invoice" in text.lower() else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(main, "embed_passages", fake_passage_embeddings)
    monkeypatch.setattr(main, "embed_query", lambda query: [1.0, 0.0])

    invoice_index = client.post(f"/documents/{invoice_id}/index")
    garden_index = client.post(f"/documents/{garden_id}/index")
    assert invoice_index.status_code == 200
    assert garden_index.status_code == 200
    assert invoice_index.json()["indexed_chunks"] == 1
    assert invoice_index.json()["dimensions"] == 2

    response = client.post(
        "/documents/semantic-search",
        json={"query": "find invoice processing details", "limit": 5},
    )
    assert response.status_code == 200
    assert response.json()
    assert response.json()[0]["document_id"] == invoice_id
    assert response.json()[0]["filename"] == "invoice.pdf"
    assert response.json()[0]["page_number"] == 1
    assert response.json()[0]["score"] == 1.0
    assert response.json()[0]["embedding_model"] == main.MODEL_NAME


def test_indexing_replaces_existing_vectors(monkeypatch):
    upload = client.post(
        "/documents/upload",
        files={"file": ("reindex.pdf", make_pdf("Reindexing replaces old vectors safely."), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    monkeypatch.setattr(main, "embed_passages", lambda texts: [[0.5, 0.5] for _ in texts])

    first = client.post(f"/documents/{document_id}/index")
    second = client.post(f"/documents/{document_id}/index")
    assert first.status_code == 200
    assert second.status_code == 200
    with main.connect_db() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM chunk_embeddings e "
            "JOIN document_chunks c ON c.chunk_id = e.chunk_id "
            "WHERE c.document_id = ?",
            (document_id,),
        ).fetchone()[0]
    assert count == first.json()["indexed_chunks"]


def test_indexing_returns_service_unavailable_when_model_fails(monkeypatch):
    upload = client.post(
        "/documents/upload",
        files={"file": ("model-error.pdf", make_pdf("Embedding model failure is handled."), "application/pdf")},
    )
    monkeypatch.setattr(main, "embed_passages", lambda texts: (_ for _ in ()).throw(RuntimeError("model unavailable")))
    response = client.post(f"/documents/{upload.json()['document_id']}/index")
    assert response.status_code == 503


def test_semantic_search_returns_404_for_unknown_document():
    response = client.post(
        "/documents/semantic-search",
        json={"query": "search missing source", "document_id": "unknown-document"},
    )
    assert response.status_code == 404


def test_cosine_similarity_handles_normal_and_invalid_vectors():
    assert main.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert main.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert main.cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert main.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0



def test_ask_documents_returns_grounded_answer_and_source_citations(monkeypatch):
    upload = client.post(
        "/documents/upload",
        files={"file": ("invoice-source.pdf", make_pdf("The invoice lists supplier names, invoice dates, and total amounts."), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    monkeypatch.setattr(main, "embed_passages", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(main, "embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        lambda question, sources: "The invoice contains supplier names, dates, and total amounts [S1].",
    )

    indexed = client.post(f"/documents/{document_id}/index")
    assert indexed.status_code == 200
    response = client.post(
        "/documents/ask",
        json={"question": "Which details does the invoice list?", "document_id": document_id},
    )
    assert response.status_code == 200
    result = response.json()
    assert "[S1]" in result["answer"]
    assert result["sources"][0]["source_id"] == "S1"
    assert result["sources"][0]["filename"] == "invoice-source.pdf"
    assert result["sources"][0]["page_number"] == 1
    assert "supplier names" in result["sources"][0]["text"]


def test_ask_documents_requires_indexed_evidence(monkeypatch):
    upload = client.post(
        "/documents/upload",
        files={"file": ("not-indexed.pdf", make_pdf("This document has not been indexed yet."), "application/pdf")},
    )
    monkeypatch.setattr(main, "embed_query", lambda query: [1.0, 0.0])
    response = client.post(
        "/documents/ask",
        json={"question": "What does this document contain?", "document_id": upload.json()["document_id"]},
    )
    assert response.status_code == 409


def test_ask_documents_handles_generation_provider_failure(monkeypatch):
    upload = client.post(
        "/documents/upload",
        files={"file": ("generation-error.pdf", make_pdf("The document contains a provider failure example."), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    monkeypatch.setattr(main, "embed_passages", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(main, "embed_query", lambda query: [1.0, 0.0])
    monkeypatch.setattr(
        main,
        "generate_grounded_answer",
        lambda question, sources: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    assert client.post(f"/documents/{document_id}/index").status_code == 200
    response = client.post(
        "/documents/ask",
        json={"question": "What is in this document?", "document_id": document_id},
    )
    assert response.status_code == 503



def test_rag_citation_validation_rejects_missing_or_unknown_sources():
    assert validate_answer_citations("A factual answer without citations.", 2) == ABSTENTION
    assert validate_answer_citations("A claim with an unknown source [S3].", 2) == ABSTENTION
    assert validate_answer_citations("A claim supported by source [S2].", 2) == "A claim supported by source [S2]."
    assert validate_answer_citations(ABSTENTION, 0) == ABSTENTION



def test_extract_page_text_uses_ocr_for_image_only_page():
    class FakePage:
        def get_text(self, option, textpage=None):
            if option == "text" and textpage is None:
                return ""
            if option == "text" and textpage == "ocr-result":
                return "Recognised text from a scanned document page."
            return ""

        def get_images(self, full=True):
            return [(1, 0, 100, 100, 8, "DeviceRGB", "", "Im1", "DCTDecode", 0)]

        def get_drawings(self):
            return []

        def get_textpage_ocr(self, language, dpi, full):
            assert language == main.OCR_LANGUAGE
            assert dpi == main.OCR_DPI
            assert full is True
            return "ocr-result"

    text, text_page, method = main.extract_page_text(FakePage())
    assert text == "Recognised text from a scanned document page."
    assert text_page == "ocr-result"
    assert method == "ocr"


def test_extract_page_text_skips_ocr_for_native_text():
    class FakePage:
        def get_text(self, option, textpage=None):
            return "This is a sufficiently long native text layer for extraction."

        def get_images(self, full=True):
            return []

        def get_drawings(self):
            return []

        def get_textpage_ocr(self, **kwargs):
            raise AssertionError("OCR should not run for a normal text-based page.")

    text, text_page, method = main.extract_page_text(FakePage())
    assert text.startswith("This is a sufficiently long")
    assert text_page is None
    assert method == "native"


def test_extract_page_text_returns_clear_error_when_ocr_engine_is_missing():
    import pytest
    from fastapi import HTTPException

    class FakePage:
        def get_text(self, option, textpage=None):
            return ""

        def get_images(self, full=True):
            return [(1, 0, 100, 100, 8, "DeviceRGB", "", "Im1", "DCTDecode", 0)]

        def get_drawings(self):
            return []

        def get_textpage_ocr(self, **kwargs):
            raise RuntimeError("Tesseract is not installed")

    with pytest.raises(HTTPException) as error:
        main.extract_page_text(FakePage())
    assert error.value.status_code == 503
    assert "Tesseract" in error.value.detail


def test_page_api_exposes_ocr_provenance(monkeypatch):
    monkeypatch.setattr(
        main,
        "extract_pdf",
        lambda data: (
            ["Recognised scanned page content."],
            1,
            [{
                "page_width": 595.0,
                "page_height": 842.0,
                "blocks": [],
                "words": [],
                "extraction_method": "ocr",
                "ocr_applied": True,
            }],
        ),
    )
    upload = client.post(
        "/documents/upload",
        files={"file": ("scanned.pdf", make_pdf("placeholder"), "application/pdf")},
    )
    assert upload.status_code == 201
    pages = client.get(f"/documents/{upload.json()['document_id']}/pages")
    assert pages.status_code == 200
    assert pages.json()[0]["extraction_method"] == "ocr"
    assert pages.json()[0]["ocr_applied"] is True
    assert pages.json()[0]["text"] == "Recognised scanned page content."



def test_extract_page_text_does_not_ocr_a_blank_page():
    class FakePage:
        def get_text(self, option, textpage=None):
            return ""

        def get_images(self, full=True):
            return []

        def get_drawings(self):
            return []

        def get_textpage_ocr(self, **kwargs):
            raise AssertionError("A blank page should not trigger OCR.")

    text, text_page, method = main.extract_page_text(FakePage())
    assert text == ""
    assert text_page is None
    assert method == "native"



def test_scanned_pdf_is_ocr_processed_end_to_end():
    import shutil
    import pytest

    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract is not installed in this environment.")

    source = fitz.open()
    source_page = source.new_page()
    source_page.insert_text((72, 90), "SCANNED INVOICE NUMBER 12345")
    pixmap = source_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

    scanned = fitz.open()
    scanned_page = scanned.new_page(width=source_page.rect.width, height=source_page.rect.height)
    scanned_page.insert_image(scanned_page.rect, stream=pixmap.tobytes("png"))
    pdf_bytes = scanned.tobytes()
    source.close()
    scanned.close()

    page_texts, page_count, layouts = main.extract_pdf(pdf_bytes)
    assert page_count == 1
    assert "SCANNED" in page_texts[0].upper()
    assert "12345" in page_texts[0]
    assert layouts[0]["extraction_method"] == "ocr"
    assert layouts[0]["ocr_applied"] is True



def make_two_column_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    for index, text in enumerate(["LEFT COLUMN FIRST", "LEFT COLUMN SECOND", "LEFT COLUMN THIRD"]):
        page.insert_text((50, 100 + index * 24), text)
    for index, text in enumerate(["RIGHT COLUMN FIRST", "RIGHT COLUMN SECOND", "RIGHT COLUMN THIRD"]):
        page.insert_text((330, 100 + index * 24), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_two_column_pdf_is_reconstructed_left_column_then_right():
    page_texts, page_count, layouts = main.extract_pdf(make_two_column_pdf())
    assert page_count == 1
    assert layouts[0]["layout_type"] == "two_column"
    assert layouts[0]["column_count"] == 2
    expected = [
        "LEFT COLUMN FIRST",
        "LEFT COLUMN SECOND",
        "LEFT COLUMN THIRD",
        "RIGHT COLUMN FIRST",
        "RIGHT COLUMN SECOND",
        "RIGHT COLUMN THIRD",
    ]
    assert page_texts[0].splitlines() == expected


def test_two_column_reconstruction_preserves_full_width_heading_and_footer():
    words = []

    def add_line(text, x0, y, width):
        words.append({
            "x0": x0, "y0": y, "x1": x0 + width, "y1": y + 10,
            "text": text, "block_no": 0, "line_no": 0, "word_no": 0,
        })

    add_line("REPORT TITLE", 50, 10, 500)
    add_line("LEFT ONE", 50, 100, 100)
    add_line("LEFT TWO", 50, 120, 100)
    add_line("RIGHT ONE", 330, 100, 100)
    add_line("RIGHT TWO", 330, 120, 100)
    add_line("FULL WIDTH FOOTER", 50, 500, 500)

    result = reconstruct_reading_order(words, page_width=600, fallback_text="native fallback")
    assert result["layout_type"] == "two_column"
    assert result["column_count"] == 2
    assert result["text"].splitlines() == [
        "REPORT TITLE",
        "LEFT ONE",
        "LEFT TWO",
        "RIGHT ONE",
        "RIGHT TWO",
        "FULL WIDTH FOOTER",
    ]


def test_single_column_page_keeps_native_reading_order():
    words = [
        {"x0": 50, "y0": 10, "x1": 180, "y1": 20, "text": "FIRST LINE"},
        {"x0": 50, "y0": 30, "x1": 190, "y1": 40, "text": "SECOND LINE"},
        {"x0": 50, "y0": 50, "x1": 170, "y1": 60, "text": "THIRD LINE"},
    ]
    result = reconstruct_reading_order(words, page_width=600, fallback_text="FIRST\nSECOND\nTHIRD")
    assert result["layout_type"] == "single_column"
    assert result["column_count"] == 1
    assert result["text"] == "FIRST\nSECOND\nTHIRD"


def test_uploaded_two_column_pdf_exposes_layout_classification():
    upload = client.post(
        "/documents/upload",
        files={"file": ("two-columns.pdf", make_two_column_pdf(), "application/pdf")},
    )
    assert upload.status_code == 201
    document_id = upload.json()["document_id"]
    pages = client.get(f"/documents/{document_id}/pages")
    assert pages.status_code == 200
    assert pages.json()[0]["layout_type"] == "two_column"
    assert pages.json()[0]["column_count"] == 2
    assert pages.json()[0]["text"].index("LEFT COLUMN THIRD") < pages.json()[0]["text"].index("RIGHT COLUMN FIRST")


def draw_ruled_table(page, top: float, rows: list[list[str]]) -> None:
    x_edges = [50, 210, 370, 530]
    row_height = 34
    bottom = top + row_height * len(rows)
    for y in [top + index * row_height for index in range(len(rows) + 1)]:
        page.draw_line((x_edges[0], y), (x_edges[-1], y), color=(0, 0, 0), width=0.8)
    for x in x_edges:
        page.draw_line((x, top), (x, bottom), color=(0, 0, 0), width=0.8)
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            page.insert_text(
                (x_edges[column_index] + 6, top + row_index * row_height + 22),
                value,
                fontsize=10,
            )


def make_two_page_continuation_pdf() -> bytes:
    doc = fitz.open()
    first = doc.new_page(width=600, height=800)
    second = doc.new_page(width=600, height=800)
    draw_ruled_table(first, 650, [
        ["Item", "Description", "Qty"],
        ["A01", "First page row", "12"],
        ["A02", "Last page row", "8"],
    ])
    draw_ruled_table(second, 30, [
        ["Item", "Description", "Qty"],
        ["A03", "Continued row", "5"],
        ["A04", "Final row", "2"],
    ])
    data = doc.tobytes()
    doc.close()
    return data


def test_native_table_adapter_preserves_multiline_and_estimates_merged_spans():
    class FakeRow:
        def __init__(self, cells):
            self.cells = cells

    class FakeHeader:
        names = ["Category", "Description", "Amount"]
        external = False

    class FakeTable:
        col_count = 3
        bbox = (10, 10, 150, 70)
        header = FakeHeader()
        cells = [
            (10, 10, 50, 70), (50, 10, 150, 40), None,
            None, (50, 40, 100, 70), (100, 40, 150, 70),
        ]
        rows = [
            FakeRow([(10, 10, 50, 70), (50, 10, 150, 40), None]),
            FakeRow([None, (50, 40, 100, 70), (100, 40, 150, 70)]),
        ]

        def extract(self):
            return [
                ["Category", "Long description\\ncontinued", None],
                [None, "Service", "50"],
            ]

    from app.tables import _table_to_payload

    result = _table_to_payload(FakeTable(), "lines")
    assert result is not None
    assert result["rows"][0]["cells"][1]["text"] == "Long description\\ncontinued"
    assert result["rows"][0]["cells"][1]["column_span"] == 2
    assert result["rows"][0]["cells"][0]["row_span"] == 2
    assert result["rows"][0]["cells"][1]["column_index"] == 1
    assert result["header_names"] == ["Category", "Description", "Amount"]

def test_ruled_pdf_table_is_extracted_with_native_geometry():
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    draw_ruled_table(page, 100, [
        ["Item", "Description", "Qty"],
        ["A01", "AI Systems", "12"],
        ["A02", "Data Science", "8"],
    ])
    pdf_bytes = doc.tobytes()
    doc.close()

    _, page_count, layouts = main.extract_pdf(pdf_bytes)
    assert page_count == 1
    tables = layouts[0]["tables"]
    assert tables
    assert tables[0]["detection_method"] == "pymupdf_lines"
    assert tables[0]["row_count"] >= 3
    assert tables[0]["column_count"] == 3
    assert tables[0]["bbox"] is not None
    assert tables[0]["rows"][0]["cells"][0]["text"] == "Item"


def test_table_endpoint_exposes_native_table_structure():
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    draw_ruled_table(page, 100, [
        ["Item", "Description", "Qty"],
        ["A01", "AI Systems", "12"],
        ["A02", "Data Science", "8"],
    ])
    pdf_bytes = doc.tobytes()
    doc.close()
    upload = client.post(
        "/documents/upload",
        files={"file": ("catalog-table.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload.status_code == 201
    response = client.get(f"/documents/{upload.json()['document_id']}/tables")
    assert response.status_code == 200
    table = response.json()[0]["tables"][0]
    assert table["detection_method"] == "pymupdf_lines"
    assert table["rows"][1]["cells"][1]["text"] == "AI Systems"
    assert table["rows"][1]["cells"][1]["column_index"] == 1


def test_tables_endpoint_links_likely_page_continuations():
    upload = client.post(
        "/documents/upload",
        files={"file": ("continued-table.pdf", make_two_page_continuation_pdf(), "application/pdf")},
    )
    assert upload.status_code == 201
    response = client.get(f"/documents/{upload.json()['document_id']}/tables")
    assert response.status_code == 200
    first, second = response.json()
    assert first["tables"] and second["tables"]
    assert first["tables"][-1]["continues_on_next_page"] is True
    assert second["tables"][0]["continues_from_previous_page"] is True
    assert first["tables"][-1]["continuation_group"] == second["tables"][0]["continuation_group"]



def test_borderless_pdf_table_uses_text_alignment_strategy():
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    rows = [
        ["Code", "Description", "Quantity"],
        ["A01", "AI Systems", "12"],
        ["A02", "Data Science", "8"],
    ]
    x_positions = [50, 220, 430]
    for row_index, row in enumerate(rows):
        y = 120 + row_index * 32
        for column_index, value in enumerate(row):
            page.insert_text((x_positions[column_index], y), value, fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()

    _, page_count, layouts = main.extract_pdf(pdf_bytes)
    assert page_count == 1
    tables = layouts[0]["tables"]
    assert tables
    assert any(table["detection_method"] == "pymupdf_text" for table in tables)
    assert any(table["column_count"] == 3 for table in tables)
