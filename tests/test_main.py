import fitz
from fastapi.testclient import TestClient

import app.main as main
from app.chunking import chunk_text
from app.layout import detect_table_candidates

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
