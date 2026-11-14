import fitz
from fastapi.testclient import TestClient

import app.main as main

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
    assert content in page["blocks"][0]["text"]


def test_layout_endpoint_returns_404_for_unknown_document():
    response = client.get("/documents/unknown-document/layout")
    assert response.status_code == 404
