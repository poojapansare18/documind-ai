import fitz
from fastapi.testclient import TestClient

from app.main import app, documents, extractive_summary

client = TestClient(app)


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def setup_function():
    documents.clear()


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_home_page_loads():
    response = client.get('/')
    assert response.status_code == 200
    assert 'DocuMind' in response.text


def test_upload_and_summary():
    content = ('Document intelligence helps teams understand business information. ' * 8).strip()
    response = client.post('/documents/upload', files={'file': ('sample.pdf', make_pdf(content), 'application/pdf')})
    assert response.status_code == 201
    document_id = response.json()['document_id']
    summary = client.get(f'/documents/{document_id}/summary')
    assert summary.status_code == 200
    assert summary.json()['summary']


def test_reject_non_pdf():
    response = client.post('/documents/upload', files={'file': ('notes.txt', b'hello', 'text/plain')})
    assert response.status_code == 415


def test_search_returns_matching_document():
    content = ('Automation reduces manual invoice processing time. ' * 8).strip()
    upload = client.post('/documents/upload', files={'file': ('invoice.pdf', make_pdf(content), 'application/pdf')})
    assert upload.status_code == 201
    response = client.post('/documents/search', json={'query': 'invoice automation'})
    assert response.status_code == 200
    assert response.json()[0]['filename'] == 'invoice.pdf'


def test_summary_empty_input_is_safe():
    assert isinstance(extractive_summary(''), str)


def test_reject_invalid_pdf_bytes():
    response = client.post('/documents/upload', files={'file': ('broken.pdf', b'not-a-pdf', 'application/pdf')})
    assert response.status_code == 415


def test_unknown_document_returns_404():
    response = client.get('/documents/does-not-exist/summary')
    assert response.status_code == 404
