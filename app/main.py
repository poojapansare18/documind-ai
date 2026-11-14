from __future__ import annotations

import os
import re
import sqlite3
import uuid
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Annotated

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="DocuMind API",
    description="A lightweight document intelligence API for PDF extraction, search, and extractive summaries.",
    version="0.2.0",
)

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
DB_PATH = Path(os.getenv("DOCUMIND_DB_PATH", "data/documind.db"))


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    character_count: int


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    page_count: int
    character_count: int
    created_at: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    document_id: str
    filename: str
    page_number: int
    snippet: str
    score: int


class DocumentDetail(BaseModel):
    document_id: str
    filename: str
    page_count: int
    character_count: int
    created_at: str


class DocumentPageResponse(BaseModel):
    document_id: str
    page_number: int
    character_count: int
    text: str


class SummaryResponse(BaseModel):
    document_id: str
    filename: str
    summary: str
    method: str = "extractive"


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "will", "with", "we", "you", "your", "our", "their", "they", "he", "she",
}


def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with closing(connect_db()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                character_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_pages (
                page_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                extracted_text TEXT NOT NULL,
                UNIQUE(document_id, page_number)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_document_pages_document_id ON document_pages(document_id)")
        connection.commit()


# Initialise the local SQLite database when the application module loads.
init_db()


def extract_pdf(data: bytes) -> tuple[list[str], int]:
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise HTTPException(status_code=422, detail="Password-protected PDFs are not supported.")
            if len(pdf) == 0:
                raise HTTPException(status_code=422, detail="The PDF contains no pages.")
            if len(pdf) > MAX_PDF_PAGES:
                raise HTTPException(status_code=413, detail=f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
            page_texts = [page.get_text("text").strip() for page in pdf]
            if not any(page_texts):
                raise HTTPException(
                    status_code=422,
                    detail="No selectable text was found. Scanned PDFs require OCR, which is not included in this version.",
                )
            return page_texts, len(pdf)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid readable PDF.") from exc


def sentence_split(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if len(part.strip()) > 25]


def extractive_summary(text: str, max_sentences: int = 5) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return text[:700]
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    frequencies = Counter(word for word in words if word not in STOP_WORDS)
    if not frequencies:
        return " ".join(sentences[:max_sentences])
    highest = max(frequencies.values())
    weights = {word: count / highest for word, count in frequencies.items()}
    scored = []
    for index, sentence in enumerate(sentences):
        tokens = re.findall(r"[a-zA-Z]{3,}", sentence.lower())
        score = sum(weights.get(token, 0) for token in tokens) / max(len(tokens), 1)
        scored.append((score, index, sentence))
    chosen = sorted(scored, reverse=True)[:max_sentences]
    chosen.sort(key=lambda item: item[1])
    return " ".join(item[2] for item in chosen)


def get_document(document_id: str) -> sqlite3.Row:
    with closing(connect_db()) as connection:
        row = connection.execute(
            "SELECT document_id, filename, extracted_text, page_count, character_count, created_at "
            "FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return row


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "documind-api", "version": "0.2.0"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DocuMind — Document Intelligence</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fb;color:#172033;font:16px system-ui,sans-serif}
main{max-width:850px;margin:7vh auto;padding:24px}.panel{background:white;border:1px solid #e2e7f0;border-radius:18px;padding:28px;box-shadow:0 12px 35px #1720330b}
h1{margin:0 0 8px;font-size:2rem}p{color:#596579;line-height:1.55}.upload{border:1px dashed #9ba8bb;border-radius:12px;padding:24px;margin:22px 0;text-align:center}
button{background:#244bd6;color:white;border:0;border-radius:9px;padding:11px 16px;font-weight:650;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
input[type=file]{max-width:100%}.result{white-space:pre-wrap;background:#f7f8fc;padding:16px;border-radius:10px;margin-top:14px;line-height:1.6}.muted{font-size:.9rem;color:#748096}
</style></head><body><main><section class="panel"><p class="muted">DOCUMENT INTELLIGENCE · MVP 0.2</p><h1>DocuMind</h1>
<p>Upload a text-based PDF to extract its contents, generate an extractive summary, and search within your saved documents.</p>
<div class="upload"><input id="file" type="file" accept="application/pdf"><p><button id="upload">Upload PDF</button></p><span class="muted">PDF only · Maximum 10 MB · 100 pages</span></div>
<div id="status" role="status" aria-live="polite" class="muted">No document uploaded yet.</div><div id="output"></div></section></main>
<script>
const file=document.querySelector('#file'),button=document.querySelector('#upload'),status=document.querySelector('#status'),output=document.querySelector('#output');
button.addEventListener('click',async()=>{if(!file.files[0]){status.textContent='Choose a PDF first.';return;}button.disabled=true;status.textContent='Processing PDF…';output.replaceChildren();
try{const data=new FormData();data.append('file',file.files[0]);const res=await fetch('/documents/upload',{method:'POST',body:data});const doc=await res.json();if(!res.ok)throw new Error(doc.detail||'Upload failed');
status.textContent='Uploaded '+doc.filename+' · '+doc.page_count+' pages · '+doc.character_count.toLocaleString()+' characters';
const summary=await fetch('/documents/'+doc.document_id+'/summary');const result=await summary.json();if(!summary.ok)throw new Error(result.detail||'Summary failed');
const box=document.createElement('div');box.className='result';box.textContent=result.summary;output.append(box);
}catch(e){status.textContent='Error: '+e.message;}finally{button.disabled=false;}});
</script></body></html>"""


@app.post("/documents/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(file: Annotated[UploadFile, File(description="A text-based PDF document")]) -> DocumentResponse:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf") or file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="The file does not have a valid PDF signature.")
    page_texts, page_count = extract_pdf(data)
    extracted_text = "\n\n".join(page_texts).strip()
    document_id = str(uuid.uuid4())
    with closing(connect_db()) as connection:
        connection.execute(
            "INSERT INTO documents (document_id, filename, extracted_text, page_count, character_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (document_id, filename, extracted_text, page_count, len(extracted_text)),
        )
        connection.executemany(
            "INSERT INTO document_pages (document_id, page_number, extracted_text) VALUES (?, ?, ?)",
            [(document_id, index, page_text) for index, page_text in enumerate(page_texts, start=1)],
        )
        connection.commit()
    return DocumentResponse(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        character_count=len(extracted_text),
    )


@app.get("/documents", response_model=list[DocumentListItem])
def list_documents() -> list[DocumentListItem]:
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT document_id, filename, page_count, character_count, created_at "
            "FROM documents ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
    return [DocumentListItem(**dict(row)) for row in rows]


@app.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document_details(document_id: str) -> DocumentDetail:
    doc = get_document(document_id)
    return DocumentDetail(
        document_id=doc["document_id"],
        filename=doc["filename"],
        page_count=doc["page_count"],
        character_count=doc["character_count"],
        created_at=doc["created_at"],
    )


@app.get("/documents/{document_id}/pages", response_model=list[DocumentPageResponse])
def get_document_pages(document_id: str) -> list[DocumentPageResponse]:
    get_document(document_id)
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT document_id, page_number, extracted_text FROM document_pages "
            "WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()
    return [
        DocumentPageResponse(
            document_id=row["document_id"],
            page_number=row["page_number"],
            character_count=len(row["extracted_text"]),
            text=row["extracted_text"],
        )
        for row in rows
    ]


@app.get("/documents/{document_id}/summary", response_model=SummaryResponse)
def summarize_document(document_id: str) -> SummaryResponse:
    doc = get_document(document_id)
    return SummaryResponse(
        document_id=doc["document_id"],
        filename=doc["filename"],
        summary=extractive_summary(doc["extracted_text"]),
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    with closing(connect_db()) as connection:
        connection.execute("DELETE FROM document_pages WHERE document_id = ?", (document_id,))
        cursor = connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        connection.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "status": "deleted"}


@app.post("/documents/search", response_model=list[SearchResult])
def search_documents(request: SearchRequest) -> list[SearchResult]:
    terms = [term for term in re.findall(r"[a-zA-Z0-9]{2,}", request.query.lower()) if term not in STOP_WORDS]
    if not terms:
        raise HTTPException(status_code=422, detail="Search query must contain at least one meaningful term.")
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT p.document_id, d.filename, p.page_number, p.extracted_text "
            "FROM document_pages p JOIN documents d ON d.document_id = p.document_id"
        ).fetchall()
    results: list[SearchResult] = []
    for page in rows:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page["extracted_text"]) if part.strip()]
        for paragraph in paragraphs:
            lowered = paragraph.lower()
            score = sum(lowered.count(term) for term in terms)
            if score:
                snippet = paragraph[:500] + ("…" if len(paragraph) > 500 else "")
                results.append(
                    SearchResult(
                        document_id=page["document_id"],
                        filename=page["filename"],
                        page_number=page["page_number"],
                        snippet=snippet,
                        score=score,
                    )
                )
    return sorted(results, key=lambda result: result.score, reverse=True)[: request.limit]
