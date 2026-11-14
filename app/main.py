from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import uuid
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Annotated

import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.chunking import chunk_text
from app.embeddings import MODEL_NAME, embed_passages, embed_query
from app.generation import generate_grounded_answer
from app.layout import detect_table_candidates, reconstruct_reading_order
from app.tables import extract_native_tables

load_dotenv()

app = FastAPI(
    title="DocuMind API",
    description="A document intelligence API for PDF extraction, OCR, layout-aware reading order, keyword search, semantic retrieval, and grounded question answering.",
    version="0.10.0",
)

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
OCR_MIN_NATIVE_CHARS = 30
OCR_LANGUAGE = os.getenv("DOCUMIND_OCR_LANGUAGE", "eng")
OCR_DPI = int(os.getenv("DOCUMIND_OCR_DPI", "200"))
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP = 150
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
    extraction_method: str = "native"
    ocr_applied: bool = False
    layout_type: str = "single_column"
    column_count: int = 1


class LayoutSpan(BaseModel):
    bbox: list[float]
    text: str
    font_name: str
    font_size: float


class LayoutLine(BaseModel):
    bbox: list[float]
    text: str
    spans: list[LayoutSpan]


class LayoutBlock(BaseModel):
    block_index: int
    bbox: list[float]
    text: str
    line_count: int
    font_names: list[str]
    lines: list[LayoutLine]


class TableCell(BaseModel):
    text: str
    bbox: list[float] | None = None
    column_index: int = 0
    column_span: int = 1


class TableRow(BaseModel):
    row_index: int = 0
    cells: list[TableCell]


class TableCandidate(BaseModel):
    row_count: int
    column_count: int
    confidence: float | None = None
    detection_method: str = "heuristic"
    bbox: list[float] | None = None
    header_names: list[str] = Field(default_factory=list)
    header_external: bool = False
    continues_from_previous_page: bool = False
    continues_on_next_page: bool = False
    continuation_group: str | None = None
    rows: list[TableRow]


class DocumentPageTablesResponse(BaseModel):
    document_id: str
    page_number: int
    tables: list[TableCandidate]


class DocumentPageLayoutResponse(BaseModel):
    document_id: str
    page_number: int
    page_width: float
    page_height: float
    blocks: list[LayoutBlock]
    extraction_method: str = "native"
    ocr_applied: bool = False
    layout_type: str = "single_column"
    column_count: int = 1


class DocumentChunkResponse(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    character_count: int
    text: str


class EmbeddingIndexResponse(BaseModel):
    document_id: str
    indexed_chunks: int
    embedding_model: str
    dimensions: int


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    limit: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None


class SemanticSearchResult(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    score: float
    text: str
    embedding_model: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    limit: int = Field(default=5, ge=1, le=8)
    document_id: str | None = None


class RAGSource(BaseModel):
    source_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSource]


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
    connection.execute("PRAGMA foreign_keys = ON")
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
                layout_json TEXT NOT NULL DEFAULT '[]',
                UNIQUE(document_id, page_number)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                UNIQUE(document_id, page_number, chunk_index)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_document_page "
            "ON document_chunks(document_id, page_number, chunk_index)"
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER PRIMARY KEY REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model "
            "ON chunk_embeddings(model_name)"
        )
        page_columns = {row["name"] for row in connection.execute("PRAGMA table_info(document_pages)").fetchall()}
        if "layout_json" not in page_columns:
            connection.execute("ALTER TABLE document_pages ADD COLUMN layout_json TEXT NOT NULL DEFAULT '{}'")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_document_pages_document_id ON document_pages(document_id)")
        # Backfill documents created before page-level storage was introduced.
        connection.execute(
            """
            INSERT INTO document_pages (document_id, page_number, extracted_text)
            SELECT d.document_id, 1, d.extracted_text
            FROM documents d
            WHERE NOT EXISTS (
                SELECT 1 FROM document_pages p WHERE p.document_id = d.document_id
            )
            """
        )
        # Backfill chunks for documents uploaded by earlier application versions.
        existing_pages = connection.execute(
            "SELECT p.document_id, p.page_number, p.extracted_text "
            "FROM document_pages p WHERE NOT EXISTS ("
            "SELECT 1 FROM document_chunks c WHERE c.document_id = p.document_id "
            "AND c.page_number = p.page_number)"
        ).fetchall()
        for page in existing_pages:
            connection.executemany(
                "INSERT OR IGNORE INTO document_chunks "
                "(document_id, page_number, chunk_index, start_char, end_char, chunk_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (page["document_id"], page["page_number"], item["chunk_index"],
                     item["start_char"], item["end_char"], item["text"])
                    for item in chunk_text(page["extracted_text"], CHUNK_MAX_CHARS, CHUNK_OVERLAP)
                ],
            )
        connection.commit()


# Initialise the local SQLite database when the application module loads.
init_db()


def extract_page_text(page) -> tuple[str, object | None, str]:
    """Extract native text and fall back to OCR when a page has no usable text layer."""
    native_text = page.get_text("text").strip()
    has_images = bool(page.get_images(full=True))
    has_drawings = bool(page.get_drawings())
    needs_ocr = len(native_text) < OCR_MIN_NATIVE_CHARS and (has_images or has_drawings)
    if not needs_ocr:
        return native_text, None, "native"

    try:
        text_page = page.get_textpage_ocr(
            language=OCR_LANGUAGE,
            dpi=OCR_DPI,
            full=True,
        )
        recognised_text = page.get_text("text", textpage=text_page).strip()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "OCR is required for this PDF page, but the OCR engine or language data is unavailable. "
                "Install Tesseract and its language pack, then check TESSDATA_PREFIX."
            ),
        ) from exc

    if not recognised_text:
        raise HTTPException(
            status_code=422,
            detail="OCR did not recognise readable text on one or more pages.",
        )
    return recognised_text, text_page, "ocr"


def extract_pdf(data: bytes) -> tuple[list[str], int, list[dict]]:
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise HTTPException(status_code=422, detail="Password-protected PDFs are not supported.")
            if len(pdf) == 0:
                raise HTTPException(status_code=422, detail="The PDF contains no pages.")
            if len(pdf) > MAX_PDF_PAGES:
                raise HTTPException(status_code=413, detail=f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
            page_texts: list[str] = []
            page_layouts: list[dict] = []
            for page in pdf:
                page_text, text_page, extraction_method = extract_page_text(page)
                text_kwargs = {"textpage": text_page} if text_page is not None else {}
                blocks = []
                for block_index, block in enumerate(page.get_text("dict", **text_kwargs).get("blocks", [])):
                    if block.get("type") != 0:
                        continue
                    source_lines = block.get("lines", [])
                    line_texts = []
                    font_names = set()
                    line_items = []
                    for line in source_lines:
                        spans = line.get("spans", [])
                        line_text = "".join(span.get("text", "") for span in spans).strip()
                        if not line_text:
                            continue
                        line_texts.append(line_text)
                        span_items = []
                        for span in spans:
                            span_text = span.get("text", "")
                            if not span_text:
                                continue
                            font_name = span.get("font", "")
                            if font_name:
                                font_names.add(font_name)
                            span_items.append({
                                "bbox": [round(float(value), 2) for value in span.get("bbox", [])],
                                "text": span_text,
                                "font_name": font_name,
                                "font_size": round(float(span.get("size", 0)), 2),
                            })
                        line_items.append({
                            "bbox": [round(float(value), 2) for value in line.get("bbox", [])],
                            "text": line_text,
                            "spans": span_items,
                        })
                    block_text = "\n".join(line_texts).strip()
                    if block_text:
                        blocks.append({
                            "block_index": block_index,
                            "bbox": [round(float(value), 2) for value in block.get("bbox", [])],
                            "text": block_text,
                            "line_count": len(line_texts),
                            "font_names": sorted(font_names),
                            "lines": line_items,
                        })
                page_words = [
                    {
                        "x0": round(float(word[0]), 2),
                        "y0": round(float(word[1]), 2),
                        "x1": round(float(word[2]), 2),
                        "y1": round(float(word[3]), 2),
                        "text": word[4],
                        "block_no": int(word[5]),
                        "line_no": int(word[6]),
                        "word_no": int(word[7]),
                    }
                    for word in page.get_text("words", **text_kwargs)
                ]
                reading_order = reconstruct_reading_order(
                    page_words,
                    page_width=float(page.rect.width),
                    fallback_text=page_text,
                )
                if reading_order["layout_type"] == "two_column":
                    page_text = reading_order["text"]
                page_texts.append(page_text)
                page_layouts.append({
                    "page_width": round(float(page.rect.width), 2),
                    "page_height": round(float(page.rect.height), 2),
                    "blocks": blocks,
                    "words": page_words,
                    "tables": extract_native_tables(page) if extraction_method == "native" else [],
                    "extraction_method": extraction_method,
                    "ocr_applied": extraction_method == "ocr",
                    "layout_type": reading_order["layout_type"],
                    "column_count": reading_order["column_count"],
                })
            if not any(page_texts):
                raise HTTPException(status_code=422, detail="No readable text was found in the PDF.")
            return page_texts, len(pdf), page_layouts
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
    return {"status": "ok", "service": "documind-api", "version": "0.10.0"}


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
</style></head><body><main><section class="panel"><p class="muted">DOCUMENT INTELLIGENCE · MVP 0.7</p><h1>DocuMind</h1>
<p>Upload a PDF to extract its contents, including scanned pages using OCR, generate an extractive summary, and search within your saved documents. Use the API documentation to index documents, search semantically, and ask questions with source citations.</p>
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
async def upload_document(file: Annotated[UploadFile, File(description="A PDF document; scanned pages are processed with OCR")]) -> DocumentResponse:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf") or file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="The file does not have a valid PDF signature.")
    page_texts, page_count, page_layouts = extract_pdf(data)
    extracted_text = "\n\n".join(page_texts).strip()
    document_id = str(uuid.uuid4())
    with closing(connect_db()) as connection:
        connection.execute(
            "INSERT INTO documents (document_id, filename, extracted_text, page_count, character_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (document_id, filename, extracted_text, page_count, len(extracted_text)),
        )
        connection.executemany(
            "INSERT INTO document_pages (document_id, page_number, extracted_text, layout_json) VALUES (?, ?, ?, ?)",
            [
                (document_id, index, page_text, json.dumps(page_layouts[index - 1], ensure_ascii=False))
                for index, page_text in enumerate(page_texts, start=1)
            ],
        )
        for page_number, page_text in enumerate(page_texts, start=1):
            connection.executemany(
                "INSERT INTO document_chunks "
                "(document_id, page_number, chunk_index, start_char, end_char, chunk_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (document_id, page_number, item["chunk_index"], item["start_char"],
                     item["end_char"], item["text"])
                    for item in chunk_text(page_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP)
                ],
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
            "SELECT document_id, page_number, extracted_text, layout_json FROM document_pages "
            "WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()
    return [
        DocumentPageResponse(
            document_id=row["document_id"],
            page_number=row["page_number"],
            character_count=len(row["extracted_text"]),
            text=row["extracted_text"],
            extraction_method=json.loads(row["layout_json"] or "{}").get("extraction_method", "native"),
            ocr_applied=json.loads(row["layout_json"] or "{}").get("ocr_applied", False),
            layout_type=json.loads(row["layout_json"] or "{}").get("layout_type", "single_column"),
            column_count=json.loads(row["layout_json"] or "{}").get("column_count", 1),
        )
        for row in rows
    ]


@app.get("/documents/{document_id}/layout", response_model=list[DocumentPageLayoutResponse])
def get_document_layout(document_id: str) -> list[DocumentPageLayoutResponse]:
    get_document(document_id)
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT document_id, page_number, layout_json FROM document_pages "
            "WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()
    pages = []
    for row in rows:
        layout = json.loads(row["layout_json"] or "{}")
        pages.append(DocumentPageLayoutResponse(
            document_id=row["document_id"],
            page_number=row["page_number"],
            page_width=layout.get("page_width", 0),
            page_height=layout.get("page_height", 0),
            blocks=[LayoutBlock(**block) for block in layout.get("blocks", [])],
            extraction_method=layout.get("extraction_method", "native"),
            ocr_applied=layout.get("ocr_applied", False),
            layout_type=layout.get("layout_type", "single_column"),
            column_count=layout.get("column_count", 1),
        ))
    return pages


@app.get("/documents/{document_id}/tables", response_model=list[DocumentPageTablesResponse])
def get_document_tables(document_id: str) -> list[DocumentPageTablesResponse]:
    get_document(document_id)
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT document_id, page_number, layout_json FROM document_pages "
            "WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()

    page_data = []
    for row in rows:
        layout = json.loads(row["layout_json"] or "{}")
        candidates = layout.get("tables") or detect_table_candidates(layout.get("words", []))
        normalised = []
        for candidate in candidates:
            candidate = dict(candidate)
            candidate.setdefault("detection_method", "heuristic")
            candidate.setdefault("bbox", None)
            candidate.setdefault("header_names", [])
            candidate.setdefault("header_external", False)
            candidate.setdefault("continues_from_previous_page", False)
            candidate.setdefault("continues_on_next_page", False)
            candidate.setdefault("continuation_group", None)
            for row_index, row in enumerate(candidate.get("rows", [])):
                row.setdefault("row_index", row_index)
                for column_index, cell in enumerate(row.get("cells", [])):
                    cell.setdefault("bbox", None)
                    cell.setdefault("column_index", column_index)
                    cell.setdefault("column_span", 1)
            normalised.append(candidate)
        page_data.append({
            "document_id": row["document_id"],
            "page_number": row["page_number"],
            "page_width": float(layout.get("page_width", 0)),
            "page_height": float(layout.get("page_height", 0)),
            "tables": normalised,
        })

    # Link likely continuations when a table reaches the bottom of one page
    # and a structurally similar table starts near the top of the next page.
    for previous, current in zip(page_data, page_data[1:]):
        if not previous["tables"] or not current["tables"]:
            continue
        previous_index = len(previous["tables"]) - 1
        previous_table = previous["tables"][previous_index]
        current_table = current["tables"][0]
        previous_bbox = previous_table.get("bbox")
        current_bbox = current_table.get("bbox")
        if not previous_bbox or not current_bbox:
            continue
        previous_height = previous["page_height"]
        current_height = current["page_height"]
        if not previous_height or not current_height:
            continue
        previous_reaches_bottom = previous_bbox[3] / previous_height >= 0.72
        current_starts_top = current_bbox[1] / current_height <= 0.28
        same_columns = previous_table["column_count"] == current_table["column_count"]
        previous_width = (previous_bbox[2] - previous_bbox[0]) / max(previous["page_width"], 1)
        current_width = (current_bbox[2] - current_bbox[0]) / max(current["page_width"], 1)
        similar_width = abs(previous_width - current_width) <= 0.15
        if previous_reaches_bottom and current_starts_top and same_columns and similar_width:
            group_id = f"p{previous['page_number']}-t{previous_index + 1}"
            previous_table["continues_on_next_page"] = True
            previous_table["continuation_group"] = group_id
            current_table["continues_from_previous_page"] = True
            current_table["continuation_group"] = group_id

    return [
        DocumentPageTablesResponse(
            document_id=page["document_id"],
            page_number=page["page_number"],
            tables=[TableCandidate(**candidate) for candidate in page["tables"]],
        )
        for page in page_data
    ]


@app.get("/documents/{document_id}/chunks", response_model=list[DocumentChunkResponse])
def get_document_chunks(document_id: str) -> list[DocumentChunkResponse]:
    get_document(document_id)
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT c.document_id, d.filename, c.page_number, c.chunk_index, "
            "c.start_char, c.end_char, c.chunk_text "
            "FROM document_chunks c JOIN documents d ON d.document_id = c.document_id "
            "WHERE c.document_id = ? ORDER BY c.page_number, c.chunk_index",
            (document_id,),
        ).fetchall()
    return [
        DocumentChunkResponse(
            document_id=row["document_id"],
            filename=row["filename"],
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            character_count=len(row["chunk_text"]),
            text=row["chunk_text"],
        )
        for row in rows
    ]


@app.post("/documents/{document_id}/index", response_model=EmbeddingIndexResponse)
def index_document(document_id: str) -> EmbeddingIndexResponse:
    get_document(document_id)
    with closing(connect_db()) as connection:
        rows = connection.execute(
            "SELECT chunk_id, chunk_text FROM document_chunks "
            "WHERE document_id = ? ORDER BY page_number, chunk_index",
            (document_id,),
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=422, detail="This document has no text chunks to index.")

    try:
        vectors = embed_passages([row["chunk_text"] for row in rows])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if len(vectors) != len(rows) or not vectors:
        raise HTTPException(status_code=503, detail="The embedding provider returned an invalid number of vectors.")

    dimensions = len(vectors[0])
    if dimensions == 0 or any(
        len(vector) != dimensions or not all(math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise HTTPException(status_code=503, detail="The embedding provider returned invalid vector data.")

    with closing(connect_db()) as connection:
        connection.executemany(
            "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
            [(row["chunk_id"],) for row in rows],
        )
        connection.executemany(
            "INSERT INTO chunk_embeddings (chunk_id, model_name, dimensions, vector_json) "
            "VALUES (?, ?, ?, ?)",
            [
                (row["chunk_id"], MODEL_NAME, dimensions, json.dumps(vector))
                for row, vector in zip(rows, vectors)
            ],
        )
        connection.commit()

    return EmbeddingIndexResponse(
        document_id=document_id,
        indexed_chunks=len(rows),
        embedding_model=MODEL_NAME,
        dimensions=dimensions,
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


@app.post("/documents/semantic-search", response_model=list[SemanticSearchResult])
def semantic_search(request: SemanticSearchRequest) -> list[SemanticSearchResult]:
    if request.document_id:
        get_document(request.document_id)
    try:
        query_vector = embed_query(request.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not query_vector or not all(math.isfinite(value) for value in query_vector):
        raise HTTPException(status_code=503, detail="The embedding provider returned an invalid query vector.")

    sql = (
        "SELECT c.document_id, d.filename, c.page_number, c.chunk_index, "
        "c.start_char, c.end_char, c.chunk_text, e.model_name, e.dimensions, e.vector_json "
        "FROM chunk_embeddings e JOIN document_chunks c ON c.chunk_id = e.chunk_id "
        "JOIN documents d ON d.document_id = c.document_id "
        "WHERE e.model_name = ?"
    )
    params: list = [MODEL_NAME]
    if request.document_id:
        sql += " AND c.document_id = ?"
        params.append(request.document_id)

    with closing(connect_db()) as connection:
        rows = connection.execute(sql, params).fetchall()
    if not rows:
        raise HTTPException(
            status_code=409,
            detail="No compatible embeddings are indexed. Call POST /documents/{document_id}/index first.",
        )

    results = []
    for row in rows:
        if row["dimensions"] != len(query_vector):
            continue
        vector = json.loads(row["vector_json"])
        score = cosine_similarity(query_vector, vector)
        results.append(SemanticSearchResult(
            document_id=row["document_id"],
            filename=row["filename"],
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            score=round(score, 6),
            text=row["chunk_text"],
            embedding_model=row["model_name"],
        ))

    return sorted(results, key=lambda result: result.score, reverse=True)[:request.limit]


@app.post("/documents/ask", response_model=AskResponse)
def ask_documents(request: AskRequest) -> AskResponse:
    retrieved = semantic_search(SemanticSearchRequest(
        query=request.question,
        limit=request.limit,
        document_id=request.document_id,
    ))
    if not retrieved:
        raise HTTPException(
            status_code=409,
            detail="No usable indexed evidence was found for this question.",
        )

    sources = [
        {
            "source_id": f"S{index}",
            "document_id": item.document_id,
            "filename": item.filename,
            "page_number": item.page_number,
            "chunk_index": item.chunk_index,
            "start_char": item.start_char,
            "end_char": item.end_char,
            "score": item.score,
            "text": item.text,
        }
        for index, item in enumerate(retrieved, start=1)
    ]
    try:
        answer = generate_grounded_answer(request.question, sources)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=[RAGSource(**source) for source in sources],
    )


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
