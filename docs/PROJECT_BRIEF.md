# Project brief — DocuMind

## Problem
People often need to locate key information inside lengthy PDF documents. Manual reading is slow and repetitive.

## MVP user story
As a user, I can upload a text-based PDF, receive a summary, and search for relevant passages without manually scanning every page.

## Acceptance criteria
- Reject non-PDF uploads.
- Reject files larger than 10 MB.
- Reject invalid, encrypted, empty, or over-100-page PDFs.
- Extract selectable PDF text.
- Return a summary for an uploaded document.
- Search across uploaded documents and rank matching paragraphs.
- Expose interactive API documentation.
- Pass automated tests.

## Non-goals for MVP 0.1
- OCR for scanned PDFs
- LLM-based generation
- Vector database / semantic retrieval
- User accounts and persistent document storage

## Risks
- In-memory storage loses documents on restart.
- Extractive summaries may miss context and are not equivalent to generative summaries.
- PDF parsing of untrusted files requires stronger isolation before production use.
