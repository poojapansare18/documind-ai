from __future__ import annotations



def _choose_boundary(text: str, start: int, limit: int) -> int:
    """Choose a readable boundary near the size limit without losing characters."""
    if limit >= len(text):
        return len(text)
    floor = start + int((limit - start) * 0.6)
    candidates = [
        text.rfind("\n\n", floor, limit),
        text.rfind("\n", floor, limit),
        text.rfind(". ", floor, limit),
        text.rfind("? ", floor, limit),
        text.rfind("! ", floor, limit),
        text.rfind(" ", floor, limit),
    ]
    boundary = max(candidates)
    if boundary <= start:
        return limit
    # Include the punctuation or paragraph delimiter in the preceding chunk.
    if text[boundary:boundary + 2] in {". ", "? ", "! ", "\n\n"}:
        return boundary + 1
    return boundary


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[dict]:
    """Split page text into overlapping chunks while preserving exact source offsets.

    Offsets are zero-based, end-exclusive character positions within the page.
    """
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars.")

    if not text or not text.strip():
        return []

    chunks: list[dict] = []
    start = 0
    chunk_index = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break

        limit = min(start + max_chars, len(text))
        end = _choose_boundary(text, start, limit)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            end = limit

        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "text": chunk,
            })
            chunk_index += 1

        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Avoid starting the next chunk in the middle of a word.
        while next_start < end and not text[next_start - 1].isspace() and not text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks
