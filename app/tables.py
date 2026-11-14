from __future__ import annotations

from statistics import median


def _bbox_values(value) -> list[float] | None:
    if value is None:
        return None
    try:
        values = [round(float(part), 2) for part in value]
    except (TypeError, ValueError):
        return None
    return values if len(values) == 4 else None


def _cluster_edges(values: list[float], tolerance: float = 2.0) -> list[float]:
    ordered = sorted(values)
    clusters: list[list[float]] = []
    for value in ordered:
        if clusters and value - median(clusters[-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [round(median(cluster), 2) for cluster in clusters]


def _nearest_edge_index(value: float, edges: list[float]) -> int:
    return min(range(len(edges)), key=lambda index: abs(edges[index] - value))


def _overlaps_existing(bbox: list[float], existing: list[tuple[float, float, float, float]]) -> bool:
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    if area == 0:
        return False
    for previous in existing:
        intersection_width = max(0.0, min(bbox[2], previous[2]) - max(bbox[0], previous[0]))
        intersection_height = max(0.0, min(bbox[3], previous[3]) - max(bbox[1], previous[1]))
        intersection = intersection_width * intersection_height
        previous_area = max(0.0, previous[2] - previous[0]) * max(0.0, previous[3] - previous[1])
        if intersection / max(min(area, previous_area), 1.0) >= 0.85:
            return True
    return False


def _table_to_payload(table, strategy: str) -> dict | None:
    extracted_rows = table.extract()
    if not extracted_rows:
        return None

    row_objects = getattr(table, "rows", [])
    raw_cells = getattr(table, "cells", []) or []
    column_count = int(getattr(table, "col_count", 0) or max((len(row) for row in extracted_rows), default=0))
    if column_count < 2:
        return None

    row_boxes: list[list[list[float] | None]] = []
    all_x_edges: list[float] = []\n    all_y_edges: list[float] = []
    for row_index, values in enumerate(extracted_rows):
        row_object = row_objects[row_index] if row_index < len(row_objects) else None
        cell_rectangles = getattr(row_object, "cells", None) if row_object is not None else None
        if cell_rectangles is None:
            start = row_index * column_count
            cell_rectangles = raw_cells[start:start + column_count]
        boxes = [_bbox_values(cell_rectangles[index]) if index < len(cell_rectangles) else None
                 for index in range(column_count)]
        row_boxes.append(boxes)
        for bbox in boxes:
            if bbox:
                all_x_edges.extend([bbox[0], bbox[2]])\n                all_y_edges.extend([bbox[1], bbox[3]])

    x_edges = _cluster_edges(all_x_edges)\n    y_edges = _cluster_edges(all_y_edges)
    rows = []
    for row_index, values in enumerate(extracted_rows):
        cells = []
        for column_index in range(column_count):
            value = values[column_index] if column_index < len(values) else None
            bbox = row_boxes[row_index][column_index]
            column_span = 1
            row_span = 1
            if bbox and len(x_edges) >= 2:
                start_edge = _nearest_edge_index(bbox[0], x_edges)
                end_edge = _nearest_edge_index(bbox[2], x_edges)
                column_span = max(1, end_edge - start_edge)
            if bbox and len(y_edges) >= 2:
                top_edge = _nearest_edge_index(bbox[1], y_edges)
                bottom_edge = _nearest_edge_index(bbox[3], y_edges)
                row_span = max(1, bottom_edge - top_edge)
            cells.append({
                "text": str(value).strip() if value is not None else "",
                "bbox": bbox,
                "column_index": column_index,
                "column_span": column_span,
                "row_span": row_span,
            })
        rows.append({"row_index": row_index, "cells": cells})

    header = getattr(table, "header", None)
    header_names = list(getattr(header, "names", []) or []) if header else []
    header_external = bool(getattr(header, "external", False)) if header else False
    bbox = _bbox_values(getattr(table, "bbox", None))
    return {
        "row_count": len(rows),
        "column_count": column_count,
        "confidence": None,
        "detection_method": f"pymupdf_{strategy}",
        "bbox": bbox,
        "header_names": [str(name) if name is not None else "" for name in header_names],
        "header_external": header_external,
        "rows": rows,
    }


def extract_native_tables(page) -> list[dict]:
    """Extract structured tables using PyMuPDF's geometry-aware table finder.

    Try ruled-line detection first, then text-alignment detection. The second
    strategy is useful for borderless tables but is more prone to false positives.
    """
    found: list[dict] = []
    seen: list[tuple[float, float, float, float]] = []
    for strategy in ("lines", "text"):
        try:
            finder = page.find_tables(strategy=strategy, min_words_vertical=3, min_words_horizontal=2)
        except Exception:
            continue
        for table in getattr(finder, "tables", []):
            try:
                payload = _table_to_payload(table, strategy)
            except Exception:
                continue
            if not payload or payload["row_count"] < 2:
                continue
            bbox = payload["bbox"]
            if bbox:
                if _overlaps_existing(bbox, seen):
                    continue
                seen.append(tuple(bbox))
            non_empty = sum(
                bool(cell["text"])
                for row in payload["rows"]
                for cell in row["cells"]
            )
            if non_empty < 4:
                continue
            found.append(payload)
    return found
