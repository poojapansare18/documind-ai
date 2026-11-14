from __future__ import annotations

from collections import defaultdict
from statistics import median


def _group_words_into_lines(words: list[dict], y_tolerance: float = 3.5) -> list[dict]:
    # Group by vertical position rather than PDF block IDs: table cells are
    # frequently emitted as separate text blocks by PDF generators.
    ordered = sorted(
        words,
        key=lambda item: (float(item.get("y0", 0)), float(item.get("x0", 0))),
    )
    lines: list[dict] = []
    for word in ordered:
        y = float(word.get("y0", 0))
        if lines and abs(y - median(lines[-1]["y_values"])) <= y_tolerance:
            lines[-1]["words"].append(word)
            lines[-1]["y_values"].append(y)
        else:
            lines.append({"y_values": [y], "words": [word]})
    for line in lines:
        line["words"].sort(key=lambda item: float(item.get("x0", 0)))
        line["y"] = median(line.pop("y_values"))
    return lines

def _split_line_into_cells(words: list[dict], minimum_gap: float = 18.0) -> list[dict]:
    if not words:
        return []
    heights = [
        max(1.0, float(word.get("y1", 0)) - float(word.get("y0", 0)))
        for word in words
    ]
    gap_threshold = max(minimum_gap, median(heights) * 1.4)
    groups: list[list[dict]] = [[words[0]]]
    for word in words[1:]:
        previous = groups[-1][-1]
        gap = float(word.get("x0", 0)) - float(previous.get("x1", 0))
        if gap >= gap_threshold:
            groups.append([word])
        else:
            groups[-1].append(word)

    cells = []
    for group in groups:
        cells.append({
            "text": " ".join(str(word.get("text", "")) for word in group).strip(),
            "bbox": [
                round(min(float(word.get("x0", 0)) for word in group), 2),
                round(min(float(word.get("y0", 0)) for word in group), 2),
                round(max(float(word.get("x1", 0)) for word in group), 2),
                round(max(float(word.get("y1", 0)) for word in group), 2),
            ],
        })
    return [cell for cell in cells if cell["text"]]


def detect_table_candidates(
    words: list[dict],
    minimum_columns: int = 3,
    minimum_rows: int = 2,
    maximum_row_gap: float = 36.0,
    minimum_cell_gap: float = 18.0,
) -> list[dict]:
    """Detect repeated, horizontally separated text rows as table candidates.

    This is a geometry heuristic, not a universal PDF table parser. It returns
    only candidates with repeated rows and reasonably aligned column starts.
    """
    candidate_rows = []
    for line in _group_words_into_lines(words):
        cells = _split_line_into_cells(line["words"], minimum_gap=minimum_cell_gap)
        if len(cells) < minimum_columns:
            continue
        candidate_rows.append({
            "y": line["y"],
            "cells": cells,
        })

    if len(candidate_rows) < minimum_rows:
        return []

    groups: list[list[dict]] = []
    current = [candidate_rows[0]]
    for row in candidate_rows[1:]:
        previous = current[-1]
        vertical_gap = row["y"] - previous["y"]
        shared_columns = min(len(row["cells"]), len(previous["cells"]))
        alignment = 0.0
        if shared_columns:
            differences = [
                abs(row["cells"][index]["bbox"][0] - previous["cells"][index]["bbox"][0])
                for index in range(shared_columns)
            ]
            alignment = sum(differences) / len(differences)
        if vertical_gap <= maximum_row_gap and alignment <= 24.0:
            current.append(row)
        else:
            if len(current) >= minimum_rows:
                groups.append(current)
            current = [row]
    if len(current) >= minimum_rows:
        groups.append(current)

    tables = []
    for group in groups:
        column_count = max(len(row["cells"]) for row in group)
        row_alignments = []
        for previous, current_row in zip(group, group[1:]):
            shared = min(len(previous["cells"]), len(current_row["cells"]))
            if shared:
                deltas = [
                    abs(previous["cells"][index]["bbox"][0] - current_row["cells"][index]["bbox"][0])
                    for index in range(shared)
                ]
                row_alignments.append(sum(deltas) / len(deltas))
        mean_alignment = sum(row_alignments) / len(row_alignments) if row_alignments else 24.0
        alignment_score = max(0.0, 1.0 - mean_alignment / 24.0)
        confidence = min(0.95, 0.55 + min(len(group) - minimum_rows, 4) * 0.05 + alignment_score * 0.15)
        tables.append({
            "row_count": len(group),
            "column_count": column_count,
            "confidence": round(confidence, 2),
            "rows": [
                {"cells": row["cells"]}
                for row in group
            ],
        })
    return tables

def reconstruct_reading_order(
    words: list[dict],
    page_width: float,
    fallback_text: str = "",
    gutter_tolerance: float = 12.0,
    minimum_column_lines: int = 2,
) -> dict:
    """Reorder clear two-column pages as left column then right column.

    Word lines are split at unusually wide horizontal gaps so that left and
    right column text sharing the same baseline are not mistaken for one line.
    Full-width lines divide the page into vertical reading bands.
    """
    lines = _group_words_into_lines(words)
    records = []
    column_gap_threshold = max(36.0, float(page_width) * 0.06)
    for line in lines:
        line_words = line["words"]
        if not line_words:
            continue

        segments = [[line_words[0]]]
        for word in line_words[1:]:
            previous = segments[-1][-1]
            gap = float(word.get("x0", 0)) - float(previous.get("x1", 0))
            if gap >= column_gap_threshold:
                segments.append([word])
            else:
                segments[-1].append(word)

        # Three or more separated groups on one baseline are more likely to
        # be table cells than prose columns. Keep that row together.
        if len(segments) > 2:
            segments = [line_words]

        for segment in segments:
            x0 = min(float(word.get("x0", 0)) for word in segment)
            x1 = max(float(word.get("x1", 0)) for word in segment)
            records.append({
                "y": float(line["y"]),
                "x0": x0,
                "x1": x1,
                "text": " ".join(str(word.get("text", "")) for word in segment).strip(),
            })

    midpoint = float(page_width) / 2.0
    left_lines = []
    right_lines = []
    spanning_lines = []
    for line in records:
        if line["x1"] <= midpoint - gutter_tolerance:
            left_lines.append(line)
        elif line["x0"] >= midpoint + gutter_tolerance:
            right_lines.append(line)
        elif line["x0"] < midpoint - gutter_tolerance and line["x1"] > midpoint + gutter_tolerance:
            spanning_lines.append(line)
        elif (line["x0"] + line["x1"]) / 2.0 < midpoint:
            left_lines.append(line)
        else:
            right_lines.append(line)

    if len(left_lines) < minimum_column_lines or len(right_lines) < minimum_column_lines:
        return {
            "text": fallback_text,
            "layout_type": "single_column",
            "column_count": 1,
        }

    ordered_text = []
    remaining_left = sorted(left_lines, key=lambda item: item["y"])
    remaining_right = sorted(right_lines, key=lambda item: item["y"])
    for spanning in sorted(spanning_lines, key=lambda item: item["y"]):
        band_left = [line for line in remaining_left if line["y"] < spanning["y"]]
        band_right = [line for line in remaining_right if line["y"] < spanning["y"]]
        ordered_text.extend(line["text"] for line in band_left)
        ordered_text.extend(line["text"] for line in band_right)
        remaining_left = [line for line in remaining_left if line["y"] >= spanning["y"]]
        remaining_right = [line for line in remaining_right if line["y"] >= spanning["y"]]
        ordered_text.append(spanning["text"])

    ordered_text.extend(line["text"] for line in remaining_left)
    ordered_text.extend(line["text"] for line in remaining_right)
    return {
        "text": "\\n".join(text for text in ordered_text if text),
        "layout_type": "two_column",
        "column_count": 2,
    }
