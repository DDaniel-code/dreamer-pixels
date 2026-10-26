#!/usr/bin/env python3
"""Render the Markdown midterm report as a compact, reviewable PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "midterm_report.md"
OUTPUT = ROOT / "output" / "pdf" / "dreamer_pixels_midterm_report.pdf"


def inline_markup(text: str) -> str:
    """Convert the small Markdown subset used in the report to ReportLab markup."""
    text = html.escape(text, quote=False)
    text = re.sub(r"&lt;(https?://[^&]+)&gt;", r"<link href='\1' color='#1f5f99'>\1</link>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(inline_markup(text), style)


def render_table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    body = [[paragraph(cell.strip(), styles["table"]) for cell in row] for row in rows]
    widths = [0.55 * inch, 0.62 * inch, 0.45 * inch, 0.72 * inch, 0.68 * inch, 0.68 * inch, 0.55 * inch, 0.78 * inch, 0.58 * inch]
    table = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#102a43")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9fb3c8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_story(styles: dict[str, ParagraphStyle]) -> list[object]:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list[object] = []
    index = 0
    skipped_document_title = False
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue

        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|---"):
            table_rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                if re.match(r"^\|\s*:?-+", lines[index]):
                    index += 1
                    continue
                table_rows.append([cell for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            story.extend([Spacer(1, 4), render_table(table_rows, styles), Spacer(1, 8)])
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            if len(heading.group(1)) == 1 and not skipped_document_title:
                skipped_document_title = True
                index += 1
                continue
            level = len(heading.group(1))
            style_name = {1: "h1", 2: "h2", 3: "h3"}[level]
            story.append(Paragraph(inline_markup(heading.group(2)), styles[style_name]))
            index += 1
            continue

        bullet = re.match(r"^[-*]\s+(.*)$", line)
        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if bullet or numbered:
            marker = "•" if bullet else f"{numbered.group(1)}."
            content = bullet.group(1) if bullet else numbered.group(2)
            story.append(Paragraph(f"<b>{marker}</b> {inline_markup(content)}", styles["list"]))
            index += 1
            continue

        story.append(paragraph(line, styles["body"]))
        index += 1

    return story


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d9e2ec"))
    canvas.line(doc.leftMargin, 0.48 * inch, letter[0] - doc.rightMargin, 0.48 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#627d98"))
    canvas.drawString(doc.leftMargin, 0.30 * inch, "CMU 11-785 | Dreamer V1 from Pixels | Preparation-stage evidence")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.30 * inch, f"{doc.page}")
    canvas.restoreState()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#102a43"), spaceAfter=16),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#102a43"), spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#243b53"), spaceBefore=7, spaceAfter=3, keepWithNext=True),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=9.6, leading=12, textColor=colors.HexColor("#334e68"), spaceBefore=5, spaceAfter=2, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.0, alignment=TA_LEFT, textColor=colors.HexColor("#243b53"), spaceAfter=5),
        "list": ParagraphStyle("List", parent=base["BodyText"], fontName="Helvetica", fontSize=9.1, leading=11.5, leftIndent=13, firstLineIndent=-10, textColor=colors.HexColor("#243b53"), spaceAfter=3),
        "table": ParagraphStyle("Table", parent=base["BodyText"], fontName="Helvetica", fontSize=6.8, leading=8.2, textColor=colors.HexColor("#243b53")),
    }

    title = "Midterm Report — Reproducing Dreamer V1 from Pixels"
    story = [Paragraph(title, styles["title"]), Paragraph("CMU 11-785 | Preparation and bounded validation", styles["body"])]
    story.extend(build_story(styles))
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.64 * inch,
        title=title,
        author="Dreamer Pixels project",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
