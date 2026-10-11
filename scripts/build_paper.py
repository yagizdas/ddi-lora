#!/usr/bin/env python3
"""Render docs/project_report.md as a four-page, two-column IEEE-style paper.

Install the optional paper dependency with: python -m pip install -e '.[paper]'
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 243


def register_fonts() -> None:
    # Embedded Times preserves the supplied PDF's typography and Turkish author name.
    font_roots = [Path("/System/Library/Fonts/Supplemental"), ROOT / "local_archive/paper_fonts"]
    names = {
        "Paper": "Times New Roman.ttf",
        "Paper-Bold": "Times New Roman Bold.ttf",
        "Paper-Italic": "Times New Roman Italic.ttf",
        "Paper-BoldItalic": "Times New Roman Bold Italic.ttf",
    }
    for family, filename in names.items():
        path = next((p / filename for p in font_roots if (p / filename).exists()), None)
        if path is None:
            raise RuntimeError(
                "Times New Roman fonts are required; supply them in local_archive/paper_fonts/"
            )
        pdfmetrics.registerFont(TTFont(family, str(path)))
    pdfmetrics.registerFontFamily(
        "Paper",
        normal="Paper",
        bold="Paper-Bold",
        italic="Paper-Italic",
        boldItalic="Paper-BoldItalic",
    )


def markup(text: str) -> str:
    text = html.escape(text)
    text = text.replace("10⁻⁴", "10<super>-4</super>")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(
        r"https://[^\s<]+",
        lambda m: f'<link href="{m.group(0)}" color="#000000">{m.group(0)}</link>',
        text,
    )
    return text


class Workflow(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = WIDTH, 184

    def draw(self):
        c = self.canv
        c.setFont("Paper-Bold", 8)
        c.drawString(8, 173, "TRAINING")
        labels = [
            ("DDInter seed records", "Interaction facts and provenance"),
            ("GPT-5-mini synthetic narratives", "Patient profiles and four-part targets"),
            ("Qwen3-14B + LoRA", "Rank 32; 0.86% trainable"),
        ]
        for i, (title, subtitle) in enumerate(labels):
            y = 138 - i * 44
            c.setStrokeColor(colors.HexColor("#555555"))
            c.setFillColor(colors.HexColor("#f5f5f5"))
            c.rect(8, y, 227, 30, fill=1)
            c.setFillColor(colors.black)
            c.setFont("Paper-Bold", 8.3)
            c.drawCentredString(121.5, y + 18, title)
            c.setFont("Paper", 7.5)
            c.drawCentredString(121.5, y + 7, subtitle)
            if i < 2:
                c.line(121.5, y, 121.5, y - 10)
                c.line(118.5, y - 7, 121.5, y - 10)
                c.line(124.5, y - 7, 121.5, y - 10)
        c.setFont("Paper-Bold", 8)
        c.drawString(8, 35, "LOCAL INFERENCE PATH")
        labels = ["Patient + drugs", "Adapted model", "Assessment"]
        for i, label in enumerate(labels):
            x = 8 + i * 80
            c.setFillColor(colors.HexColor("#f5f5f5"))
            c.rect(x, 7, 67, 20, fill=1)
            c.setFillColor(colors.black)
            c.setFont("Paper", 7.8)
            c.drawCentredString(x + 33.5, 14, label)
            if i < 2:
                c.line(x + 67, 17, x + 78, 17)
                c.line(x + 75, 20, x + 78, 17)
                c.line(x + 75, 14, x + 78, 17)


class Outcomes(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = WIDTH, 96

    def draw(self):
        c = self.canv
        left, scale = 60, 10
        for label, passed, failed, y in [("Baseline", 9, 6, 58), ("+ LoRA", 13, 2, 32)]:
            c.setFont("Paper", 8.5)
            c.drawRightString(53, y + 4, label)
            c.setFillColor(colors.HexColor("#5b8c85"))
            c.rect(left, y, passed * scale, 16, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#b9bec6"))
            c.rect(left + passed * scale, y, failed * scale, 16, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.drawCentredString(left + passed * scale / 2, y + 4, str(passed))
            c.setFillColor(colors.black)
            c.drawCentredString(left + passed * scale + failed * scale / 2, y + 4, str(failed))
        c.setFont("Paper", 8)
        for value in [0, 5, 10, 15]:
            c.drawCentredString(left + value * scale, 19, str(value))
        c.drawCentredString(135, 7, "Number of scenarios")
        c.setFillColor(colors.HexColor("#5b8c85"))
        c.rect(60, 83, 7, 7, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(70, 83, "Pass")
        c.setFillColor(colors.HexColor("#b9bec6"))
        c.rect(117, 83, 7, 7, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(127, 83, "Fail")


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Paper", 8)
    canvas.drawCentredString(306, 25, str(doc.page))
    if doc.page == 1:
        canvas.setFont("Paper", 22)
        canvas.drawCentredString(306, 718, "Patient-Aware Drug Interaction Assessment with a")
        canvas.drawCentredString(306, 693, "LoRA-Fine-Tuned Language Model")
        canvas.setFont("Paper", 10)
        for line, y in [
            ("Kemal Yağız Daşkıran", 670),
            ("Department of Electronics and Communications Engineering", 658),
            ("Istanbul Technical University, Istanbul, Turkey", 646),
            ("Email: daskiran21@itu.edu.tr", 634),
        ]:
            canvas.drawCentredString(306, y, line)
    canvas.restoreState()


def main() -> None:
    register_fonts()
    body = ParagraphStyle(
        "body",
        fontName="Paper",
        fontSize=10,
        leading=11.4,
        alignment=TA_JUSTIFY,
        firstLineIndent=10,
        spaceAfter=4,
    )
    abstract = ParagraphStyle(
        "abstract",
        parent=body,
        fontName="Paper-Bold",
        fontSize=9,
        leading=10.3,
        firstLineIndent=0,
        spaceAfter=8,
    )
    section = ParagraphStyle(
        "section",
        parent=body,
        fontSize=10,
        firstLineIndent=0,
        alignment=TA_CENTER,
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
    )
    subsection = ParagraphStyle(
        "subsection",
        parent=body,
        fontName="Paper-Italic",
        alignment=TA_LEFT,
        firstLineIndent=0,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )
    caption = ParagraphStyle(
        "caption",
        parent=body,
        fontSize=8,
        leading=9.2,
        alignment=TA_LEFT,
        firstLineIndent=0,
        spaceBefore=3,
        spaceAfter=7,
    )
    table_title = ParagraphStyle(
        "tabletitle",
        parent=caption,
        alignment=TA_CENTER,
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True,
    )
    cell = ParagraphStyle(
        "cell",
        parent=body,
        fontSize=8.2,
        leading=9.3,
        firstLineIndent=0,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    reference = ParagraphStyle(
        "reference",
        parent=body,
        fontSize=8,
        leading=9.1,
        firstLineIndent=-10,
        leftIndent=10,
        alignment=TA_LEFT,
        spaceAfter=5,
    )
    source = (ROOT / "docs/project_report.md").read_text(encoding="utf-8")
    source = source[source.index("**Abstract") :]
    blocks = re.split(r"\n\s*\n", source.strip())
    story = []
    for block in blocks:
        block = block.strip()
        if block.startswith("## "):
            story.append(Paragraph(markup(block[3:]), section))
        elif block.startswith("### "):
            story.append(Paragraph(markup(block[4:]), subsection))
        elif block == "[FIGURE workflow]":
            story.append(Workflow())
        elif block == "[FIGURE outcomes]":
            story.append(Outcomes())
        elif block.startswith("Fig. "):
            story.append(Paragraph(markup(block), caption))
        elif block.startswith("TABLE "):
            story.append(Paragraph(markup(block), table_title))
        elif block.startswith("|"):
            rows = [
                [c.strip() for c in line.strip().strip("|").split("|")]
                for line in block.splitlines()
                if not re.match(r"^\|[\s|:-]+\|$", line)
            ]
            if rows[0][0] == "Case":
                widths = [40, 101.5, 101.5]
            elif len(rows[0]) == 2:
                widths = [122, 121]
            else:
                widths = [107, 31, 31, 74]
            data = [
                [Paragraph(markup(f"**{c}**" if i == 0 else c), cell) for c in row]
                for i, row in enumerate(rows)
            ]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            row_padding = 1.5 if rows[0][0] == "Case" else 3
            table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), row_padding),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), row_padding),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.black),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
                    ]
                )
            )
            story.append(KeepTogether([table, Spacer(1, 6)]))
        elif block.startswith("**Abstract") or block.startswith("**Index Terms"):
            story.append(Paragraph(markup(block), abstract))
        elif re.match(r"^\[\d+\]", block):
            story.append(Paragraph(markup(block), reference))
        else:
            story.append(Paragraph(markup(block.replace("\n", " ")), body))
    first_frames = [
        Frame(
            x,
            54,
            WIDTH,
            552,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id=f"first_{i}",
        )
        for i, x in enumerate([54, 315])
    ]
    frames = [
        Frame(
            x,
            54,
            WIDTH,
            684,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id=f"body_{i}",
        )
        for i, x in enumerate([54, 315])
    ]
    output = ROOT / "docs/project_report.pdf"
    doc = BaseDocTemplate(
        str(output),
        pagesize=(612, 792),
        title="Patient-Aware Drug Interaction Assessment with a LoRA-Fine-Tuned Language Model",
        author="Kemal Yağız Daşkıran",
        creator="Author manuscript revision",
        subject="Patient-aware drug interaction research",
        pageCompression=1,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="First", frames=first_frames, onPage=draw_page, autoNextPageTemplate="Later"
            ),
            PageTemplate(id="Later", frames=frames, onPage=draw_page),
        ]
    )
    doc.build(story)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
