"""Build the final submission report PDF (selectable text, <= 8 pages) from saved results."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

REPO = 'https://github.com/mantrakhandelwaljee-lgtm/procoder-zeckathon-credit-ews'
TEAL = colors.HexColor('#087f8c'); RED = colors.HexColor('#c0392b'); GREY = colors.HexColor('#5f6b76')
LIGHT = colors.HexColor('#eef5f6')


def fonts():
    base = '/usr/share/fonts/truetype/dejavu/'
    try:
        pdfmetrics.registerFont(TTFont('DV', base + 'DejaVuSans.ttf'))
        pdfmetrics.registerFont(TTFont('DVB', base + 'DejaVuSans-Bold.ttf'))
        pdfmetrics.registerFont(TTFont('DVI', base + 'DejaVuSans-Oblique.ttf'))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily('DV', normal='DV', bold='DVB', italic='DVI', boldItalic='DVB')
        return 'DV', 'DVB'
    except Exception:
        return 'Helvetica', 'Helvetica-Bold'


def styles():
    f, fb = fonts()
    ss = getSampleStyleSheet()
    S = {
        'title': ParagraphStyle('t', fontName=fb, fontSize=17, leading=21, textColor=colors.black, spaceAfter=2),
        'sub': ParagraphStyle('s', fontName=f, fontSize=9.2, leading=12, textColor=GREY),
        'h1': ParagraphStyle('h1', fontName=fb, fontSize=12.2, leading=15, textColor=TEAL, spaceBefore=7, spaceAfter=3),
        'h2': ParagraphStyle('h2', fontName=fb, fontSize=9.6, leading=12, spaceBefore=4, spaceAfter=1.5),
        'body': ParagraphStyle('b', fontName=f, fontSize=8.6, leading=11.4, alignment=TA_LEFT, spaceAfter=3),
        'small': ParagraphStyle('sm', fontName=f, fontSize=7.4, leading=9.4, textColor=GREY, spaceAfter=2),
        'cell': ParagraphStyle('c', fontName=f, fontSize=7.6, leading=9.4),
        'cellb': ParagraphStyle('cb', fontName=fb, fontSize=7.6, leading=9.4),
        'bul': ParagraphStyle('bu', fontName=f, fontSize=8.6, leading=11.2, leftIndent=9, bulletIndent=0, spaceAfter=1.5),
        'kpi': ParagraphStyle('k', fontName=fb, fontSize=15, leading=17, textColor=TEAL),
        'kpil': ParagraphStyle('kl', fontName=f, fontSize=7.2, leading=9, textColor=GREY),
    }
    return S


def table(rows, widths, S, header=True, zebra=True, bold_col0=False):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), S['cellb'] if (header and i == 0) or (bold_col0 and j == 0) else S['cell'])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), .6, TEAL),
          ('TOPPADDING', (0, 0), (-1, -1), 2.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
          ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), LIGHT))
    if zebra:
        for i in range(2 if header else 1, len(rows), 2):
            st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f7f9fa')))
    t.setStyle(TableStyle(st))
    return t


def pct(x, d=0):
    return '–' if x is None else f'{100 * x:.{d}f}%'


def num(x, d=2):
    return '–' if x is None or x != x else f'{x:.{d}f}'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(GREY)
    canvas.drawString(15 * mm, 9 * mm, 'Team procoder · Zeckathon PS2 Risk & Fraud · Credit Early-Warning System')
    canvas.drawRightString(195 * mm, 9 * mm, f'Page {doc.page}')
    canvas.restoreState()


def bullets(items, S):
    return [Paragraph(x, S['bul'], bulletText='•') for x in items]
