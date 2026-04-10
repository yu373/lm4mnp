import fitz  # PyMuPDF

doc = fitz.open("Fig3.pdf")

new_doc = fitz.open()

margin = 28.35  # 1 cm

page = doc[0]  # ⭐ 只处理第一页

blocks = page.get_text("blocks")

if blocks:
    x0 = min(b[0] for b in blocks)
    y0 = min(b[1] for b in blocks)
    x1 = max(b[2] for b in blocks)
    y1 = max(b[3] for b in blocks)

    rect = fitz.Rect(x0, y0, x1, y1)

    rect = fitz.Rect(
        rect.x0 - margin,
        rect.y0 - margin,
        rect.x1 + margin,
        rect.y1 + margin
    )

    rect = rect & page.rect
    page.set_cropbox(rect)

new_doc.insert_pdf(doc, from_page=0, to_page=0)

new_doc.save("output.pdf")