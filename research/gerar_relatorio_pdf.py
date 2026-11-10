"""Gera o PDF do relatório a partir de research/RELATORIO.md.

Converte o Markdown em HTML (com as figuras embutidas) e usa o Chromium do
Playwright para renderizar o PDF, o mesmo caminho usado no resto do projeto.

Uso:
    pip install markdown playwright && playwright install chromium
    python research/gerar_relatorio_pdf.py
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

AQUI = Path(__file__).resolve().parent
RELATORIO_MD = AQUI / "RELATORIO.md"
PDF_SAIDA = AQUI / "RELATORIO.pdf"

CSS = """
@page { size: A4; margin: 15mm 18mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
       font-size: 10pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 13pt; margin-top: 18px; color: #222; }
h3 { font-size: 11pt; margin-top: 12px; color: #333; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background: #f2f2f2; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
pre code { background: none; padding: 0; }
img { max-width: 100%; max-height: 68mm; width: auto; display: block; margin: 6px auto; }
table, tr, pre { break-inside: avoid; }
p:has(+ p img) { break-after: avoid; }
h2, h3 { break-after: avoid; }
blockquote { border-left: 3px solid #ccc; margin: 10px 0; padding-left: 12px; color: #555; }
"""


def embed_images(html: str, base_dir: Path) -> str:
    """Converte caminhos de imagem em data URIs para o PDF ficar autocontido."""

    def replace(match: re.Match[str]) -> str:
        src = match.group(1)
        image_path = (base_dir / src).resolve()
        if not image_path.exists():
            return match.group(0)
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    return re.sub(r'src="([^"]+)"', replace, html)


def main() -> None:
    text = RELATORIO_MD.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    body = embed_images(body, AQUI)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.pdf(path=str(PDF_SAIDA), prefer_css_page_size=True, print_background=True)
        browser.close()
    print(f"PDF gerado: {PDF_SAIDA}")


if __name__ == "__main__":
    main()
