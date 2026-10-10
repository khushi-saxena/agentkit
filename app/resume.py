"""
Load the subject's resume from whatever format it happens to be in.

Public repos systematically underrate anyone whose serious work lives in
private company repos, so the resume is the single highest-leverage input to
this pipeline. Requiring it to be plain text was a bad default - resumes are
PDFs. This accepts .pdf, .txt, .md and .docx, and tells you plainly when it
cannot read one.
"""
from __future__ import annotations

import glob
import os

CANDIDATES = ("resume.pdf", "resume.txt", "resume.md", "resume.docx",
              "cv.pdf", "cv.txt", "cv.md", "cv.docx")


def _from_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(
            f"Found {os.path.basename(path)} but cannot read PDFs. "
            "Run: pip install pypdf")
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if len(text.strip()) < 200:
        raise RuntimeError(
            f"{os.path.basename(path)} yielded almost no text - it is probably a "
            "scanned image rather than a text PDF. Export a text version, or "
            "paste the content into resume.txt instead.")
    return text


def _from_docx(path: str) -> str:
    try:
        import docx
    except ImportError:
        raise RuntimeError(
            f"Found {os.path.basename(path)} but cannot read .docx. "
            "Run: pip install python-docx")
    return "\n".join(p.text for p in docx.Document(path).paragraphs)


def find_resume(directory: str = ".") -> str | None:
    """First matching resume file in the directory, or None."""
    for name in CANDIDATES:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    # Fall back to anything resume-shaped, so resume_2026.pdf still works.
    for pattern in ("resume*.pdf", "resume*.txt", "cv*.pdf", "cv*.txt"):
        hits = sorted(glob.glob(os.path.join(directory, pattern)))
        if hits:
            return hits[0]
    return None


def load_resume(directory: str = ".") -> tuple[str, str | None]:
    """
    Returns (text, source_path). Text is "" when there is nothing to read.

    Never raises on a missing resume - that is a normal, if weaker, run. Does
    raise when a resume exists but cannot be read, because silently treating
    that as "no resume" is how you end up debugging the wrong thing.
    """
    path = find_resume(directory)
    if not path:
        return "", None

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = _from_pdf(path)
    elif ext == ".docx":
        text = _from_docx(path)
    else:
        text = open(path, encoding="utf-8", errors="replace").read()

    if not text.strip():
        raise RuntimeError(f"{os.path.basename(path)} is empty.")
    return text, path
