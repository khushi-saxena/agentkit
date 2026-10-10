"""
Resume loading. A resume is the highest-leverage input to this pipeline, and
requiring plain text was a bad default - resumes are PDFs.
"""
import pytest

from app.resume import find_resume, load_resume


def test_missing_resume_is_not_an_error(tmp_path):
    """A run without a resume is weaker, not broken."""
    text, source = load_resume(str(tmp_path))
    assert text == ""
    assert source is None


def test_plain_text_resume(tmp_path):
    (tmp_path / "resume.txt").write_text("Software Engineer at UBS. " * 20)
    text, source = load_resume(str(tmp_path))
    assert "UBS" in text
    assert source.endswith("resume.txt")


def test_pdf_resume_is_read(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    pytest.importorskip("pypdf")
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "resume.pdf"
    c = canvas.Canvas(str(pdf))
    y = 760
    for line in ["Software Engineer at UBS",
                 "Led Oracle to PostgreSQL migration, 100% data consistency",
                 "Research Intern at DRDO, multi-threaded C++ simulation",
                 "LogLens: FastAPI, React, WebSockets, OpenAI"] * 4:
        c.drawString(50, y, line)
        y -= 14
    c.save()

    text, source = load_resume(str(tmp_path))
    assert source.endswith("resume.pdf")
    for probe in ("UBS", "PostgreSQL", "DRDO", "LogLens"):
        assert probe in text


def test_pdf_is_preferred_over_txt(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "resume.txt").write_text("x" * 500)
    (tmp_path / "resume.pdf").write_bytes(b"not really a pdf")
    assert find_resume(str(tmp_path)).endswith("resume.pdf")


def test_scanned_pdf_raises_rather_than_silently_returning_nothing(tmp_path):
    """
    An image-only PDF extracts no text. Treating that as "no resume" sends you
    debugging the wrong thing, so it raises with the actual cause.
    """
    pytest.importorskip("pypdf")
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(tmp_path / "resume.pdf"))
    c.drawString(50, 700, "hi")          # far under the 200-char floor
    c.save()

    with pytest.raises(RuntimeError, match="scanned image"):
        load_resume(str(tmp_path))


def test_numbered_filename_still_found(tmp_path):
    (tmp_path / "resume_2026_final.txt").write_text("UBS " * 100)
    assert find_resume(str(tmp_path)) is not None


def test_empty_resume_file_raises(tmp_path):
    (tmp_path / "resume.txt").write_text("   \n  ")
    with pytest.raises(RuntimeError, match="empty"):
        load_resume(str(tmp_path))
