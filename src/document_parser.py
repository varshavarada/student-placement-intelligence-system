from pathlib import Path
import io

from pypdf import PdfReader
from docx import Document


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


def extract_text_from_pdf(file_bytes):
    reader = PdfReader(
        io.BytesIO(file_bytes)
    )

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages).strip()


def extract_text_from_docx(file_bytes):
    document = Document(
        io.BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(
        paragraphs
    ).strip()


def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode(
            "utf-8"
        ).strip()

    except UnicodeDecodeError:
        return file_bytes.decode(
            "latin-1"
        ).strip()


def extract_document_text(
    uploaded_file,
):
    if uploaded_file is None:
        return {
            "text": "",
            "filename": None,
            "extension": None,
            "success": False,
            "message": (
                "No document was provided."
            ),
        }

    filename = uploaded_file.name

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if (
        extension
        not in SUPPORTED_EXTENSIONS
    ):
        return {
            "text": "",
            "filename": filename,
            "extension": extension,
            "success": False,
            "message": (
                "Unsupported file type. "
                "Please upload PDF, DOCX or TXT."
            ),
        }

    try:
        file_bytes = (
            uploaded_file.getvalue()
        )

        if extension == ".pdf":
            text = extract_text_from_pdf(
                file_bytes
            )

        elif extension == ".docx":
            text = extract_text_from_docx(
                file_bytes
            )

        else:
            text = extract_text_from_txt(
                file_bytes
            )

        if not text:
            return {
                "text": "",
                "filename": filename,
                "extension": extension,
                "success": False,
                "message": (
                    "The document was read, "
                    "but no usable text could "
                    "be extracted."
                ),
            }

        return {
            "text": text,
            "filename": filename,
            "extension": extension,
            "success": True,
            "message": (
                "Document text extracted "
                "successfully."
            ),
        }

    except Exception as error:
        return {
            "text": "",
            "filename": filename,
            "extension": extension,
            "success": False,
            "message": (
                f"Unable to read document: "
                f"{error}"
            ),
        }