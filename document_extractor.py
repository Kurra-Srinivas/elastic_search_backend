"""Module for extracting text from PDF and DOCX files"""
import PyPDF2
from docx import Document
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def extract_pdf_text(file_path: Path) -> Optional[str]:
    """Extract text from a PDF file"""
    try:
        text_parts = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract PDF text from {file_path}: {e}")
        return None


def extract_docx_text(file_path: Path) -> Optional[str]:
    """Extract text from a DOCX file"""
    try:
        doc = Document(file_path)
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract DOCX text from {file_path}: {e}")
        return None


def extract_text_from_file(file_path: Path) -> Optional[str]:
    """Extract text from a file (PDF or DOCX)"""
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        return extract_pdf_text(file_path)
    elif file_ext in ['.docx', '.doc']:
        return extract_docx_text(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_ext}")
        return None

