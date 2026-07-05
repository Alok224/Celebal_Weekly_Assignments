import os
import re
import fitz
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(file_path):
    pages = []
    try:
        doc = fitz.open(file_path)
        for page in doc:
            pages.append(page.get_text())
        doc.close()
    except Exception as e:
        print("PyMuPDF failed, falling back to pypdf:", e)
        reader = PdfReader(file_path)
        for page in reader.pages:
            pages.append(page.extract_text() or "")

    print("Document Loaded:", file_path)
    print("Pages Loaded:", len(pages))
    return pages


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print("Document Loaded:", file_path)
    print("Pages Loaded: 1")
    return [text]


def load_hf_dataset(dataset_name, text_column="text", split="train", limit=200):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    texts = list(ds[text_column][:limit])
    print("Document Loaded: HuggingFace dataset -", dataset_name)
    print("Pages Loaded:", len(texts))
    return texts


def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        pages = load_pdf(file_path)
    elif ext == ".txt":
        pages = load_txt(file_path)
    else:
        raise ValueError("Unsupported file type: " + ext)
    return pages


def chunk_text(pages, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for page in pages:
        cleaned = clean_text(page)
        if cleaned == "":
            continue
        page_chunks = splitter.split_text(cleaned)
        chunks.extend(page_chunks)

    chunks = [c for c in chunks if c.strip() != ""]
    print("Chunks Created:", len(chunks))
    return chunks
