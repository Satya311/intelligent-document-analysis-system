import io
import pypdf
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

class DocumentLoader:
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def extract_text_from_pdf(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_data = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_data.append({"page_number": i + 1, "text": text})
        return pages_data

    def extract_text_from_txt(self, raw_text: str) -> List[Dict[str, Any]]:
        return [{"page_number": 1, "text": raw_text}]

    def process_document(self, file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
        if file_name.lower().endswith(".pdf"):
            pages = self.extract_text_from_pdf(file_bytes)
        else:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            pages = self.extract_text_from_txt(raw_text)

        chunks_with_metadata = []
        global_chunk_id = 0

        for page in pages:
            chunks = self.text_splitter.split_text(page["text"])
            for chunk_text in chunks:
                chunks_with_metadata.append({
                    "chunk_id": global_chunk_id,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                    "file_name": file_name
                })
                global_chunk_id += 1

        return chunks_with_metadata
