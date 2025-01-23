from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from vectordb_handler import load_vectordb
from utils import load_config, timeit
from typing import List
from io import BytesIO
from langchain_core.documents.base import Document

import pypdfium2

config = load_config()

def get_pdf_texts(pdfs_bytes_list: List[BytesIO]) -> List[str]:
    """
    Extracts text content from a list of PDF files.

    Args:
        pdfs_bytes_list (List[BytesIO]): A list of BytesIO objects containing the PDF files' bytes.

    Returns:
        List[str]: A list of strings, each containing the text content of a PDF file.
    """
    return [extract_text_from_pdf(pdf_bytes.getvalue()) for pdf_bytes in pdfs_bytes_list]

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts text content from a single PDF file.

    Args:
        pdf_bytes (bytes): The bytes of the PDF file.

    Returns:
        str: The extracted text content of the PDF file.
    """
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range() for page_number in range(len(pdf_file)))
    
def get_text_chunks(text: str) -> List[str]:
    """
    Splits a text string into smaller chunks based on the configured chunk size and overlap.

    Args:
        text (str): The text string to be split into chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=config["pdf_text_splitter"]["chunk_size"], 
                                              chunk_overlap=config["pdf_text_splitter"]["overlap"],
                                                separators=config["pdf_text_splitter"]["separators"])
    return splitter.split_text(text)

def get_document_chunks(text_list: List[str]) -> List[Document]:
    """
    Splits a list of text strings into smaller chunks and creates Document objects for each chunk.

    Args:
        text_list (List[str]): A list of text strings to be split into chunks.

    Returns:
        List[Document]: A list of Document objects, each containing a chunk of text.
    """
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content = chunk))
    return documents

@timeit
def add_documents_to_db(pdfs_bytes: List[BytesIO]) -> None:
    """
    Adds the text content of PDF files to the vector database.

    Args:
        pdfs_bytes (List[BytesIO]): A list of BytesIO objects containing the PDF files' bytes.

    Returns:
        None
    """

    texts = get_pdf_texts(pdfs_bytes)
    documents = get_document_chunks(texts)
    vector_db = load_vectordb()
    vector_db.add_documents(documents)
    print("Documents added to db.")