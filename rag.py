import os
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)
DashScope_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DashScope_API_KEY
)

FAISS_DIR = "faiss_db_bond"


def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def vector_store(chunks):
    db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    db.save_local(FAISS_DIR)


def check_database_exists():
    return Path(FAISS_DIR).exists() and (Path(FAISS_DIR) / "index.faiss").exists()
