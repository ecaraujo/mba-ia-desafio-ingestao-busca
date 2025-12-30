import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()
for k in ("OPENAI_API_KEY", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

project_root = Path(__file__).resolve().parents[1]

PDF_PATH = (project_root / os.getenv("PDF_PATH", "document.pdf")).resolve()

print("PDF:", PDF_PATH)

docs = PyPDFLoader(str(PDF_PATH)).load()

splits = RecursiveCharacterTextSplitter(
    chunk_size=700, 
    chunk_overlap=250, add_start_index=False).split_documents(docs)
if not splits:
    raise SystemExit(0)

enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
    for d in splits
]    

ids = [f"doc-{i}" for i in range(len(enriched))]

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-ada-002"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)

store.add_documents(documents=enriched, ids=ids)