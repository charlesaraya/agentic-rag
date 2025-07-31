import uuid

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore.init import get_vectorstore

def ingest_documents(urls: list[str]):
    # Load docs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)
    uuids = [str(uuid.uuid4()) for _ in doc_splits]
    # Embedding
    vectorstore = get_vectorstore()
    doc_ids = vectorstore.add_documents(documents=doc_splits, ids=uuids)
    return doc_ids
