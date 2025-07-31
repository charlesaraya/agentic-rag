import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def get_vectorstore():
    """Loads an existing Chroma vector store collection"""
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME")
    if not collection_name:
        raise ValueError("failed to load CHROMA_COLLECTION_NAME envs")
    persist_directory = os.environ.get("CHROMA_DIR")
    if not persist_directory:
        raise ValueError("failed to load CHROMA_DIR envs")
    return Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )

def get_retriever():
    return get_vectorstore().as_retriever()