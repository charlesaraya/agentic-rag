import os
import sqlite3

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

def get_llm():
    llm_name = os.environ.get("LLM_NAME")
    llm_provider = os.environ.get("LLM_PROVIDER")

    if not llm_name:
        raise ValueError("failed to load LLM_NAME env")
    if not llm_provider:
        raise ValueError("failed to load LLM_PROVIDER env")

    llm = init_chat_model(model=llm_name, model_provider=llm_provider)
    return llm

def get_agent_memory():
    db_string = os.environ.get("AGENT_STATE_DB_NAME")
    conn = sqlite3.connect(db_string, check_same_thread=False)
    return SqliteSaver(conn)
