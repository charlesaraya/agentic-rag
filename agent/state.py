from typing import TypedDict

from langgraph.graph import MessagesState

class State(MessagesState):
    question: str
    context: str