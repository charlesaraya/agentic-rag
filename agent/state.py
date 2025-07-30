from typing import List
from pydantic import BaseModel, Field

from langgraph.graph import MessagesState

class GraphState(MessagesState):
    """
    Represents the state of the graph.

    Attributes:
        question: user question
        context: retrieved context related to the question
    """
    question: str
    documents: List[str]
    context: str


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )
