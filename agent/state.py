from pydantic import BaseModel, Field

from langgraph.graph import MessagesState

class State(MessagesState):
    question: str
    context: str


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )
