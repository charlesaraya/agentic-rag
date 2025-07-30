from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool

import app.config as config
from agent.rag import init_rag
from agent.state import State, GradeDocuments
from agent.prompt import RETRIEVER_SYS_PROMPT, GRADE_PROMPT, GENERATE_PROMPT, REWRITE_PROMPT

llm = config.get_llm()

retriever = init_rag()
retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts.",
    )

def generate_query_or_respond(state: State):
    """Call the model to generate a response based on the current state.

    Given the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    question = state.get("question", state["messages"][0].content)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RETRIEVER_SYS_PROMPT),
        ("placeholder", "{messages}"),
    ])
    chain = prompt | llm.bind_tools([retriever_tool])
    response = chain.invoke(state)
    return {"messages": [response], "question": question}

def grade_retrieved_documents(state: State) -> str:
    """Determine whether the retrieved documents are relevant to the question."""
    context = state["messages"][-1].content
    grader_prompt = GRADE_PROMPT.format(context=context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", grader_prompt),
        ("human", "{question}"),
    ])
    chain = prompt | llm.with_structured_output(GradeDocuments)

    response = chain.invoke(state)
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

def generate_answer(state: State):
    """Generate an answer."""
    state["context"] = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATE_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke(state)

    return {"messages": [response]}

def rewrite_question(state: State):
    """Rewrite the original user question."""
    state["context"] = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke(state)
    question = response.content
    return {"messages": [response], "question": question}

def build_graph():
    graph_builder = StateGraph(State)

    # Nodes
    graph_builder.add_node("generate_query_or_respond", generate_query_or_respond)
    graph_builder.add_node("retrieve", ToolNode([retriever_tool]))
    graph_builder.add_node("rewrite_question", rewrite_question)
    graph_builder.add_node("generate_answer", generate_answer)

    # Control Flow
    graph_builder.add_edge(START, "generate_query_or_respond")
    graph_builder.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition, {
            "tools": "retrieve",
            END: END,
        }
    )
    graph_builder.add_conditional_edges(
        "retrieve",
        grade_retrieved_documents,
        ["generate_answer", "rewrite_question"],
    )
    graph_builder.add_edge("rewrite_question", "generate_query_or_respond")
    graph_builder.add_edge("generate_answer", END)

    # Short-term (within-thread) memory
    #memory = config.get_agent_memory()

    return graph_builder.compile(
        name = "Agentic RAG",
        #checkpointer = memory,
    )

def update_graph(graph, thread_id: str, user_id: str, user_input: str | None = None):
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }
    messages = None
    if user_input:
        messages = {"messages": [HumanMessage(content=user_input)]}
    messages = graph.invoke(messages, config)
    return messages
