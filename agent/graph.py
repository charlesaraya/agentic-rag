from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool

import app.config as config
from agent.rag import init_rag
from agent.state import State
from agent.prompt import RETRIEVER_SYS_PROMPT, GENERATE_PROMPT

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
    question = state["messages"][0].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", RETRIEVER_SYS_PROMPT),
        ("placeholder", "{messages}"),
    ])
    chain = prompt | llm.bind_tools([retriever_tool])
    response = chain.invoke(state)
    return {"messages": [response], "question": question}


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

def build_graph():
    graph_builder = StateGraph(State)

    # Nodes
    graph_builder.add_node("generate_query_or_respond", generate_query_or_respond)
    graph_builder.add_node("retrieve", ToolNode([retriever_tool]))
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
    graph_builder.add_edge("retrieve", "generate_answer")
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
