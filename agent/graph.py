from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

import agent.config as config
from agent.state import GraphState, GradeDocuments, GradeHallucinations
import agent.prompt as prompts

import vectorstore

llm = config.get_llm()

web_search_tool = config.get_search_tool()

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
vectorstore.ingest_documents(urls)
retriever = vectorstore.get_retriever()

def grade_retrieved_documents(state: GraphState):
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for doc in documents:
        context = doc.page_content.replace("{", "{{").replace("}", "}}")
        grader_prompt = prompts.GRADE_PROMPT.format(context=context)
        prompt = ChatPromptTemplate.from_messages([
            ("system", grader_prompt),
            ("human", "{question}"),
        ])
        chain = prompt | llm.with_structured_output(GradeDocuments)
        response = chain.invoke(state)
        score = response.binary_score
        if score == "yes":
            filtered_docs.append(doc)
        else:
            continue
    return {"documents": filtered_docs, "question": question}

def grade_generation(state: GraphState) -> str:
    """Determine whether an LLM generation is grounded in or supported by retrieved facts."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.HALLUCINATION_PROMPT),
    ])
    chain = prompt | llm.with_structured_output(GradeHallucinations)
    response = chain.invoke(state)
    score = response.binary_score

    if score == "yes":
        return "supported"
    else:
        return "not supported"

def generate_answer(state: GraphState):
    """Generate an answer."""
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents)
    state["context"] = context.replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.GENERATE_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke(state)
    generation = response.content
    return {"messages": [response], "generation": generation}

def rewrite_question(state: GraphState):
    """Rewrite the original user question to improve it."""
    question = state["messages"][0].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.REWRITE_PROMPT),
        ("human", question),
    ])
    chain = prompt | llm
    response = chain.invoke(state)
    question = response.content
    return {"messages": [response], "question": question}

def retrieve(state: GraphState):
    """Retrieves documents from a vector store"""
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def web_search(state: GraphState):
    """Web search based a user question."""
    search_results = web_search_tool.invoke({"query": state["question"]})
    documents = list(map(lambda doc: Document(
        page_content = doc["content"],
        metadata = {"source": doc["url"], "title": doc["title"]}), search_results["results"]))
    return {"documents": documents}

def generate_or_search(state: GraphState) -> str:
    """Determines whether to generate an answer, or search the web to gather context."""
    filtered_documents = state["documents"]
    if not filtered_documents:
        return "web_search"
    else:
        return "generate_answer"

def build_graph():
    graph_builder = StateGraph(GraphState)

    # Nodes
    graph_builder.add_node("rewrite_question", rewrite_question)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("grade_retrieved_documents", grade_retrieved_documents)
    graph_builder.add_node("web_search", web_search)
    graph_builder.add_node("generate_answer", generate_answer)

    # Control Flow
    graph_builder.add_edge(START, "rewrite_question")
    graph_builder.add_edge("rewrite_question", "retrieve")
    graph_builder.add_edge("retrieve", "grade_retrieved_documents")
    graph_builder.add_conditional_edges("grade_retrieved_documents", generate_or_search, ["generate_answer", "web_search"])
    graph_builder.add_edge("web_search", "grade_retrieved_documents")
    graph_builder.add_conditional_edges("generate_answer", grade_generation, {"supported": END, "not supported": "generate_answer"},)

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
