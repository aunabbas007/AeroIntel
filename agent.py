import os

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from source.tools import (
    search_aviation_knowledge,
    search_airline,
    search_aircraft,
)


load_dotenv()


# ============================================================
# 1. STATE
# ============================================================

class State(MessagesState):
    # Keep the user's wording immutable. Only search_query is rewritten.
    original_question: str
    search_query: str
    context: str
    retrieval_attempts: int
    retrieval_relevant: bool
    evaluation_reason: str
    answer: str


# ============================================================
# 2. LLM AND TOOLS
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL_PRIMARY", "gemini-3.8-flash")
)

tools = [
    search_aviation_knowledge,
    search_airline,
    search_aircraft,
]

llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


# ============================================================
# 3. RETRIEVAL EVALUATION
# ============================================================

class RetrievalEvaluation(BaseModel):
    relevant: bool
    reason: str


evaluator = llm.with_structured_output(RetrievalEvaluation)


def _content_to_text(content) -> str:
    """Normalize provider message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content)


# ============================================================
# 4. AGENT NODE
# ============================================================

def agent(state: State):
    system_message = """
You are AeroIntel, an aviation intelligence agent.

Retrieve evidence from the AeroIntel knowledge base before
answering. You must call at least one appropriate tool.

Choose the most specific tool:
- search_airline for airline-specific questions
- search_aircraft for aircraft-specific questions
- search_aviation_knowledge for general aviation questions

Do not answer from your own knowledge. Use the current search
query as the retrieval query.
"""

    response = llm_with_tools.invoke(
        [
            ("system", system_message),
            ("human", state["search_query"]),
        ]
    )

    return {"messages": [response]}


# ============================================================
# 5. ROUTE AFTER AGENT
# ============================================================

def route_after_agent(state: State):
    last_message = state["messages"][-1]

    # Never generate an answer when the model skipped retrieval.
    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "insufficient"


# ============================================================
# 6. COLLECT AND EVALUATE THIS ATTEMPT ONLY
# ============================================================

def evaluate_retrieval(state: State):
    messages = state["messages"]
    last_ai_message = next(
        (
            message
            for message in reversed(messages)
            if getattr(message, "tool_calls", None)
        ),
        None,
    )

    if last_ai_message is None:
        return {
            "context": "",
            "retrieval_relevant": False,
            "evaluation_reason": "No retrieval tool call was made.",
        }

    # Match only ToolMessages generated for the latest AI tool-call batch.
    current_tool_call_ids = {
        tool_call["id"] for tool_call in last_ai_message.tool_calls
    }
    current_tool_messages = [
        message
        for message in messages
        if message.type == "tool"
        and getattr(message, "tool_call_id", None) in current_tool_call_ids
    ]

    context = "\n\n".join(
        _content_to_text(message.content)
        for message in current_tool_messages
        if message.content
    ).strip()

    if not context:
        return {
            "context": "",
            "retrieval_relevant": False,
            "evaluation_reason": "The retrieval tools returned no usable content.",
        }

    evaluation_prompt = f"""
Assess whether the retrieved evidence is sufficient to answer
the user's original aviation question.

Original user question:
{state["original_question"]}

Search query used for this attempt:
{state["search_query"]}

Retrieved evidence:
{context}

Set relevant=true only if this evidence directly supports a
useful answer to the original question. Otherwise set relevant=false.
Give a concise reason that can guide a better search query.
"""

    result = evaluator.invoke(evaluation_prompt)

    return {
        "context": context,
        "retrieval_relevant": result.relevant,
        "evaluation_reason": result.reason,
    }


# ============================================================
# 7. ROUTE AFTER EVALUATION
# ============================================================

def route_after_evaluation(state: State):
    if state["retrieval_relevant"]:
        return "generate"

    if state["retrieval_attempts"] >= 3:
        return "insufficient"

    return "rewrite"


# ============================================================
# 8. QUERY REWRITE
# ============================================================

def rewrite_query(state: State):
    rewrite_prompt = f"""
Rewrite the search query for the AeroIntel aviation knowledge base.

Original user question:
{state["original_question"]}

Previous search query:
{state["search_query"]}

Why the previous retrieval was insufficient:
{state["evaluation_reason"]}

Create a more specific search query that addresses the missing
evidence. Preserve the user's actual intent. Return only the query.
"""

    response = llm.invoke(rewrite_prompt)
    rewritten_query = _content_to_text(response.content).strip()

    if not rewritten_query:
        rewritten_query = state["search_query"]

    return {
        "search_query": rewritten_query,
        "retrieval_attempts": state["retrieval_attempts"] + 1,
        "context": "",
        "retrieval_relevant": False,
        "evaluation_reason": "",
    }


# ============================================================
# 9. GENERATE FINAL ANSWER
# ============================================================

def generate_answer(state: State):
    prompt = f"""
You are AeroIntel, an aviation intelligence assistant.

Answer the original user question using ONLY the retrieved
knowledge-base evidence below. Do not use outside knowledge.
If the evidence does not support a detail, do not invent it.

Retrieved evidence:
{state["context"]}

Original user question:
{state["original_question"]}
"""

    response = llm.invoke(prompt)

    return {"answer": _content_to_text(response.content)}


# ============================================================
# 10. INSUFFICIENT EVIDENCE
# ============================================================

def insufficient_evidence(state: State):
    return {
        "answer": (
            "I don't have enough information in the knowledge base "
            "to answer that question."
        )
    }


# ============================================================
# 11. BUILD GRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_node("evaluate", evaluate_retrieval)
graph.add_node("rewrite", rewrite_query)
graph.add_node("generate", generate_answer)
graph.add_node("insufficient", insufficient_evidence)

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "insufficient": "insufficient",
    },
)

graph.add_edge("tools", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "insufficient": "insufficient",
    },
)

graph.add_edge("rewrite", "agent")
graph.add_edge("generate", END)
graph.add_edge("insufficient", END)

app = graph.compile()


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":
    question = input("Ask AeroIntel: ").strip()

    if not question:
        raise SystemExit("Please enter a question.")

    result = app.invoke(
        {
            "messages": [],
            "original_question": question,
            "search_query": question,
            "context": "",
            "retrieval_attempts": 1,
            "retrieval_relevant": False,
            "evaluation_reason": "",
            "answer": "",
        }
    )

    print("\nAeroIntel:")
    print(result["answer"])
