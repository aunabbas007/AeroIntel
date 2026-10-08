from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END
)

from langgraph.prebuilt import ToolNode

from source.tools import (
    search_aviation_knowledge,
    search_airline,
    search_aircraft
)


load_dotenv()


# ============================================================
# 1. STATE
# ============================================================

class State(MessagesState):

    question: str

    context: str

    retrieval_attempts: int

    retrieval_relevant: bool

    answer: str


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)


# ============================================================
# 3. TOOLS
# ============================================================

tools = [
    search_aviation_knowledge,
    search_airline,
    search_aircraft
]


llm_with_tools = llm.bind_tools(
    tools
)


tool_node = ToolNode(
    tools
)


# ============================================================
# 4. RETRIEVAL EVALUATION
# ============================================================

class RetrievalEvaluation(BaseModel):

    relevant: bool

    reason: str


evaluator = llm.with_structured_output(
    RetrievalEvaluation
)


# ============================================================
# 5. AGENT NODE
# ============================================================

def agent(state: State):

    system_message = """
You are AeroIntel, an aviation intelligence agent.

You answer questions about airlines and aircraft using
the available AeroIntel knowledge-base tools.

You MUST use an appropriate tool to retrieve information
before answering the user's question.

Choose the most specific tool available:

- search_airline for airline-specific questions
- search_aircraft for aircraft-specific questions
- search_aviation_knowledge for general aviation questions

Do not answer from your own knowledge.
"""

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": state["question"]
        }
    ]

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# 6. ROUTE AFTER AGENT
# ============================================================

def route_after_agent(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "generate"


# ============================================================
# 7. EVALUATE TOOL RESULTS
# ============================================================

def evaluate_retrieval(state: State):

    tool_messages = [
        message
        for message in state["messages"]
        if message.type == "tool"
    ]

    if not tool_messages:

        return {
            "retrieval_relevant": False,
            "context": ""
        }

    context = "\n\n".join(
        message.content
        for message in tool_messages
    )

    evaluation_prompt = f"""
You are evaluating retrieved information for an
aviation question.

Question:
{state["question"]}

Retrieved information:
{context}

Determine whether the retrieved information contains
enough relevant evidence to answer the question.

Return relevant=True only when the retrieved information
directly supports answering the question.

Return relevant=False if the information is unrelated,
insufficient, or does not contain enough evidence.
"""

    result = evaluator.invoke(
        evaluation_prompt
    )

    return {
        "context": context,
        "retrieval_relevant": result.relevant
    }


# ============================================================
# 8. ROUTE AFTER EVALUATION
# ============================================================

def route_after_evaluation(state: State):

    if state["retrieval_relevant"]:

        return "generate"

    if state["retrieval_attempts"] >= 3:

        return "insufficient"

    return "rewrite"


# ============================================================
# 9. QUERY REWRITE
# ============================================================

def rewrite_query(state: State):

    rewrite_prompt = f"""
Rewrite the user's question into a better search query
for the AeroIntel aviation knowledge base.

Original question:
{state["question"]}

The previous retrieval attempt did not provide enough
relevant information.

Make the new query more specific and useful for semantic
search.

Return ONLY the rewritten search query.
"""

    response = llm.invoke(
        rewrite_prompt
    )

    return {
        "question": response.content,
        "retrieval_attempts": (
            state["retrieval_attempts"] + 1
        ),
        "context": "",
        "retrieval_relevant": False
    }


# ============================================================
# 10. GENERATE FINAL ANSWER
# ============================================================

def generate_answer(state: State):

    prompt = ChatPromptTemplate.from_template(
        """
You are AeroIntel, an aviation intelligence assistant.

Answer the user's question using ONLY the provided
knowledge-base context.

Do not use outside knowledge.

If the context does not contain enough information,
say:

"I don't have enough information in the knowledge base
to answer that question."

Knowledge-base context:

{context}

User question:

{question}
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": state["context"],
            "question": state["question"]
        }
    )

    return {
        "answer": response.content
    }


# ============================================================
# 11. INSUFFICIENT EVIDENCE
# ============================================================

def insufficient_evidence(state: State):

    return {
        "answer": (
            "I don't have enough information in the "
            "knowledge base to answer that question."
        )
    }


# ============================================================
# 12. BUILD GRAPH
# ============================================================

graph = StateGraph(State)


graph.add_node(
    "agent",
    agent
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "evaluate",
    evaluate_retrieval
)

graph.add_node(
    "rewrite",
    rewrite_query
)

graph.add_node(
    "generate",
    generate_answer
)

graph.add_node(
    "insufficient",
    insufficient_evidence
)


# ============================================================
# 13. GRAPH EDGES
# ============================================================

graph.add_edge(
    START,
    "agent"
)


graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "generate": "generate"
    }
)


graph.add_edge(
    "tools",
    "evaluate"
)


graph.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "insufficient": "insufficient"
    }
)


graph.add_edge(
    "rewrite",
    "agent"
)


graph.add_edge(
    "generate",
    END
)


graph.add_edge(
    "insufficient",
    END
)


# ============================================================
# 14. COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# 15. RUN
# ============================================================

if __name__ == "__main__":

    question = input(
        "Ask AeroIntel: "
    )

    result = app.invoke(
        {
            "messages": [],
            "question": question,
            "context": "",
            "retrieval_attempts": 1,
            "retrieval_relevant": False,
            "answer": ""
        }
    )

    print("\nAeroIntel:")
    print(result["answer"])