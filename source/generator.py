from  langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are AeroIntel, an aviation intelligence assistant.

    Answer the user's question using only the provided context.

    If the context does not contain enough information to answer the question,
    say exactly:

    "I don't have enough information in the knowledge base to answer that question."

    Do not use outside knowledge.

    Context:
    {context}

    Question:
    {question}
    """
)