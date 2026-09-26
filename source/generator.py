import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client= genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def  generate_answer(question,context):
    prompt= f"""You are AeroIntel, an aviation intelligence assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer the question,
say that the information is not available in the knowledge base.

Context:
{context}

Question:
{question}
"""
    response=client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
    )

    return response.text