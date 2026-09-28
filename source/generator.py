import os
import time

from google.genai.errors import ServerError
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODEL="gemini-3.8-flash"

client= genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(question, context):
    if not context.strip():
        return "I don't have enough information in the knowledge base to answer that question."

    prompt = f"""
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
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            break
        except ServerError:
            if attempt == 2:
                raise

            time.sleep(2)

    return response.text