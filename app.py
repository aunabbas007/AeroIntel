from langchain_core.runnables import RunnablePassthrough

from source.loader import load_documents
from source.chunking import split_documents
from source.vector_store import create_vector_store, create_retriever
from source.context import format_docs
from source.generator import prompt, llm


# -----------------------------
# Load documents
# -----------------------------

documents = load_documents()


# -----------------------------
# Split documents
# -----------------------------

chunks = split_documents(documents)


# -----------------------------
# Create vector store
# -----------------------------

vector_store = create_vector_store(
    "aerointel-v2"
)


# -----------------------------
# Add documents to Pinecone
# -----------------------------

vector_store.add_documents(chunks)


# -----------------------------
# Create retriever
# -----------------------------

retriever = create_retriever(
    vector_store,
    k=3
)


# -----------------------------
# Build RAG chain
# -----------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# -----------------------------
# Ask AeroIntel
# -----------------------------

if __name__ == "__main__":

    question = input("Ask AeroIntel: ")

    response = rag_chain.invoke(question)

    print("\nAeroIntel:")
    print(response.content)