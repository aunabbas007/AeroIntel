from source.loader import load_documents
from source.chunking import split_documents
from source.vector_store import create_vector_store


documents = load_documents()
chunks = split_documents(documents)

print(f"Loaded documents: {len(documents)}")
print(f"Created chunks: {len(chunks)}")

vector_store = create_vector_store("aerointel-v2")

vector_store.add_documents(chunks)

print("AeroIntel v2 index created and populated.")