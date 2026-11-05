from source.loader import load_documents
from source.chunking import split_documents
from source.indexing import stable_document_ids
from source.vector_store import create_vector_store, ensure_vector_index


documents = load_documents()
chunks = split_documents(documents)

print(f"Loaded documents: {len(documents)}")
print(f"Created chunks: {len(chunks)}")

ensure_vector_index()
vector_store = create_vector_store()

vector_store.add_documents(
    chunks,
    ids=stable_document_ids(chunks),
)

print("AeroIntel v2 index created and populated.")
