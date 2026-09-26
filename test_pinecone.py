from source.vector_store import create_index


index = create_index("aerointel-v1")

print("Pinecone connection successful!")
print(index)