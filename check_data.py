from pathlib import Path

data_path = Path("data")

documents=[]

for file_path in data_path.rglob("*.txt"):
    text=file_path.read_text(encoding="utf-8")

    documents.append({
        "source": str(file_path),
        "text": text
    })

print(f"Loaded {len(documents)} documents \n")

for document in documents:
    print("="*50)
    print(document["source"])
    print(document["text"][:300])