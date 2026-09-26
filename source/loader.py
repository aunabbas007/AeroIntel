from pathlib import Path


def load_documents(data_path="data"):
    documents = []

    data_path = Path(data_path)

    for file_path in data_path.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": str(file_path),
            "text": text
        })

    return documents