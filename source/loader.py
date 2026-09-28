from pathlib import Path


def load_documents(data_path="data"):
    documents = []

    data_path = Path(data_path)

    for file_path in data_path.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        if "airlines" in file_path.parts:
            document_type = "airline"
        elif "aircrafts" in file_path.parts:
            document_type = "aircraft"
        else:
            document_type = "unknown"

        entity_id = file_path.stem

        display_name = entity_id.replace("_", " ").title()

        documents.append({
            "source": str(file_path),
            "text": text,
            "type": document_type,
            "entity_id": entity_id,
            "display_name": display_name
        })

    return documents