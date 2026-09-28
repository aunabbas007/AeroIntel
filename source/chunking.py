def chunk_text(text, chunk_size=300, overlap=50):
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        paragraph_size = len(paragraph_words)

        if current_size + paragraph_size <= chunk_size:
            current_chunk.extend(paragraph_words)
            current_size += paragraph_size

        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            overlap_words = current_chunk[-overlap:] if overlap else []

            current_chunk = overlap_words + paragraph_words
            current_size = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks