def build_context(results):
    context = []

    for match in results["matches"]:
        text = match["metadata"]["text"]
        source = match["metadata"]["source"]

        context.append(
            f"Source: {source}\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(context)