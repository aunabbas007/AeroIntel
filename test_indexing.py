from dataclasses import dataclass

from source.indexing import stable_document_ids


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict


def test_document_ids_are_repeatable_and_unique_per_source_chunk():
    documents = [
        FakeDocument("Aircraft facts", {"source": "data/aircraft.txt"}),
        FakeDocument("More aircraft facts", {"source": "data/aircraft.txt"}),
        FakeDocument("Airline facts", {"source": "data/airline.txt"}),
    ]

    first = stable_document_ids(documents)
    second = stable_document_ids(documents)

    assert first == second
    assert len(set(first)) == len(documents)


def test_document_ids_normalize_windows_and_posix_paths():
    windows_document = FakeDocument(
        "Facts",
        {"source": r"data\aircraft.txt"},
    )
    posix_document = FakeDocument(
        "Facts",
        {"source": "data/aircraft.txt"},
    )

    assert stable_document_ids([windows_document]) == stable_document_ids(
        [posix_document]
    )
