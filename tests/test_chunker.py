from ingestion.chunker import SemanticChunker
from ingestion.pdf_extractor import PageContent


def test_chunk_pages_preserves_source_metadata_and_adds_context_header():
    page = PageContent(
        page_num=2,
        source="invoice.pdf",
        text="This is a sufficiently long sentence for testing. "
        "Another sufficiently long sentence keeps the chunk above the minimum.",
        has_tables=True,
        metadata={"department": "finance"},
    )

    chunks = SemanticChunker(chunk_size=200, min_chars=20).chunk_pages([page])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source == "invoice.pdf"
    assert chunk.page_num == 2
    assert chunk.text.startswith("[Source: invoice.pdf | Page: 2]\n")
    assert chunk.metadata["department"] == "finance"
    assert chunk.metadata["has_tables"] is True
    assert chunk.metadata["chunk_index"] == 0


def test_chunk_pages_skips_empty_pages_and_short_chunks():
    pages = [
        PageContent(page_num=1, source="empty.pdf", text="   "),
        PageContent(page_num=2, source="short.pdf", text="Too short."),
    ]

    chunks = SemanticChunker(min_chars=100).chunk_pages(pages)

    assert chunks == []


def test_chunk_ids_are_deterministic_for_same_page_and_index():
    page = PageContent(
        page_num=1,
        source="report.pdf",
        text="A sentence that is long enough to produce a stable chunk.",
    )
    chunker = SemanticChunker(min_chars=10)

    first = chunker.chunk_pages([page])[0]
    second = chunker.chunk_pages([page])[0]

    assert first.id == second.id
