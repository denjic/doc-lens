import pytest

from rag.prompt_manager import PromptManager


def test_prompt_manager_renders_context_and_query():
    manager = PromptManager(version="v1")

    system, user = manager.render("A source passage", "What is the answer?")

    assert "A source passage" in user
    assert "What is the answer?" in user
    assert system


def test_prompt_manager_rejects_unknown_versions():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        PromptManager(version="missing")


def test_format_chunks_as_context_includes_citations_and_separators():
    context = PromptManager.format_chunks_as_context(
        [
            {
                "document": "First passage",
                "metadata": {"source": "one.pdf", "page_num": 1},
            },
            {
                "document": "Second passage",
                "metadata": {"source": "two.pdf", "page_num": 4},
            },
        ]
    )

    assert "[1] [Source: one.pdf, Page: 1]" in context
    assert "[2] [Source: two.pdf, Page: 4]" in context
    assert "First passage\n\n---\n\n" in context
