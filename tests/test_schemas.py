import pytest
from pydantic import ValidationError

from backend.schemas import QueryRequest


def test_query_request_uses_documented_defaults():
    request = QueryRequest(question="What is the total?")

    assert request.top_k == 8
    assert request.rerank_top_k == 4
    assert request.use_mmr is True
    assert request.prompt_version == "v2"


@pytest.mark.parametrize("question", ["", "ab"])
def test_query_request_rejects_questions_shorter_than_three_characters(question):
    with pytest.raises(ValidationError):
        QueryRequest(question=question)


def test_query_request_rejects_out_of_range_retrieval_limits():
    with pytest.raises(ValidationError):
        QueryRequest(question="Valid question", top_k=21)

    with pytest.raises(ValidationError):
        QueryRequest(question="Valid question", rerank_top_k=11)
