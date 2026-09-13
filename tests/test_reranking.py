"""Tests for reranking.py"""

import pytest
from unittest.mock import patch
from reranking import rerank

@patch("reranking.get_reranker")
def test_rerank_sorts_by_cross_encoder_score(mock_get_reranker):
    """Articles should come back sorted by cross-encoder score descending."""
    articles = [
        {"uuid": "1", "title": "Low relevance article", "description": "..."},
        {"uuid": "2", "title": "High relevance article", "description": "..."},
        {"uuid": "3", "title": "Medium relevance article", "description": "..."},
    ]

    mock_get_reranker.return_value.predict.return_value = [0.1, 0.9, 0.5]

    results = rerank("some query", articles, top_k=3)

    assert [a["uuid"] for a in results] == ["2", "3", "1"]


@patch("reranking.get_reranker")
def test_rerank_respects_top_k(mock_get_reranker):
    """Should never return more than top_k results"""
    articles = [{"uuid": str(i), "title": f"Article {i}", "description": "..."} for i in range(5)]
    mock_get_reranker.return_value.predict.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = rerank("some query", articles, top_k=2)

    assert len(results) == 2


@patch("reranking.get_reranker")
def test_rerank_handles_empty_input(mock_get_reranker):
    """Should not raise on an empty article list."""
    mock_get_reranker.return_value.predict.return_value = []

    results = rerank("some query", [], top_k=5)

    assert results == []


@patch("reranking.get_reranker")
def test_rerank_builds_correct_query_article_pairs(mock_get_reranker):
    """Should pass [query, title+description] pairs to the cross-encoder"""
    articles = [{"uuid": "1", "title": "Apple earnings", "description": "Record quarter."}]
    mock_get_reranker.return_value.predict.return_value = [0.5]

    rerank("query about apple", articles, top_k=1)

    called_pairs = mock_get_reranker.return_value.predict.call_args[0][0]
    assert called_pairs == [["query about apple", "Apple earnings. Record quarter."]]