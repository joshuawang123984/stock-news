"""Tests for hybrid search: dense_search, sparse_search, and
reciprocal_rank_fusion."""

import pytest
from unittest.mock import patch, MagicMock

from hybrid_search import sparse_search, reciprocal_rank_fusion, dense_search


# --- reciprocal_rank_fusion ---
# Pure logic, no dependencies — test directly.

def test_rrf_favors_articles_ranked_highly_in_both_lists():
    """An article near the top of both lists should outrank one that's
    only in one list, even if that one is ranked #1 there."""
    article_a = {"uuid": "a"}
    article_b = {"uuid": "b"}
    article_c = {"uuid": "c"}

    dense_results = [article_a, article_b]   # a is #1, b is #2
    sparse_results = [article_c, article_a]  # c is #1, a is #2

    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    # 'a' appears in both lists (ranked well in each), so it should win
    # even though 'c' was ranked #1 in the sparse list alone.
    assert fused[0]["uuid"] == "a"


def test_rrf_handles_no_overlap_between_lists():
    """Articles that only appear in one list should still be included,
    just ranked lower than overlapping ones."""
    dense_results = [{"uuid": "a"}]
    sparse_results = [{"uuid": "b"}]

    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    result_uuids = {a["uuid"] for a in fused}
    assert result_uuids == {"a", "b"}


def test_rrf_handles_empty_inputs():
    """Should not raise on empty lists."""
    assert reciprocal_rank_fusion([], []) == []


# --- sparse_search ---
# Uses real BM25 computation — deterministic, no network calls, no
# reason to mock this.

def test_sparse_search_ranks_keyword_match_higher():
    """An article whose title/description actually contains the query
    terms should rank above one that doesn't."""
    articles = [
        {"uuid": "1", "title": "Apple reports record iPhone sales", "description": "Strong quarter."},
        {"uuid": "2", "title": "Local weather update", "description": "Rain expected tomorrow."},
    ]

    results = sparse_search("iPhone sales", articles, top_k=2)

    assert results[0]["uuid"] == "1"


def test_sparse_search_respects_top_k():
    """Should never return more than top_k results."""
    articles = [
        {"uuid": str(i), "title": f"Article {i} about stocks", "description": "Stock market news."}
        for i in range(5)
    ]

    results = sparse_search("stocks", articles, top_k=2)

    assert len(results) == 2


# --- dense_search ---
# Talks to a real model and a real Qdrant instance — mock both so this
# test doesn't depend on Qdrant being up or actually loading model weights.

@patch("hybrid_search.client")
@patch("hybrid_search.model")
def test_dense_search_returns_payloads_from_qdrant_results(mock_model, mock_client):
    """Should return the payload of each point Qdrant returns, in order."""
    mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    fake_point = MagicMock()
    fake_point.payload = {"uuid": "abc", "title": "Test article"}

    mock_response = MagicMock()
    mock_response.points = [fake_point]
    mock_client.query_points.return_value = mock_response

    results = dense_search("some query", top_k=5)

    assert results == [{"uuid": "abc", "title": "Test article"}]
    mock_client.query_points.assert_called_once()