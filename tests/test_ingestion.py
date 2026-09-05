"""Tests for ingestion.py"""

import pytest
from unittest.mock import patch, MagicMock
from ingestion import _extract_sentiment, ingestion


# --- Tests for _extract_sentiment ---
# Pure logic  — call the function directly

def test_extract_sentiment_finds_matching_ticker():
    """Should return the score for the exact ticker requested, even when
    other exchange variants are also tagged on the same article."""
    fake_article = {
        "entities": [
            {"symbol": "AAPL.MX", "sentiment_score": 0.1},
            {"symbol": "AAPL", "sentiment_score": 0.65},
            {"symbol": "AAPL.BA", "sentiment_score": 0.2},
        ]
    }
    assert _extract_sentiment(fake_article, "AAPL") == 0.65


def test_extract_sentiment_returns_none_if_ticker_not_tagged():
    """Should return None rather than raising if the requested ticker
    doesn't appear in the article's entities at all."""
    fake_article = {
        "entities": [{"symbol": "META", "sentiment_score": 0.3}]
    }
    assert _extract_sentiment(fake_article, "AAPL") is None


def test_extract_sentiment_handles_missing_entities_key():
    """Should not crash if an article has no 'entities' key at all."""
    fake_article = {}
    assert _extract_sentiment(fake_article, "AAPL") is None


def test_extract_sentiment_handles_empty_entities_list():
    """Should return None (not raise) if entities exists but is empty."""
    fake_article = {"entities": []}
    assert _extract_sentiment(fake_article, "AAPL") is None


# --- Tests for ingestion() ---
# This function "mocks" requests.get so it returns a controlled response 

@patch("ingestion.requests.get")
def test_ingestion_parses_successful_response(mock_get):
    """Should correctly flatten and extract fields from a successful
    API response."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "data": [
            {
                "uuid": "abc-123",
                "title": "SRPT CRASHES",
                "description": "3 people dead from srpt drug",
                "snippet": "3 passed away...",
                "url": "https://example.com/article",
                "published_at": "2026-09-03T12:00:00.000000Z",
                "source": "example.com",
                "entities": [{"symbol": "SRPT", "sentiment_score": -0.8}],
            }
        ]
    }
    mock_get.return_value = fake_response

    articles, failed = ingestion(["SRPT"])

    assert len(articles) == 1
    assert failed == []
    assert articles[0]["uuid"] == "abc-123"
    assert articles[0]["ticker"] == "SRPT"
    assert articles[0]["sentiment_score"] == -0.8


@patch("ingestion.requests.get")
def test_ingestion_collects_failed_tickers(mock_get):
    """A non-200 response should add the ticker to `failed` and 
    shouldn't stop other tickers in the same batch from being processed."""
    fake_response = MagicMock()
    fake_response.status_code = 429  
    mock_get.return_value = fake_response

    articles, failed = ingestion(["SRPT"])

    assert articles == []
    assert failed == ["SRPT"]


@patch("ingestion.requests.get")
def test_ingestion_handles_mixed_success_and_failure(mock_get):
    """One ticker succeeding and another failing in the same batch should
    only affect the failing ticker's result."""
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "data": [
            {
                "uuid": "xyz-789",
                "title": "Meta settlement news",
                "description": "Meta reaches a settlement.",
                "snippet": "Meta reaches...",
                "url": "https://example.com/meta-article",
                "published_at": "2026-09-03T13:00:00.000000Z",
                "source": "example.com",
                "entities": [{"symbol": "META", "sentiment_score": -0.27}],
            }
        ]
    }

    failure_response = MagicMock()
    failure_response.status_code = 500

    # side_effect lets us return a different fake response on each call,
    # matching the order tickers are passed in
    mock_get.side_effect = [success_response, failure_response]

    articles, failed = ingestion(["META", "BADTICKER"])

    assert len(articles) == 1
    assert articles[0]["ticker"] == "META"
    assert failed == ["BADTICKER"]


@patch("ingestion.requests.get")
def test_ingestion_returns_empty_when_no_articles_found(mock_get):
    """A 200 response with an empty data list should produce an empty
    articles list, not an error."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"data": []}
    mock_get.return_value = fake_response

    articles, failed = ingestion(["OBSCURETICKER"])

    assert articles == []
    assert failed == []