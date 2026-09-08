"""Tests for embed_and_store.py"""

import pytest
from unittest.mock import patch, MagicMock
from embed_and_store import embed, store

from qdrant_client.models import VectorParams, Distance, PointStruct
from constants import COLLECTION_NAME

# --- Tests for embed ---

def test_embed_creates_points():
    """Embed should create a PointStruct for each article."""

    articles = [
        {
            "uuid": "1",
            "title": "title 1",
            "description": "desc 1",
        },
        {
            "uuid": "2",
            "title": "title 2",
            "description": "desc 2",
        },
    ]

    result = embed(articles)

    assert len(result) == 2
    assert result[0].id == "1"
    assert result[1].id == "2"
    assert result[0].payload == articles[0]
    assert result[1].payload == articles[1]

def test_embed_returns_empty_list_if_articles_is_empty():
    """Embed should return nothing"""
    articles = []
    result = embed(articles)

    assert len(result) == 0

@patch("embed_and_store.model.encode")
def test_embed_uses_title_and_description(mock_encode):
    """Embed should return the mocked embedding"""
    articles = [
        {
            "uuid": "1",
            "title": "title 1",
            "description": "desc 1",
        }
    ]

    fake_vector = MagicMock()
    fake_vector.tolist.return_value = [0.1, 0.2, 0.3]
    mock_encode.return_value = [fake_vector]

    result = embed(articles)

    mock_encode.assert_called_once_with(["title 1. desc 1"])
    assert result[0].vector == [0.1, 0.2, 0.3]

# --- Tests for store ---

@patch("embed_and_store.client")
def test_store_creates_collection_if_needed(mock_client):
    """Should create the Qdrant collection if it does not already exist."""

    points = [
        PointStruct(
            id="1",
            vector=[0.1, 0.2],
            payload={"title": "test"}
        )
    ]

    mock_client.collection_exists.return_value = False

    store(points)

    mock_client.collection_exists.assert_called_once_with(COLLECTION_NAME)
    mock_client.create_collection.assert_called_once_with(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        ),
    )


@patch("embed_and_store.client")
def test_store_does_not_create_collection_if_exists(mock_client):
    """Should not create the Qdrant collection if it already exists."""

    points = [
        PointStruct(
            id="1",
            vector=[0.1, 0.2],
            payload={"title": "test"}
        )
    ]

    mock_client.collection_exists.return_value = True

    store(points)

    mock_client.create_collection.assert_not_called()


@patch("embed_and_store.client")
def test_store_upserts_points(mock_client):
    """Should upsert the provided points into the Qdrant collection."""

    points = [
        PointStruct(
            id="1",
            vector=[0.1, 0.2],
            payload={"title": "test"}
        ),
        PointStruct(
            id="2",
            vector=[0.3, 0.4],
            payload={"title": "test 2"}
        )
    ]

    mock_client.collection_exists.return_value = True

    store(points)

    mock_client.upsert.assert_called_once_with(
        collection_name=COLLECTION_NAME,
        points=points,
    )

# --- Tests for embed_and_store ---