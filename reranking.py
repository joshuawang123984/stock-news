from constants import reranker

def rerank(query: str, articles: list[dict], top_k: int = 5) -> list[dict]:
    """Re-scores a list of candidate articles against the query using a
    cross-encoder, and returns the top_k most relevant.

    Unlike dense_search/sparse_search, this looks at the query and each
    article together (not as separately-computed embeddings), producing
    a more accurate judgment.

    Args:
        query: The search query text.
        articles: Candidate articles to re-score (already narrowed down
            by hybrid_search — not the full corpus).
        top_k: How many of the reranked results to return.

    Returns:
        articles, re-sorted by cross-encoder relevance score, truncated
        to top_k.
    """
    pairs = [[query, f"{a['title']}. {a['description']}"] for a in articles]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)
    return [article for article, _ in ranked[:top_k]]