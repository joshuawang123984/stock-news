from rank_bm25 import BM25Okapi
from constants import model, client, COLLECTION_NAME

def dense_search(query: str, top_k: int = 20) -> list[dict]:
    """Vector similarity search in Qdrant. Returns similar articles."""
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    return [point.payload for point in results.points]


def sparse_search(query: str, articles: list[dict], top_k: int = 20) -> list[dict]:
    """BM25 keyword search against a corpus of articles already in memory."""

    tokenized_corpus = [a["title"].split() + a["description"].split() for a in articles]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)
    return [article for article, score in ranked[:top_k]]


def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict], k: int = 60) -> list[dict]:
    """Merge two lists into one, using each item's rank position
    so differently-scaled metrics can be combined fairly."""
    scores = {}

    for rank, article in enumerate(dense_results):
        scores[article["uuid"]] = scores.get(article["uuid"], 0) + 1 / (k + rank)

    for rank, article in enumerate(sparse_results):
        scores[article["uuid"]] = scores.get(article["uuid"], 0) + 1 / (k + rank)

    all_articles = {a["uuid"]: a for a in dense_results + sparse_results}
    ranked_uuids = sorted(scores, key=scores.get, reverse=True)

    return [all_articles[uuid] for uuid in ranked_uuids]


def hybrid_search(query: str, articles: list[dict], top_k: int = 10) -> list[dict]:
    """calls dense + sparse search then fuses."""
    dense_results = dense_search(query)
    sparse_results = sparse_search(query, articles)
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    return fused[:top_k]