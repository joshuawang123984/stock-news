import json
from ingestion import ingestion
from hybrid_search import hybrid_search

EVAL_SET_PATH = "eval_set.json"

def precision_at_k(returned_uuids: list[str], relevant_uuids: list[str]) -> float:
    """Fraction of returned results that were judged relevant.

    Returns 1.0 if correctly found nothing when nothing existed, and 0.0 if 
    results were returned but none were relevant.
    """
    if not returned_uuids:
        return 1.0 if not relevant_uuids else 0.0

    #dont want to doublecount
    relevant_set = set(relevant_uuids)
    hits = sum(1 for uuid in returned_uuids if uuid in relevant_set)
    return hits / len(returned_uuids)


def score_eval_set(use_reranking: bool = False):
    """Scores evl set from file in EVAL_SET_PATH"""
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    all_tickers = list({
        article["ticker"]
        for entry in eval_set
        for article in entry.get("candidate_articles", [])
    })
    articles, _ = ingestion(all_tickers)

    scores = []
    for entry in eval_set:
        query = entry["query"]
        relevant_uuids = entry["relevant_uuids"]

        tickers_seen = {a["ticker"] for a in entry.get("candidate_articles", [])}
        ticker = tickers_seen.pop() if len(tickers_seen) == 1 else None

        results = hybrid_search(query, articles, ticker=ticker, top_k=5, use_reranking=use_reranking)
        returned_uuids = [a["uuid"] for a in results]

        score = precision_at_k(returned_uuids, relevant_uuids)
        scores.append(score)
        print(f"{query[:60]:60s} precision@5: {score:.2f}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nAverage precision@5 across {len(scores)} queries: {avg:.3f}")
    return avg


if __name__ == "__main__":
    print("=== Without reranking ===")
    score_eval_set(use_reranking=False)

    print("\n=== With reranking ===")
    score_eval_set(use_reranking=True)