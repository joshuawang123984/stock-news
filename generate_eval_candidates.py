import json
from ingestion import ingestion
from embed_and_store import embed_and_store
from hybrid_search import hybrid_search

MY_TICKERS = [
    "SRPT", "CAN", "META", "MGNX", "NKTX", "PLX", "IVVD", "LCTX",
]

GENERAL_QUERIES = [
    "What's the latest news on {ticker}?",
    "Any recent clinical trial or FDA updates for {ticker}?",
]

CROSS_TICKER_QUERIES = [
    "Which of my holdings had FDA-related news this week?",
    "Which of my holdings do analysts say are bearish?",
]

def build_query_list() -> list[str]:
    """Expands GENERAL_QUERIES across each ticker in MY_TICKERS, 
    then appends the cross-ticker queries"""
    queries = []
    for ticker in MY_TICKERS:
        for template in GENERAL_QUERIES:
            queries.append(template.format(ticker=ticker))
    queries.extend(CROSS_TICKER_QUERIES)
    return queries
 
 
def generate_candidates():
    print(f"Fetching articles for {len(MY_TICKERS)} tickers...")
    articles, failed = ingestion(MY_TICKERS)
    if failed:
        print(f"Warning: these tickers failed to fetch: {failed}")
    print(f"Fetched {len(articles)} articles total.\n")

    print("Embedding and storing articles in Qdrant...")
    embed_and_store(articles)
    print("Done.\n")
 
    queries = build_query_list()
    print(f"Running {len(queries)} queries through hybrid_search...\n")
 
    candidates = []
    for query in queries:
        results = hybrid_search(query, articles, top_k=5)
 
        entry = {
            "query": query,
            "candidate_articles": [
                {"uuid": a["uuid"], "title": a["title"], "ticker": a["ticker"]}
                for a in results
            ],
            "relevant_uuids": [],  # <-- fill in by hand
        }
        if not results:
            entry["note"] = "no candidate articles returned — confirm this is correct before treating as resolved"
 
        candidates.append(entry)
        print(f"  {query} -> {len(results)} candidates")
 
    with open("eval_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
 
    print(f"\nWrote {len(candidates)} queries to eval_candidates.json")
    print("Next: open the file and fill in relevant_uuids for each query.")
 
 
if __name__ == "__main__":
    generate_candidates()