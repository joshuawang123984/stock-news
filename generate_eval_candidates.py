import json
from datetime import date
from ingestion import ingestion
from embed_and_store import embed_and_store
from hybrid_search import hybrid_search

OUTPUT_PATH = "eval_top_dawgs.json"

NOTES = "largest publicly traded companies in the US (s&p 500 top dawgs)"
 
MY_TICKERS = [
    "NVDA", "AAPL", "GOOG", "GOOGL", "MSFT", "AMZN", "AVGO", "TSLA"
]

GENERAL_QUERIES = [
    "What's the latest news on {ticker}?",
]

CROSS_TICKER_QUERIES = [
    "Which of my holdings do analysts say are bullish?",
    "Which of my holdings do analysts say are bearish?",
]

def build_query_list() -> list[tuple[str, str | None]]:
    """Expands GENERAL_QUERIES across each ticker in MY_TICKERS, 
    then appends the cross-ticker queries"""
    queries = []
    for ticker in MY_TICKERS:
        for template in GENERAL_QUERIES:
            queries.append((template.format(ticker=ticker), ticker))
    for query in CROSS_TICKER_QUERIES:
        queries.append((query, None))
    return queries
 
 
def generate_candidates(top_k : int):
    print(f"Fetching articles for {len(MY_TICKERS)} tickers...")
    articles, failed = ingestion(MY_TICKERS)
    if failed:
        print(f"Warning: these tickers failed to fetch: {failed}")
    print(f"Fetched {len(articles)} articles total.\n")

    print("Embedding and storing articles in Qdrant...")
    embed_and_store(articles)
    print("Done.\n")
  
    candidates = []
    for query, ticker in build_query_list():
        results = hybrid_search(query, articles, ticker=ticker, top_k=top_k)
 
        entry = {
            "query": query,
            "candidate_articles": [
                {"uuid": a["uuid"], "title": a["title"], "ticker": a["ticker"]}
                for a in results
            ],
            "relevant_uuids": [],  #fill in by hand in json
        }
        if not results:
            entry["note"] = "no candidate articles returned"
 
        candidates.append(entry)
        print(f"  {query} -> {len(results)} candidates")

    output = {
        "metadata": {
            "created_at": date.today().isoformat(),
            "tickers": MY_TICKERS,
            "general_query_templates": GENERAL_QUERIES,
            "cross_ticker_queries": CROSS_TICKER_QUERIES,
            "notes": NOTES,
        },
        "queries": candidates,
    }
 
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
 
    print(f"\nWrote {len(candidates)} queries to {OUTPUT_PATH}")
    print("Next: open the file and fill in relevant_uuids for each query.")
 
 
if __name__ == "__main__":
    generate_candidates(top_k=20)