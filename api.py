from fastapi import FastAPI
from ingestion import ingestion
from local_generation import answer_query

app = FastAPI()

@app.get("/ask")
def ask(ticker: str, query: str):
    """Fetches fresh articles for the given ticker and answers the query
    using them.

    Example: GET /ask?ticker=SRPT&query=What's the latest news on SRPT?
    """
    articles, failed = ingestion([ticker])
    if failed:
        return {"error": f"Failed to fetch data for ticker: {ticker}"}

    answer, stats = answer_query(query, articles, ticker=ticker)
    return {
        "ticker": ticker,
        "query": query,
        "answer": answer,
        "articles_used": len(articles),
        "stats": stats,
    }