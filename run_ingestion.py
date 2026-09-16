from ingestion import ingestion
from embed_and_store import embed_and_store

TICKERS = ["SRPT"]

def main():
    articles, failed = ingestion(TICKERS)

    print(f"Fetched {len(articles)} articles")

    if articles:
        embed_and_store(articles)
        print(f"Stored {len(articles)} articles in Qdrant")

    if failed:
        print(f"Failed tickers: {failed}")

if __name__ == "__main__":
    main()