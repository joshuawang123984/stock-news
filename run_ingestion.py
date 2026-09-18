from ingestion import ingestion
from embed_and_store import embed_and_store
import boto3

ssm = boto3.client("ssm", region_name="us-east-1")

def get_tickers():
    response = ssm.get_parameter(
        Name="/stock-news/TICKERS"
    )

    return response["Parameter"]["Value"].split(",")


def main():
    tickers = get_tickers()
    print(f"Ingesting tickers: {tickers}")

    articles, failed = ingestion(tickers)

    print(f"Fetched {len(articles)} articles")

    if articles:
        embed_and_store(articles)
        print(f"Stored {len(articles)} articles in Qdrant")

    if failed:
        print(f"Failed tickers: {failed}")

if __name__ == "__main__":
    main()