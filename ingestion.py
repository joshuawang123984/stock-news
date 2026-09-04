import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MARKETAUX_API_KEY")


def _extract_sentiment(article: dict, ticker: str) -> float | None:
    """Find the sentiment score for a specific ticker within an article's
    tagged entities.

    Marketaux tags an article with every related entity it can match
    (e.g. an Apple article may list mulptiple tickers: AAPL, AAPL.MX, AAPL.BA). This function 
    picks out only the score for the exact ticker we're tracking.

    Args:
        article: A single article dict from Marketaux's response.
        ticker: The exact ticker symbol to match (e.g. "AAPL", not "AAPL.MX").

    Returns:
        The sentiment score for that ticker, or None if it isn't tagged
        (shouldn't normally happen, since we searched by this ticker).
    """

    for entity in article.get("entities", []):
        if entity.get("symbol") == ticker:
            return entity.get("sentiment_score")
    return None

def ingestion(tickers: list[str]) -> tuple[list[dict], list[str]]:
    """Fetch news articles for a list of stock tickers from the Marketaux API.

    Makes one API request per ticker. Failures (non-200 responses) are
    collected separately in a separate list

    Args:
        tickers: Stock ticker symbols to fetch news for, e.g. ["CAN", "META"].
        Expected to be uppercase, matching Marketaux's symbol format.

    Returns:
        A tuple of:
            - articles: A flat list of article dicts across all requested
              tickers, each with the shape:
                {
                    "uuid": str,              # Marketaux's unique article ID
                    "title": str,
                    "description": str,       # short summary
                    "snippet": str,            # truncated excerpt of body text
                    "url": str,
                    "published_at": str,      # ISO 8601 timestamp
                    "source": str,             # publisher domain
                    "ticker": str,             # the exact ticker searched for
                    "sentiment_score": float | None,
                }

              sentiment_score is in a range between -1.0 to 1.0, with 0 being
              neutral. This reflects the tone of the article's language
              about the ticker (e.g. "record earnings" scores positive,
              "settlement" or "overvalued" scores negative).

            - failed: Tickers whose request did not return HTTP 200.
    """

    articles = []
    failed = []

    for ticker in tickers:
        ticker_response = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params={
                "symbols": ticker,
                "filter_entities": "true",
                "language": "en",
                "api_token": API_KEY,
            },
        )

        if ticker_response.status_code == 200:
            ticker_data = ticker_response.json()

            for article in ticker_data.get("data", []):
                articles.append({
                    "uuid": article["uuid"],
                    "title": article["title"],
                    "description": article["description"],
                    "snippet": article["snippet"],
                    "url": article["url"],
                    "published_at": article["published_at"],
                    "source": article["source"],
                    "ticker": ticker,
                    "sentiment_score": _extract_sentiment(article, ticker),
                })

        else:
            failed.append(ticker)

    return articles, failed

# for testing
if __name__ == "__main__":
    articles, failed = ingestion(["CAN"])
    print(f"Fetched {len(articles)} articles, {len(failed)} tickers failed")
    for article in articles[:2]:
        print(article)