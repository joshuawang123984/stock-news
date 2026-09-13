from ingestion import ingestion
from local_generation import answer_query
from resources import get_llm
import psutil
import os

TICKER = "SRPT"
QUERIES = [
    "What's the latest news on SRPT?",
    "Why are analysts saying SRPT is bullish?",
    "Why are analysts saying SRPT is bearish?",
]

def get_memory_mb() -> float:
    """Returns the current process's resident memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print(f"Memory at script start: {get_memory_mb():.1f} MB")

articles, failed = ingestion([TICKER])
print(f"Fetched {len(articles)} articles.\n")

get_llm()  # force model load ONCE, explicitly, before any queries
print(f"Memory after model load: {get_memory_mb():.1f} MB")

for query in QUERIES:
    print(f"\nQuery: {query}")
    answer = answer_query(query, articles, ticker=TICKER)
    print(f"Memory after this generation: {get_memory_mb():.1f} MB")
    print(f"Answer: {answer}\n")