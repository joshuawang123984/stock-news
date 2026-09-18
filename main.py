from local_generation import answer_query
from resources import get_llm
from datetime import datetime
import psutil
import os
import json
import time

TICKER = "SRPT"
QUERIES = [
    "What's the latest news on SRPT?",
    "Why are analysts saying SRPT is bullish?",
    "Why are analysts saying SRPT is bearish?",
]

MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q8_0.gguf"
OUTPUT_PATH = "benchmark_results_2.json"

def get_memory_mb() -> float:
    """Returns the current process's resident memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark():
    results = {
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat(),
        "memory_at_start_mb": get_memory_mb(),
        "queries": [],
    }

    get_llm() 
    results["memory_after_model_load_mb"] = get_memory_mb()

    for query in QUERIES:
        start = time.perf_counter()
        answer, stats = answer_query(query, ticker=TICKER)
        elapsed = time.perf_counter() - start

        results["queries"].append({
            "query": query,
            "answer": answer,
            "memory_after_mb": get_memory_mb(),
            "elapsed_seconds": elapsed,
            "tokens_per_second": stats["completion_tokens"] / elapsed if elapsed > 0 else None,
            **stats,
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_benchmark()