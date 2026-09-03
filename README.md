# stock-news

A local RAG (Retrieval-Augmented Generation) tool for tracking news on
stocks I'm actually invested in. The idea is to pull in news for my
tickers, store it so I can search it semantically, and eventually have a local LLM summarize new information on a stock instead of me manually scrolling through headlines every day.

Doing this entirely with a local/open-source LLM so it can run fully offline annd its free

## planned pipeline

1. **Ingestion** - pull news articles for a configured list of tickers
   from a news API and sentiment endpoint(probably NewsAPI or Alpha Vantage's News)
2. **Embedding + storage** - embed articles with `sentence-transformers`,
   store them in Qdrant along with metadata (ticker, timestamp, source)
3. **Hybrid search** - combine dense vector search with keyword/BM25
   search, since exact ticker/company name matches matter a lot here
   and pure semantic similarity can miss them
4. **Reranking** - cross-encoder reranker on top of hybrid search
   results before anything gets sent to the LLM
5. **Local generation** - run a small open model locally via
   `llama.cpp` to summarize retrieved articles, with citations back to
   the source 
6. **Quantization benchmarking** (maybe) - measure VRAM usage and
   quality tradeoffs running the model at FP16 vs INT8 vs INT4

## why local, not an API

A few reasons: cost (free), and it's a better
showcase of actually understanding the RAG stack (embedding, retrieval,
reranking, quantization) instead of just wrapping someone else's LLM
API call.

## tech stack

- **Python 3.11** python
- **Qdrant** - vector database, run locally via Docker
- **sentence-transformers** - embeddings + cross-encoder reranking
- **llama.cpp** local LLM inference
- **rank_bm25** - keyword/sparse search half of the hybrid search step

## running it

You'll need Docker installed and running.

Start Qdrant:
```
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

In a separate terminal, activate the venv and run the connectivity check:
```
source .venv/bin/activate
python main.py
```

## setup from scratch

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```