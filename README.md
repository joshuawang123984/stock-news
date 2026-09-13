# stock-news

A local RAG (Retrieval-Augmented Generation) tool for tracking news on
stocks I'm actually invested in. The idea is to pull in news for my
tickers, store it so I can search it semantically, and eventually have a local LLM summarize new information on a stock instead of me manually scrolling through headlines every day.

The generation component uses a local/open-source LLM so inference can be
performed without relying on a paid hosted LLM API.

## planned pipeline

1. **Ingestion** - pull news articles for a configured list of tickers
   from a news API and sentiment endpoint (Marketaux)
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

## why local LLM 

The generation component runs locally using `llama.cpp` instead of relying on
a hosted LLM API.

A few reasons: cost (free), and it's a better
showcase of actually understanding the RAG stack (embedding, retrieval,
reranking, quantization) instead of just wrapping someone else's LLM
API call.

## Retrieval evaluation

Built a hand labeled eval set to measure retrieval quality rather than
assuming it works: one general query template ("What's the latest news
on {ticker}?") applied across 8 major tickers, plus 2 cross ticker
queries about analyst sentiment (10 query instances total which is a limitation of this 
initial eval set, noted below).

- Hybrid search (dense + sparse fusion) alone: precision@20 = 0.675
- With cross-encoder reranking added: precision@20 = 0.655

Reranking did not show a measurable improvement here, though given the
small number of distinct query patterns, this result should be treated
as preliminary rather than conclusive. A larger, more varied eval set
(more query types, not just more tickers) would be needed to draw a
confident general conclusion about reranking's value for this pipeline.

## Docker

Docker is used to provide a reproducible environment for the project.

Qdrant runs as a Docker container and acts as the vector database. The Python
application connects to it through `QdrantClient`.

The stock-news application is also packaged into a Docker image containing
the Python runtime, dependencies, and application code. This allows the same
image to be tested locally and deployed to AWS.

## Cloud infrastructure

The application is containerized with Docker and deployed to AWS.

The Docker image is pushed to Amazon ECR and executed as a Fargate task.

AWS is used for scheduled ingestion workloads, while S3 provides persistent
storage for raw and processed article data. Qdrant is used for persistent
vector storage.

## Monitoring and Benchmarking

The project will measure performance across the major stages of the RAG
pipeline, including:

- Ingestion latency
- Article processing time
- Embedding generation time
- Vector search latency
- BM25 search latency
- Cross-encoder reranking latency
- LLM inference latency
- Tokens per second
- Memory / VRAM usage

Different LLM quantization levels such as FP16, INT8, and INT4 will be
benchmarked to evaluate the tradeoffs between inference speed, resource
usage, and output quality.

## tech stack

- **Python 3.11** - application and RAG pipeline
- **Qdrant** - vector database, run locally via Docker
- **sentence-transformers** - embeddings + cross-encoder reranking
- **llama.cpp** local LLM inference
- **rank_bm25** - keyword/sparse search half of the hybrid search step
- **Docker** - application containerization and local infrastructure
- **AWS ECS/Fargate** - cloud container execution
- **Amazon ECR** - Docker image registry
- **Amazon S3** - persistent article storage
- **Marketaux** - financial news API

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
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF', filename='mistral-7b-instruct-v0.2.Q4_K_M.gguf', local_dir='./models')"

```