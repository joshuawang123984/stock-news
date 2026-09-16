# stock-news

A local RAG (Retrieval-Augmented Generation) tool for tracking news on
stocks I'm actually invested in. The idea is to pull in news for my
tickers, store it so I can search it semantically, and eventually have a local LLM summarize new information on a stock instead of me manually scrolling through headlines every day.

The generation component uses a local/open-source LLM so inference can be
performed without relying on a paid hosted LLM API.

## pipeline

1. **Ingestion** - pull news articles for a configured list of tickers
   from Marketaux
2. **Embedding + storage** - embed articles with `sentence-transformers`,
   store them in Qdrant along with metadata (ticker, timestamp, source)
3. **Hybrid search** - combine dense vector search with keyword/BM25
   search, since exact ticker/company name matches matter a lot here
   and pure semantic similarity can miss them
4. **Reranking** - cross-encoder reranker on top of hybrid search
   results before anything gets sent to the LLM (off by default since 
   small increase in precision wasn't enough to justify using)
5. **Local generation** - run a small open model locally via
   `llama.cpp` to summarize retrieved articles, with citations back to
   the source 
6. **Quantization benchmarking** (maybe) - measured memory and speed tradeoffs
   between Q4_K_M and Q8_0

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

LLM inference is benchmarked across GGUF quantization levels (Q4_K_M,
Q8_0), measuring:

- Load time
- Prompt processing speed (tokens/sec)
- Generation speed (tokens/sec)
- Memory usage

Output is evaluated at each quantization level using a subset of
the retrieval eval set's queries, and checking for proper citations 
and grounding as the model size gets smaller from Q8_0 -> Q4_K_M.

Retrieval quality (hybrid search, with and without reranking) is
separately measured against a hand labeled eval set

## tech stack

- **Python 3.11** - application and RAG pipeline
- **Qdrant** - vector database, run locally via Docker
- **sentence-transformers** - embeddings + cross-encoder reranking
- **llama.cpp** local LLM inference
- **rank_bm25** - keyword/sparse search half of the hybrid search step
- **FastAPI + Uvicorn** - API layer
- **Docker** - application containerization and local infrastructure
- **AWS ECS/Fargate** - cloud container execution
- **Amazon ECR** - Docker image registry
- **Amazon EventBridge** - scheduled ingestion trigger
- **Amazon S3** - persistent article storage
- **Marketaux** - financial news API

## running it

You'll need Docker installed and running. Depending on the model you select (default is 4.4GB), 
you will need to increase memory allocated to Docker (Settings → Resources → Memory)

1. Download the model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main
2. Start everything via: docker-compose up --build
3. Once you see "Uvicorn running on http://0.0.0.0:8000" in the logs,
   visit `http://localhost:8000/docs` for interactive API documentation,
   or query directly:

## setup from scratch

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF', filename='mistral-7b-instruct-v0.2.Q4_K_M.gguf', local_dir='./models')"

```