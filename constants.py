from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "stock_news"

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")