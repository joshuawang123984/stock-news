from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from llama_cpp import Llama

import platform
import torch

COLLECTION_NAME = "stock_news"
MODEL_PATH_1 = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH_2 = "./models/mistral-7b-instruct-v0.2.Q8_0.gguf"

_model = None
_client = None
_reranker = None
_llm = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
    return _client

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def get_gpu_layers() -> int:
    """Returns the number of layers to offload to gpu, or 0 for cpu.
    """
    try:
        if torch.cuda.is_available():
            return -1  
        if platform.system() == "Darwin" and platform.processor() == "arm":
            return -1  
    except ImportError:
        pass
    return 0  # no gpu support detected, or torch not installed —> cpu 

def get_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH_2,
            n_ctx=4096,
            n_gpu_layers=get_gpu_layers(),
        )
    return _llm
