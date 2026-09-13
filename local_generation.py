import platform
from llama_cpp import Llama
from hybrid_search import hybrid_search

def get_gpu_layers() -> int:
    """Returns the number of layers to offload to gpu, or 0 for cpu.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return -1  
        if platform.system() == "Darwin" and platform.processor() == "arm":
            return -1  
    except ImportError:
        pass
    return 0  # no gp support detected, or torch not installed —> cpu 


# caches per process (not lifetime)
_llm = None

def get_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_ctx=4096,
            n_gpu_layers=get_gpu_layers(),
        )
    return _llm

def generate_summary(query: str, articles: list[dict]) -> str:
    """Generates a grounded answer to the query using only the provided
    articles as source material

    Args:
        query: The user's question.
        articles: Retrieved articles (from hybrid_search) to base the
            answer on 

    Returns:
        A generated response from the model (llama).
    """
    context = "\n\n".join(
        f"[Article {i+1}] {a['title']}\n{a['description']}"
        for i, a in enumerate(articles)
    )

    prompt = f"""You are responsible for keeping track of my investments. Think of yourself as my financial news assistant. 
                Answer the question using ONLY the articles below. Cite the article number(s) for each claim you make. If the articles 
                don't contain enough information to answer, say so explicitly — do not use outside knowledge.   

    Articles:
    {context}

    Question: {query}

    Answer:"""

    response = get_llm()(prompt, max_tokens=300, temperature=0.2, stop=["Question:"])
    return response["choices"][0]["text"].strip()

def answer_query(query: str, articles: list[dict], ticker: str | None = None) -> str:
    results = hybrid_search(query, articles, ticker=ticker, top_k=5, use_reranking=False) 
    return generate_summary(query, results)