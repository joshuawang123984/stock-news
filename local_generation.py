from hybrid_search import hybrid_search
from resources import get_llm

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
    stats = {
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
    }

    return response["choices"][0]["text"].strip(), stats

def answer_query(query: str, articles: list[dict], ticker: str | None = None) -> str:
    results = hybrid_search(query, articles, ticker=ticker, top_k=5, use_reranking=False) 
    return generate_summary(query, results)