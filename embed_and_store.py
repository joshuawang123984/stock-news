from qdrant_client.models import VectorParams, Distance, PointStruct
from constants import model, client, COLLECTION_NAME


def embed(articles: list[dict]) -> list[PointStruct]:
    """Embed each article's title+description into a vector representation.

    Args:
        articles: Article dicts as returned by ingestion(), each
            containing at least "uuid", "title", and "description".

    Returns:
        A list of PointStruct objects ready to nsert into Qdrant with
        each point's ID set to the article's own uuid and its payload
        set to the full article dict.
    """

    texts = [f"{a['title']}. {a['description']}" for a in articles]
    vectors = model.encode(texts)

    points = [
        PointStruct(id=a["uuid"], vector=v.tolist(), payload=a)
        for a, v in zip(articles, vectors)
    ]

    return points

def store(points: list[PointStruct]) -> None:
    """Insert embedded articles into Qdrant, creating the collection if needed.

    Uses each point's existing ID (the article's own uuid) so re-running ingestion 
    on the same articles overwrites rather than duplicates.

    Args:
        points: PointStruct objects as returned by embed().
    """
    
    if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def embed_and_store(articles: list[dict]) -> None:
    """Embed a batch of articles and store them in Qdrant.

    Calls embed() then store() — see each for details.

    Args:
        articles: Article dicts as returned by ingestion().
    """

    points = embed(articles)
    store(points)