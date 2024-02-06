import logging

from fire import Fire

from streaming_pipeline import constants, initialize
from streaming_pipeline.embeddings import EmbeddingModelSingleton
from streaming_pipeline.qdrant import build_qdrant_client


logger = logging.getLogger(__name__)

def search(query_string: str):
    """
    Recherche les points les plus proches du texte de recherche dans la base de données vectorielle.

    Args:
        query_string (str): Le texte pour lequel on fait la recherche.

    Returns:
        None
    """
    initialize()
    
    client = build_qdrant_client()
    model = EmbeddingModelSingleton()
    query_embedding = model(query_string, to_list=True)
    
    hits = client.search(
        collection_name=constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=5,  # Retourne les 5 points les plus proches
        )
    
    for hit in hits:
        logger.info(hit)
    
if __name__ == "__main__":
    Fire(search)