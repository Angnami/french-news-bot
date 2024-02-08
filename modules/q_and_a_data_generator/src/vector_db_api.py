import os
#from typing import List, Dict
from qdrant_client import QdrantClient

QDRANT_API_URL = os.environ["QDRANT_API_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]


def get_qdrant_client()->QdrantClient:
    qdrant_client = QdrantClient(
        url=QDRANT_API_URL,
        api_key=QDRANT_API_KEY,    
        
    )
    return qdrant_client


def init_collection(qdrant_client:QdrantClient,
                    collection_name:str,
                    vector_size:int,
                    #schema:str=''
                    )->QdrantClient:
    from qdrant_client.http.api_client import UnexpectedResponse
    from qdrant_client.http.models import Distance, VectorParams
    
    try:
        qdrant_client.get_collection(
            collection_name=collection_name
        )
    except (UnexpectedResponse, ValueError):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
            #schema=schema
        )
    
    return qdrant_client


