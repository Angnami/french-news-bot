import os
from typing import Optional

from bytewax.outputs import DynamicOutput, StatelessSink
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from features_pipeline import constants
from features_pipeline.models import Document

class QdrantVectorOutput(DynamicOutput):
    """
    Représente une classe de vecteur Qdrant de sortie.
    Args:
        -vector_size (int): La taille du vector.
        -collection_name (str, optional): Le nom nde la collection. constants.VECTOR_DB_OUTPUT_COLLECTION_NAME par défaut.
        -client (Optional[QdrantClient], optional): Le client Qdrant. None par défaut.
    """
    def __init__(
        self,
        vector_size: int,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        client: Optional[QdrantClient] = None,):
        
        self._collection_name = collection_name
        self._vector_size = vector_size

        if client:
            self.client = client
        else:
            self.client = build_qdrant_client()

        try:
            self.client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )
    
    def build(self, worker_index:int, worker_count:int):
        """Builds a QdrantVectorSink object.

        Args:
            worker_index (int): L'indice du worker.
            worker_count (int): Le nombre total des workers.

        Returns:
            QdrantVectorSink: Un objet QdrantVectorSink.
        """

        return QdrantVectorSink(self.client, self._collection_name)



def build_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Construit un un objet QdrantClient avec l'URL et la clé de l'API.

    Args:
        -url (Optional[str]): L'URL du serveur Qdrant. Si elle n'est pas fournie, 
        elle sera lue à partir de la variable d'environnement QDRANT_URL.
        -api_key (Optional[str]): La clé de l'API à utiliser pour l'authentification. Si elle n'est pas fournie,
        elle sera lue à partir de la variable d'environnement QDRANT_API_KEY.

    Raises:
        KeyError: Si les variables d'environnement QDRANT_URL et/ou QDRANT_API_KEY ne sont pas données en arguments.

    Returns:
        QdrantClient: Un objet QdrantClient connecté au serveur Qdrant spécifié.
    """

    if url is None:
        try:
            url = os.environ["QDRANT_URL"]
        except KeyError:
            raise KeyError(
                "QDRANT_URL doit être définie en variable d'environnement ou manuellement passée en argument."
            )

    if api_key is None:
        try:
            api_key = os.environ["QDRANT_API_KEY"]
        except KeyError:
            raise KeyError(
                "QDRANT_API_KEYdoit être définie en variable d'environnement ou manuellement passée en argument."
            )

    client = QdrantClient(url=url, api_key=api_key)

    return client


class QdrantVectorSink(StatelessSink):
    """
    Un sink qui écrit des embeddings d'undocument dans une collection Qdrant.

    Args:
        client (QdrantClient): Le client Qdrant à utiliser pour l'écriture.
        collection_name (str, optional): Le nom de la collection dans laquelle on écrit.
         constants.VECTOR_DB_OUTPUT_COLLECTION_NAME par défaut.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
    ):
        self._client = client
        self._collection_name = collection_name

    def write(self, document: Document):
        ids, payloads = document.to_payloads()
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document.embeddings, payloads)
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)