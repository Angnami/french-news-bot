import logging
import os
from typing import Optional

import qdrant_client

logger = logging.getLogger(__name__)



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

    client = qdrant_client.QdrantClient(url=url, api_key=api_key)

    return client
