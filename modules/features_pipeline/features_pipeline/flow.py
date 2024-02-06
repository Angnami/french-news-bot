from pathlib import Path
from typing import Optional, List, Union

from bytewax.inputs import Input
from bytewax.outputs import Output
from bytewax.dataflow import Dataflow
from pydantic import parse_obj_as
from qdrant_client import QdrantClient

from features_pipeline.newsdata_batch import NewsDataBatchInput
from features_pipeline.embeddings import EmbeddingModelSingleton
from features_pipeline.models import NewsArticle
from features_pipeline.qdrant import QdrantVectorOutput


def build(
    timeframe:Union[int,str]="15m",
    model_cache_dir: Optional[Path] = None,) -> Dataflow:
    """
    Construit un pipeline dataflow pour le traitement des articles de presse.

    Args:
        -timeframe(Union[int,str]):la période récupération des données.
        -model_cache_dir (Optional[Path]): Le repertoire de cache du modèle d'embedding.

    Returns:
        Dataflow: Le pipeline dataflow pour le traitement des articles de presse.
    """
    model = EmbeddingModelSingleton(cache_dir=model_cache_dir)
    
    flow = Dataflow()
    flow.input(
        "input",
        _build_input(timeframe),
        )
    flow.flat_map(lambda messages: parse_obj_as(List[NewsArticle], messages))
    flow.map(lambda article: article.to_document())
    flow.map(lambda document: document.compute_chunks(model))
    flow.map(lambda document: document.compute_embeddings(model))
    flow.output("output", _build_output(model))
    
    
def _build_input(timeframe:Union[int,str]="15m",) -> Input:

        return NewsDataBatchInput(timeframe=timeframe)


def _build_output(model: EmbeddingModelSingleton, in_memory: bool = False) -> Output:
    if in_memory:
        return QdrantVectorOutput(
            vector_size=model.max_input_length,
            client=QdrantClient(":memory:"),
        )
    else:
        return QdrantVectorOutput(
            vector_size=model.max_input_length,
        )