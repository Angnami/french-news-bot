import logging
from typing import Union
from datetime import datetime

from features_pipeline import initialize
from features_pipeline.flow import build as flow_builder


def build_flow(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
    timeframe: Union[int,str]='15m',): 
    
    """
    Construit un flux Bytewax pour le traitement en batch des articles de presse.
    Args:
        -env_file_path (str): Chemin du fichier des variables d'environnement.
        -logging_config_path (str): Chemin du fichier de configuration du logging.
        -model_cache_dir (str): Chemin du repertoire où le modèle sera caché.
        -timeframe: Union[int,str]: La période d'extraction des données.

    Returns:
        flow (prefect.Flow): Le flux Bytewax pour le traitement en batch des articles.
    """
    initialize(logging_config_path=logging_config_path,env_file_path=env_file_path)
    
    logger = logging.getLogger(__name__)
    
    timeframe_dtypes = {
        'int': 'hours',
        'str': 'minutes'
    }
    logger.info(msg=f'Extraction des articles de presse pour les {timeframe}.\
                    dernières {timeframe_dtypes[type(timeframe).__name__]} à .\
                    {datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.'
                    )
    
    flow = flow_builder(timeframe=timeframe,model_cache_dir=model_cache_dir)
    
    return flow