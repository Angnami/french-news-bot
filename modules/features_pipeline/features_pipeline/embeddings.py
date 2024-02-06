import logging
import traceback
from typing import Optional, Union
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

from features_pipeline import constants
from features_pipeline.base import SingletonMeta


logger = logging.getLogger(__name__)


class EmbeddingModelSingleton(metaclass=SingletonMeta):
    """
    
    Une classe Singleton qui fornit un modèle pré-entrainé permettant de générer l'embedding du texte donné en input.
    Args:
        -model_id(str):l'identifiant du modèle pré-entrainé à utiliser.
        -max_input_length(int):la taille maximale du texte à tokenizer.
        -device(str):le périphérique à utiliser pour calculer les embeddings('cpu' ou 'cuda).
        -cache_dir(Optional[path]):le repertoire de cache des fichiers du modèle pré-entrainé. Si cet argument n'est pas renseigné
        le repertoire par défaut sera utilisé.
    Attributes:
        -max_input_length(int):la taille maximale du texte à tokenizer.
        -tokenizer(AutoTokenizer):le tokenizer à utiliser pour tokéniser le texte.
    """
    
    def __init__(self,
                 model_id:str=constants.EMBEDDING_MODEL_ID,
                 max_input_length:int=constants.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
                 cache_dir:Optional[Path]=None,
                 device:str=constants.EMBEDDING_MODEL_DEVICE):
        """
        Inintialise l'instance SingletonMeta.
        Args:
            -model_id(str):l'identifiant du modèle pré-entrainé à utiliser.
            -max_input_length(int):la taille maximale du texte à tokenizer.
            -device(str):le périphérique à utiliser pour calculer les embeddings('cpu' ou 'cuda).
            -cache_dir(Optional[path]):le repertoire de cache des fichiers du modèle pré-entrainé. Si cet argument n'est pas renseigné
            le repertoire par défaut sera utilisé.
        """
        self._model_id = model_id
        self._max_input_length=max_input_length
        self._cache_dir=cache_dir
        self._device=device
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model=AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_id,
            cache_dir=str(cache_dir) if cache_dir else None
            ).to(self._device)
        self._model.eval()
        
    @property
    def tokenizer(self)->AutoTokenizer:
        """
        Renvoie le tokenizer à utilser pour tokéniser le texte.
        
        Returns:
            -AutoTokenozer:le tokenizer à utilser pour tokéniser le texte.
        """
        
        return self._tokenizer
    
    @property
    def max_input_length(self)->int:
        """
        Retourne la taille maximale du texte à tokenizer.
        Returns:
            -max_input_length(int):la taille maximale du texte à tokenizer.
        """   
        return self._max_input_length
    
    def __call__(self,input_text:str,to_list:bool=True)-> Union[np.ndarray,list]:
        """
        Produit les embeddings du texte fourni en input en utilisant le modèle pré-entrainé.
        Args:
            -input_text(str):le texte pour lequel on doit calculer les embeddings.
            -to_list(bool):pour indiquer s'il faut retourner les embeddings sous forme de liste ou array.
        Returns:
            -Union[np.ndarray,List]:les embeddings générés pour le texte.
        
        """
        try:
            tokenized_text = self._tokenizer(
                input_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._max_input_length,
            ).to(self._device)
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(f"Erreur de la tokénisation du texte: {input_text}.")

            return [] if to_list else np.array([])

        try:
            result = self._model(**tokenized_text)
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(
                f"Erreur de la génération des embeddings pour le model_id: {self._model_id} et le texte: {input_text}."
            )

            return [] if to_list else np.array([])

        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        if to_list:
            embeddings = embeddings.flatten().tolist()

        return embeddings