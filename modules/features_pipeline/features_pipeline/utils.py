from typing import List, Union
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

def read_requirements(file_path:str)->List[str]:
    """
    Lit un fichier contenant une liste des 'requirements' et retoure une liste contenant des textes.
    Args:
        -file_path(str):le chemin du fichier contenant les requirements.
    Returns:
        - List(str):une liste des requirements sous forme de texte
    
    """
    with open(file=file_path,mode='r') as file:
        requirements = [line.strip() for line in file if line.strip()]
    
    return requirements


#La liste des rubriques d'informations retenues.
list_of_categories = ['business', 'entertainment','environment', 'health', 'politics', 'science','sports', 'technology', 'top', 'world']


def complete_text_for_timeframe(timeframe:Union[int,str])-> str:
    """
    Cette fonction permet d'afficher un texte en fonction de la valeur de l'argument timeframe.
    Elle est utilisée pour compléter le texte de logging lors de la récupération des
    articles de presse à partir de l'API NewsData.
    Args:
        timeframe(Union[int,str]):La période pour laquelle les articles de presse seront récupérés.
    Returns:
        str:Le texte à afficher en fonction de la valeur de timeframe.
    """
    timeframe_dtypes = {
        'int': 'heures',
        'str': 'minutes'
    }
    
    if type(timeframe).__name__ =='int':
        if timeframe > 1:
            return f'pour les {timeframe} dernières {timeframe_dtypes[type(timeframe).__name__]} à {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.'
        elif timeframe == 1:
            return f'pour la dernière heure à {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.'
        else:
            logger.error(msg="L'argument timeframe doit être un entier strictement positif. Veuillez saisir un nombre correct!")
    else:
        if int(timeframe[:-1]) > 1:
            return f'pour les {timeframe} dernières {timeframe_dtypes[type(timeframe).__name__]} à {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.'
        elif int(timeframe[:-1]) == 1:
            return f'pour la dernière minute à {datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.'
        else:
            logger.error(msg="L'argument timeframe doit être un entier strictement positif. Veuillez saisir un nombre correct!")
         
       
    