import logging
import os
from typing import  List, Union, Dict

import requests
from bytewax.inputs import DynamicInput, StatelessSource

from features_pipeline.utils import list_of_categories
from features_pipeline.utils import complete_text_for_timeframe
from config import set_environment

set_environment()

logger = logging.getLogger()


class NewsDataBatchInput(DynamicInput):
    """
    Une classe Input qui réçoit un lot de nouvelles à partir de l'API newsdata.
    Args:
        -timeframe(Union[int,str]):la période pour laquelle le lot d'informations sera récupérée.
    """
    
    def __init__(self,timeframe:Union[int,str]):
        self._timeframe = timeframe
    
    def build(self, worker_index:int, worker_count:int):
        """
        Répartit les différents rubriques d'informations entre différents workers sur la base du nombre total worker_count.
        """
        category = list_of_categories[worker_index]
        
        logger.info(msg=f'worker_index:{worker_index}, articles de la rubrique {category} {complete_text_for_timeframe(timeframe=self._timeframe)}.'
                    )
        
        return NewsDataBatchSource(
            timeframe=self._timeframe,
            category = category
        )
    
class NewsDataBatchSource(StatelessSource):
    """
    Source batch pour récupérer les données à partir de m'api newsdata.
    Args:
        -category(str):la rubrique d'information.
        -timeframe(Union[int,str]):la période de récupération des données.
    """
    def __init__(self,category:str,timeframe:Union[int,str]):
        self._newsdata_client = build_newsdata_client(
            category = category,
            timeframe=timeframe,
        )
        
    def next(self):
        """
        Récupère le lot suivant d'articles de presse.
        Returns:
            -List[dict]:une liste d'articles de presse.
        """
        news = self._newsdata_client.list()
        
        if news is None or len(news)==0:
            raise StopIteration()

        return news

    def close(self):
        """
        Ferme la source batch.
        """
        pass 
    


def build_newsdata_client(timeframe:Union[int,str], 
                          category:str,
                          params:dict={'apikey': None,
                                       'country': 'fr',
                                       'full_content': 1,
                                       'language': 'fr',
                                       'size':50,
                                       },
                          worker_count:int=len(list_of_categories),
                          )->"NewsDataBatchClient":
    """
    Construit un object NewsDataBatchClient avec les paramètres spécifiés.
    Args:
        -timeframe(Union[int,str]):la période récupération des données.
        -params(dict):les paramètres de la requête tels api_key, country, etc.
        -worker_count:le nombre total de workers correspondant au nombre des rubriques à récupérer.
    Returns:
        -NewsDataBatchClient:l'objet NewsDataBatchClient.
    Raises:
        -Génère une erreur si la variable api_key n'est pas fournie.
    
    """
    if params["apikey"] is None:
        try:
            params["apikey"] = os.environ["NEWS_DATA_API_KEY"]
        except KeyError:
            raise (
                "La clef de l'api newsdata doit être définie manuellement ou à travers une variable environnementale."
            )
    params.update([('timeframe',timeframe),('category',category)])

    return NewsDataBatchClient(
        params=params
    )     
    


class NewsDataBatchClient:
    """
    Client de l'api newsdata qui permet de récupérer des données.
    Args:
        -params(dict):un dictionnaire des paramètres de la requête. 
    """
    NEWS_URL = "https://newsdata.io/api/1/news"
    
    def __init__(self,params):
        """
        Initialise une instance de NewsDataBatchClient.
        Args:
            -params(dict):un dictionnaire des paramètres de la requête
        """
        self._params = params
        
        self._page_token = None
        self._first_request = True
        
        
    @property
    def try_request(self)-> bool:
        """
        Une proprieté indiquant si une requête doit être tentée ou pas.
        Returns:
            -bool:True si une requête doit être tentée et False sinon.
        
        """
        return self._first_request or self._page_token is not None
    
    def list(self)->List[Dict]:
        """
        Une fonction qui permet de récupérer un lot d'articles à partir de l'api newsdata.
        Returns:
            - List[Dict]:une liste d'informations.
        
        """
        if not self.try_request:
            return None
        
        self._first_request=False
        
        if self._page_token is not None:
            self._params['page_token'] = self._page_token
        # Interroger l'API 
        res = requests.get(url=self.NEWS_URL, params=self._params)
        
        # Analyse du résultat de la requête
        self._page_token = None
        if res.status_code == 200:
            news_json = res.json()
            # Extraire la référence de la page suivante si elle existe
            next_page_token = news_json.get("nextPage", None)
            # # Construire la liste des nouvelles
            # for news in news_json['results']:
            #     list_of_news.append(
            #         dict(
            #             title=news['title'],
            #             category=news['category'][0],
            #             content=news['content'],
            #             pubDate=news['pubDate'],
            #             link=news['link'],
            #             description=news['description'],
            #         )
            #     )
        else:
            logger.error(
                msg=f"La reqête a échoué avec le code error: {res.status_code}"
            )
   
        self._page_token = next_page_token
        
        return news_json['results']
        
