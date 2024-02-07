from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import requests
import json
import datetime
import os
from config import set_environment

from src.logger import get_console_logger
from src.paths import DATA_DIR

logger = get_console_logger()

set_environment()

list_of_categories = ['business', 'entertainment','environment', 'health', 'politics', 'science','sports', 'technology', 'top', 'world']

try:
    apikey = os.environ['NEWS_DATA_API_KEY']
except KeyError:
    
    logger.error(
        msg="Veuillez définir la variable d'environnement NEWS_DATA_API_KEY pour acceder à l'API NEWSDATA!"
    )
    raise

@dataclass
class News:
    title: str
    category: str
    content: str
    pubDate: datetime.datetime
    link:str
    description:str
    article_id:str

def fetch_batch_of_news(category: str, from_date:datetime, to_date:datetime, page_token: Optional[str]=None) -> Tuple[News, str]:
    """
    Cette fonction permet de recupérer un lot d'informations concernant un sujet d'actualité sur une période donnée.
    Elle prend deux arguments. Les actualités sont filtrées uniquement sur la France.
    Args:
        - category: str : est une chaine de caractères comprenant un sujet d'actualité.Exemple: 'politics'.
        - from_date(datetime):date de début de la période de récupération.
        - to_date(datetime):date de fin de la période de récupération.
        - page_token: Cet argument optionnel renseigne sur la présence d'une page suivante pour le document. La valeur
        par défaut est None.
    Returns:
        - Une liste des nouvelles correspondant aux arguments category, from_date et to_date. 
        - Une chaine de caractère correspondant à la page suivante si elle existe.

    """
    # Définir les paramètres de la requête
    params = {
        'apikey': apikey,
        'from_date': from_date,
        'to_date':to_date,
        'country': 'fr',
        'full_content': 1,
        'category': category,
        'language': 'fr'
    }
    if page_token is not None:
        params['page_token'] = page_token
    
    ARCHIVE_URL = "https://newsdata.io/api/1/archive"

    # Interroger l'API 
    res = requests.get(url=ARCHIVE_URL, params=params)

    # Analyse du résultat de la requête
    list_of_news = []
    next_page_token = None
    if res.status_code == 200:
        news_json = res.json()
        # Extraire la référence de la page suivante si elle existe
        next_page_token = news_json.get("nextPage", None)
        # Construire la liste des nouvelles
        for news in news_json['results']:
            list_of_news.append(
                News(
                    title=news['title'],
                    category=news['category'][0],
                    content=news['content'],
                    pubDate=news['pubDate'],
                    link=news['link'],
                    description=news['description'],
                    article_id =news['article_id']
                )
            )
    else:
        logger.error(
            msg=f"La reqête a échoué avec le code error: {res.status_code}"
        )

    return list_of_news, next_page_token



def save_news_to_json(news_list:List[News], filename:Path):
    """
    Cette fonction permet de sauvegarder les informations extraites grâce à la fonction fetch_batch_of_news au format json. 
    Elle prend deux arguments.
    Args:
        - news_list: List[News]: liste des informations à sauvegarder.
        - filename: Path: le chemin d'enregistrement.
    """
    news_data = [
        {
         'title': news.title,
         'category': news.category,
         'pubDate': news.pubDate,
         'content': news.content,
         'link'   :news.link,
         'description':news.description,
         'article_id':news.article_id,
        }
        for news in news_list
    ]

    with open(file=filename, mode='w') as json_file:
        json.dump(obj=news_data, fp=json_file, indent=4)


def download_historical_data(from_date:datetime, to_date:datetime)-> Path:
    """
    Cette fonction permet de télécharger les données historiques pour sur une période donnée pour toutes les catégories 
    choisies.
    Args:
        - from_date(datetime):cet argument correspond à la date de début de la période de récupération.
        - to_date(datetime):cet argument correspond à la date de fin de la période de récupération.
    Returns:
        - Le chemin d'enregistrement des données.
        
    """

    # Extraction des données historiques pour toutes les catégories
    global_list_of_news = []
    logger.info(msg=f"Téléchargement des données historiques de {from_date} à {to_date}.")
    for cat in list_of_categories:
        list_of_news, next_page_token = fetch_batch_of_news(category=cat, from_date=from_date, to_date=to_date)
        logger.info(msg=f"Nouvelles recupérées pour la catégorie {cat}: {len(list_of_news)}")
        logger.debug(f"Référence de la page suivante: {next_page_token}")
        while next_page_token is not None and len(list_of_news)<50:
            batch_of_news, next_page_token = fetch_batch_of_news(category=cat,
                                                                 from_date=from_date, 
                                                                 to_date=to_date,
                                                                 page_token=next_page_token)
            list_of_news += batch_of_news
            logger.info(msg=f"Nouvelles recupérées: {len(list_of_news)}")
            logger.debug(f"Référence de la page suivante: {next_page_token}")
            logger.debug(msg=f"La dernière date du lot d'informations dans la catégorie {cat} est: {batch_of_news[-1].pubDate}")
        global_list_of_news += list_of_news

    # Sauvegarder les données dans un fichier
    path_to_file = (
        DATA_DIR/ f'news_{from_date.strftime("%Y-%m-%d")}_{to_date.strftime("%Y-%m-%d")}.json'
    )
    save_news_to_json(news_list=global_list_of_news, filename=path_to_file)
    logger.info(msg=f'Les données des nouvelles sont enregistrées dans: {path_to_file}')

    # Renvoyer le chemin des données enregistrées
    return path_to_file
