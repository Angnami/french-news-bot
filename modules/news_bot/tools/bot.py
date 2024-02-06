from pathlib import Path
from typing import List, Tuple
import logging

import fire
from beam import App, Image, Volume, VolumeType, Runtime


logger = logging.getLogger(__name__)


# === Beam Apps ===
news_bot = App(
    name="news_bot",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="T4",
        image=Image(python_version="python3.11", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(
            path="./model_cache", name="model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)


def load_bot(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_cache",
    embedding_model_device: str = "cuda:0",
    #debug: bool = False,
):
    """
    Charge le modèle de l'assistant d'actualités en mode production ou development sur la base du paramètre `debug`.

    En mode DEV, le modèle d'embedding s'exécute sur CPU et le LLM fine-tuné est simulé.
    Sinon, le modèle d'embedding s'exécute sur GPU et le LLM fine-tuné est utilisé.

    Args:
        env_file_path (str): Le chemin du fichier d'environnemnt.
        logging_config_path (str): Le chemin du fichier de configuration de logging.
        model_cache_dir (str): Le chemin du repertoire de cache du modèle.
        embedding_model_device (str): Périphérique utilisé pour le modèle d'embedding.
        #debug (bool): Indique si le bot sera exécuté en mode production ou developpement.

    Returns:
        NewsBot: Une instance de la classe NewsBot.
    """

    from financial_bot import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from news_bot import utils
    from news_bot.langchain_bot import NewsBot

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    bot = NewsBot(
        model_cache_dir=Path(model_cache_dir) if model_cache_dir else None,
        embedding_model_device=embedding_model_device,
    )

    return bot


@news_bot.rest_api(keep_warm_seconds=300, loader=load_bot)
def run(**inputs):
    """
   Exécute le bot sur l'endpoint Beam RESTful API.

     Args:
        inputs (dict): Un dictionnaire contenant les clés suivants:
            - context: L'instance du bot.
            - news_category (str): La rubrique d'information choisie par l'utilisateur.
            - question (str): La question de l'utilisateur.
            - history (list): Une liste des précédentes conversations (optionnnel).

    Returns:
        str: La réponse du bot à la question de l'utilisateur.
    """

    response = _run(**inputs)

    return response



def run_local(
    about_me: str,
    question: str,
    history: List[Tuple[str, str]] = None,
    #debug: bool = False,
):
    """
    Exécute le bot localement en mode prod ou dev.

    Args:
        news_category (str): Un texte contenat la rubrique d'information.
        question (str): Un texte contenant la question de l'utilisateur.
        history (List[Tuple[str, str]], optional): Une liste des tuples contenant les précédentes questions de l'utilisateur
        et les réponses du bot. None par défaut.
        #debug (bool, optional): Un booléen qui indique si le bot s'exécute en mode prod ou dev. False par défaut.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    # if debug is True:
    #     bot = load_bot_dev(model_cache_dir=None)
    # else:
    #     bot = load_bot(model_cache_dir=None)
    bot = load_bot(model_cache_dir=None)
    inputs = {
        "news_category": about_me,
        "question": question,
        "history": history,
        "context": bot,
    }

    response = _run(**inputs)

    return response


def _run(**inputs):
    """
    Une fonction centrale qui invoque le bot et retourne sa réponse.

    Args:
        inputs (dict): Un dictionnaire contenant les clés suivants:
            - context: L'instance du bot.
            - news_category (str): La rubrique d'information choisie par l'utilisateur.
            - question (str): La question de l'utilisateur.
            - history (list): Une liste des précédentes conversations (optionnnel).

    Returns:
        La réponse du bot à la question de l'utilisateur.
    """

    from news_bot import utils

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    bot = inputs["context"]
    input_payload = {
        "news_category": inputs["news_category"],
        "question": inputs["question"],
        "to_load_history": inputs["history"] if "history" in inputs else [],
    }
    response = bot.answer(**input_payload)

    return response



if __name__ == "__main__":
    fire.Fire(run_local)