import logging
import os
from typing import Iterable, List, Tuple
from pathlib import Path

from langchain import chains
from langchain.memory import ConversationBufferWindowMemory

from news_bot import constants
from news_bot.chains import (
    ContextExtractorChain,
    NewsBotQAChain,
    StatelessMemorySequentialChain
)

from news_bot.embeddings import EmbeddingModelSingleton
from news_bot.handlers import CometLLMMonitoringHandler
from news_bot.models import build_huggingface_pipeline
from news_bot.qdrant import build_qdrant_client
from news_bot.template import get_llm_template

logger = logging.getLogger(__name__)


class NewsBot:
    """
    Un robot conversationnel qui utilise un modèle de langue pour générer des réponses aux inputs de l'utilisateur.

    Args:
        llm_model_id (str): L'ID du modèle de langue de Hugging Face à utiliser.
        llm_qlora_model_id (str): L'ID du modèle QLora de Hugging Face à utiliser.
        llm_template_name (str): Le nom du template LLM à utiliser.
        llm_inference_max_new_tokens (int): Le nombre maximum des tokens à générer lors de l'inférence.
        llm_inference_temperature (float): La temperature à utiliser lors de l'inférence.
        vector_collection_name (str): Le nom de la collection de vecteurs Qdrant à utiliser.
        vector_db_search_topk (int): Le nombre des voisins les plus proches à chercher dans la base de vecteurs Qdrant.
        model_cache_dir (Path): Le repertoire à utiliser pour le caching des modèles de langue et d'embedding.
        streaming (bool): Indique si api de streaming de Hugging Face est utilisé ou pas pour l'inférence.
        embedding_model_device (str): Le périphérique à utiliser pour le modèle d'embedding.

    Attributes:
        newsbot_chain (Chain): La chaîne linguistique qui génère des réponses aux entrées de l'utilisateur.
    """

    def __init__(
        self,
        llm_model_id: str = constants.LLM_MODEL_ID,
        llm_qlora_model_id: str = constants.LLM_QLORA_CHECKPOINT,
        llm_template_name: str = constants.TEMPLATE_NAME,
        llm_inference_max_new_tokens: int = constants.LLM_INFERNECE_MAX_NEW_TOKENS,
        llm_inference_temperature: float = constants.LLM_INFERENCE_TEMPERATURE,
        vector_collection_name: str = constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
        vector_db_search_topk: int = constants.VECTOR_DB_SEARCH_TOPK,
        model_cache_dir: Path = constants.CACHE_DIR,
        streaming: bool = False,
        embedding_model_device: str = "cuda:0",):
        
        self._llm_model_id = llm_model_id
        self._llm_qlora_model_id = llm_qlora_model_id
        self._llm_template_name = llm_template_name
        self._llm_template = get_llm_template(name=self._llm_template_name)
        self._llm_inference_max_new_tokens = llm_inference_max_new_tokens
        self._llm_inference_temperature = llm_inference_temperature
        self._vector_collection_name = vector_collection_name
        self._vector_db_search_topk = vector_db_search_topk

        self._qdrant_client = build_qdrant_client()

        self._embd_model = EmbeddingModelSingleton(
            cache_dir=model_cache_dir, device=embedding_model_device
        )
        self._llm_agent, self._streamer = build_huggingface_pipeline(
            llm_model_id=llm_model_id,
            llm_lora_model_id=llm_qlora_model_id,
            max_new_tokens=llm_inference_max_new_tokens,
            temperature=llm_inference_temperature,
            use_streamer=streaming,
            cache_dir=model_cache_dir, )
        
        self.newsbot_chain = self.build_chain()

    @property
    def is_streaming(self) -> bool:
        return self._streamer is not None

    def build_chain(self) -> chains.SequentialChain:
        """
        Construit et retourne un robot (une chaine de langue) des actualités.
        Cette chaine est désignée pour recevoir le nom d'une rubrique d'informations `news_category` et une `question` et se
        connecte à une base de vecteurs, recherche des articles de presse liés à la question de l'utilisateur et les intègre dans
        le payload passé comme prompt au modèle LLM fine-tuné qui va produire une réponse.

        Cette chaine est constituée de deux étapes primaires:
        1. Extracteur de contexte: Cette étape est responsable de l'embedding de la question d'utlisateur,
        qui signifie la conversion du texte de la question en une représentation numérique.
        L'embedding de cette question est ensuite utilisé pour récupérer du contexte pertinent à partir de la base de vecteurs.
        L'output de cette chaine est un dictionnaire payload.

        2. Générateur LLM: Une fois que le contexte est récupéré, cette étape l'utilise pour formater un prompt complet pour
        le LLM et alimente le modèle afin d'obtenir une réponse pertinente pour la question de l'utilisateur.

        Returns
        -------
        chains.SequentialChain
            La chaine de conversation construite.

        Notes
        -----
        Le flux réél de traitement dans la chaine peut se matérialiser comme suit:
        [news_category: str][question: str] > ContextChain >
        [news_category: str][question:str] + [context: str] > NewsChain >
        [answer: str]
        """

        logger.info("Construction 1/3 - ContextExtractorChain")
        context_retrieval_chain = ContextExtractorChain(
            embedding_model=self._embd_model,
            vector_store=self._qdrant_client,
            vector_collection=self._vector_collection_name,
            top_k=self._vector_db_search_topk,
        )

        logger.info("Construction 2/3 - NewsBotQAChain")
        try:
            comet_project_name = os.environ["COMET_PROJECT_NAME"]
        except KeyError:
            raise RuntimeError(
                "Veuillez définir la variable d'environnement COMET_PROJECT_NAME!"
            )
        callabacks = [
            CometLLMMonitoringHandler(
                project_name=f"{comet_project_name}-monitor-prompts",
                llm_model_id=self._llm_model_id,
                llm_qlora_model_id=self._llm_qlora_model_id,
                llm_inference_max_new_tokens=self._llm_inference_max_new_tokens,
                llm_inference_temperature=self._llm_inference_temperature,
            )
        ]
        llm_generator_chain = NewsBotQAChain(
            hf_pipeline=self._llm_agent,
            template=self._llm_template,
            callbacks=callabacks,
        )

        logger.info("Construction 3/3 - Connexion des chains en SequentialChain")
        seq_chain = StatelessMemorySequentialChain(
            history_input_key="to_load_history",
            memory=ConversationBufferWindowMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                k=3,
            ),
            chains=[context_retrieval_chain, llm_generator_chain],
            input_variables=["news_category", "question", "to_load_history"],
            output_variables=["answer"],
            verbose=True,
        )

        logger.info("Construction de SequentialChain terminée.")
        logger.info("Workflow:")
        logger.info(
            """
            [news_category: str][question: str] > ContextChain > 
            [news_category: str][question:str] + [context: str] > NewsChain > 
            [answer: str]
            """
        )

        return seq_chain

    def answer(
        self,
        news_category: str,
        question: str,
        to_load_history: List[Tuple[str, str]] = None,) -> str:
        
        """
        Etant données une rubrique d'informations et une question, fait générer une réponse au LLM.

        Parameters
        ----------
        news_category : str
            La rubrique d'information choisie.
        question : str
            La question de l'utilisateur.

        Returns
        -------
        str
            La réponse générée par le LLM.
        """

        inputs = {
            "news_category": news_category,
            "question": question,
            "to_load_history": to_load_history if to_load_history else [],
        }
        response = self.newsbot_chain.run(inputs)

        return response

    def stream_answer(self) -> Iterable[str]:
        """ Diffuse la réponse du LLM après que chaque token soit généré après avoir appelé `answer()`"""

        assert (
            self.is_streaming
        ), "Réponse en flux non disponible. Construit le bot avec `use_streamer=True`."

        partial_answer = ""
        for new_token in self._streamer:
            if new_token != self._llm_template.eos:
                partial_answer += new_token

                yield partial_answer
