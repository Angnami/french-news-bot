from typing import Any, Dict

from langchain.callbacks.base import BaseCallbackHandler
import comet_llm

from news_bot import constants


class CometLLMMonitoringHandler(BaseCallbackHandler):
    """
    Un gestionnaire de callback pour suivre les modèles LMM en utilisant comet.lm. 
        Args:
        project_name (str): Le nom du projet Comet.ml pour l'écriture des logs.
        llm_model_id (str): L'ID du moèdle LLM à utiliser pour l'inférence.
        llm_qlora_model_id (str): L'ID du modèle PEFT à utiliser pour l'inférence.
        llm_inference_max_new_tokens (int): Le nombre maximum de nouveaux tokens à générer durant l'inférence.
        llm_inference_temperature (float): La température à utiliser durant l'inférence.
    """
    
    
    def __init__(
        self,
        project_name: str = None,
        llm_model_id: str = constants.LLM_MODEL_ID,
        llm_qlora_model_id: str = constants.LLM_QLORA_CHECKPOINT,
        llm_inference_max_new_tokens: int = constants.LLM_INFERNECE_MAX_NEW_TOKENS,
        llm_inference_temperature: float = constants.LLM_INFERENCE_TEMPERATURE,
    ):
        self._project_name = project_name
        self._llm_model_id = llm_model_id
        self._llm_qlora_model_id = llm_qlora_model_id
        self._llm_inference_max_new_tokens = llm_inference_max_new_tokens
        self._llm_inference_temperature = llm_inference_temperature
        
        
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """
        Une fonction callback qui écrit le prompt et l'output dans Comet.ml.

        Args:
            outputs (Dict[str, Any]): L'output du modèle LLM.
            **kwargs (Any): D'autres arguments à passer à la fonction.
        """

        should_log_prompt = "metadata" in kwargs
        if should_log_prompt:
            metadata = kwargs["metadata"]

            comet_llm.log_prompt(
                project=self._project_name,
                prompt=metadata["prompt"],
                output=outputs["answer"],
                prompt_template=metadata["prompt_template"],
                prompt_template_variables=metadata["prompt_template_variables"],
                metadata={
                    "usage.prompt_tokens": metadata["usage.prompt_tokens"],
                    "usage.total_tokens": metadata["usage.total_tokens"],
                    "usage.max_new_tokens": self._llm_inference_max_new_tokens,
                    "usage.temperature": self._llm_inference_temperature,
                    "usage.actual_new_tokens": metadata["usage.actual_new_tokens"],
                    "model": self._llm_model_id,
                    "peft_model": self._llm_qlora_model_id,
                },
                duration=metadata["duration_milliseconds"],
            )