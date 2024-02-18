import logging
import time
import os
from pathlib import Path
from typing import Tuple, Optional

import comet_llm
from datasets import Dataset
from peft import PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_pipeline import constants, models
from training_pipeline.configs import InferenceConfig
from training_pipeline.data import qa, utils
from training_pipeline.prompt_templates.prompter import get_llm_template

try:
    comet_project_name = os.environ["COMET_PROJECT_NAME"]
except KeyError:
    raise RuntimeError("Veuillez définir la variable environnementale COMET_PROJECT_NAME")

logger = logging.getLogger(name=__name__)

class InferenceAPI:
    """
    Une classe effectuant l'inférence en utilisant un modèle LLM pré-entrainé.
    Args:
        -peft_model_id(str):l'ID du modèle peft à utiliser.
        -model_id(str):l'ID du modèle LLM à utiliser.
        -template_name(str):le nom du template LLM à utiliser.
        -root_dataset_dir(Path):le repertoire racine du dataset
        -test_dataset_file(Path):le chemin du fichier du dataset de test.
        -name(str,optional):le nom de l'API d'inférence.La valeur par défaut est "inference-api". 
        -max_new_tokens(int,optional):le nombre maximal de nouveaux tokens à générer.La valeur par
        défaut est 50. 
        -temperature(float,optional):la température à utiliser pour générer des nouveaux tokens.La valeur 
        par défaut est 1.0. 
        -model_cache_dir(Path,optional):le repertoire de cache du modèle.La valeur par défaut est constants.CACHE_DIR. 
        -debug(bool,optional):indique s'il faut exécuter en mode debogage ou pas.La valeur par défaut est False.
        -device(str,optional):le périphérique à utiliser pour l'inférence.La valeur par défaut est "cuda:0".
    """
    def __init__(
        self,
        peft_model_id:str,
        model_id:str,
        template_name:str,
        root_dataset_dir:Path,
        test_dataset_file:Path,
        name:str="inference-api",
        max_new_tokens:int=50,
        temperature:float=1.0,
        model_cache_dir:Path=constants.CACHE_DIR,
        debug:bool=False,
        device:str="cuda:0",):

        self._peft_model_id=peft_model_id
        self._model_id=model_id
        self._template_name=template_name
        self._prompt_template = get_llm_template(template_name)
        self._root_dataset_dir=root_dataset_dir
        self._test_dataset_file=test_dataset_file
        self._name=name
        self._max_new_tokens=max_new_tokens
        self._temperature=temperature
        self._model_cache_dir=model_cache_dir
        self._debug=debug
        self._device=device

        self._model, self._tokenizer,self._peft_config=self.load_model()
        if self._root_dataset_dir is not None:
            self._dataset = self.load_data()
        else:
            self._dataset = None
        
    @classmethod
    def from_config(
        cls,
        config:InferenceConfig,
        root_dataset_dir:Path,
        model_cache_dir:Path=constants.CACHE_DIR,
        ):
        """
        Crée une instance de classe InferenceApi à partir d'un object InferenceConfig.
        Args:
            -config(InferenceConfig):l'objet InferenceConfig à utiliser.
            -root_dataset_dir(Path):le repertoire racine du dataset à utiliser.
            -model_cache_dir(Path,optional):le repertoire de la cache du modèle LLM.La valeur par
            défaut est constants.CACHE_DIR. 
        Returns:
            InferenceApi:une instance de la classe InferenceApi.
        """

        return cls(
            peft_model_id=config.peft_model["id"],
            model_id=config.model["id"],
            template_name=config.model["template_name"],
            root_dataset_dir=root_dataset_dir,
            test_dataset_file=config.dataset["file"],
            model_cache_dir=model_cache_dir,
            max_new_tokens=config.model["max_new_tokens"],
            temperature=config.model["temperature"],
            debug=config.setup.get("debug",False),
            device=config.setup.get("device","cuda:0")
        )

    def load_data(self)->Dataset:
        """
        Récupère le dataset des QR(QA).
        Returns:
            -Dataset:le dataset récupéré.
        """
        logger.info(msg=f"Récupération du dataset des QA à partir de {self._root_dataset_dir=}")
        if self._debug:
            max_samples=3
        else:
            max_samples=None
        dataset=qa.NewsDataset(
            data_path=self._root_dataset_dir/self._test_dataset_file,
            template=self._template_name,
            scope=constants.Scope.INFERENCE,
            max_samples=max_samples
            ).to_huggingface()
        logger.info(msg=f"{len(dataset)} échantillons récupérés pour l'inférence.")
        
        return dataset
    
    def load_model(self)->Tuple[AutoModelForCausalLM,AutoTokenizer,PeftConfig]:
        """
        Récupère le LMM modèle pour l'inférence.
        Returns:
            -Tuple[AutoModelForCausalLM,AutoTokenizer,PeftConfig]:un tuple contenant le modèle LLM chargé, 
            le tokenizer et la configuration PEFT.
        """
        logger.info(msg=f"Chargement du modèle {self._model_id=} et de la {self._peft_config=}")
        model, tokenizer, peft_config=models.build_qlora_model(
            pretrained_model_name_or_path=self._model_id,
            peft_pretrained_model_name_or_path=self._peft_model_id,
            gradient_checkpointing=False,
            cache_dir=self._model_cache_dir)
        model.eval()

        return model, tokenizer, peft_config

    def infer(self,infer_prompt:str,infer_payload:dict)->str:
        """
        Effectue l'inférence en utilisant le modèle LLM récupéré.
        Args:
            -infer_prompt(str):le prompt à utiliser pour l'inférence.
            -infer_payload(str):payload à utiliser pour l'inférence.
        Returns:
            -str:la réponse générée.
        """
        start_time =time.time()
        answer=models.prompt(
            model=self._model_id,
            tokenizer=self._tokenizer,
            input_text=infer_prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            device=self._device,
            return_only_answer=True
        )
        end_time=time.time()
        duration_milliseconds=(end_time - start_time)*1000

        if not self._debug:
            comet_llm.log_prompt(
                project=f"{comet_project_name}-{self._name}-monitor-prompts",
              prompt=infer_prompt,
              output=answer,
              prompt_template=self._prompt_template.infer_raw_template,
              prompt_template_variables=infer_payload,
              # TODO: Count tokens instead of using len().
              metadata={
                "usage.prompt_tokens": len(infer_prompt),
                "usage.total_tokens": len(infer_prompt) + len(answer),
                "usage.max_new_tokens": self._max_new_tokens,
                "usage.actual_new_tokens": len(answer),
                "model": self._model_id,
                "peft_model": self._peft_model_id,
              },
              duration=duration_milliseconds  
            )
        
        return answer
    
    def infer_all(self,output_file:Optional[Path]=None)->None:
        """
        Effectue l'inférence sur l'ensemble d'échantillons récupérés dans dataset.
        Args:
            -output_file(Optional[Path],optional):le fichier de sauvegarde de l'output. la valeur par
            défaut est None.
        """
        assert (
            self._dataset is not None
            ), "Dataset non chargé.Fournir un repertoire dataset au constructeur de la classe:'root_dataset_dir.'"
        promt_and_answers = []
        should_save_output = output_file is not None
        for sample in tqdm(self._dataset):
            answer=self.infer(
                infer_prompt=sample["prompt"],infer_payload=sample["payload"]
            )
            if should_save_output:
                promt_and_answers.append(
                    {
                        "prompt":sample["prompt"],
                        "answer":answer
                    }
                )
        if should_save_output:
            utils.write_json(promt_and_answers,output_file)
