import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from comet_ml import API
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline
)

from news_bot import constants


logger = logging.getLogger(__name__)

def download_from_model_registry(model_id:str,cache_dir:Optional[Path]=None)->Path:
    """
    Télécharge un modèle à partir du registre de modèles Comet ML.
    
    Args:
        model_id(str):Identifiant du modèle à récupérer sous la form 'workspace/model_name:version'. 
        cache_dir(Path, optionnel):le chemin de cache du modèle.'constants.CACHE_DIR' par défaut. 
    Returns:
        Path:Le chemin de téléchargement du modèle.
    """
    if cache_dir is None:
        cache_dir = constants.CACHE_DIR
    output_folder = cache_dir/'models'/model_id
    
    alrady_downloaded = output_folder.exists()
    
    if not alrady_downloaded:
        workspace,model_id = model_id.split('/')
        model_name,version = model_id.split(':')
        
        api = API()
        model=api.get_model(workspace=workspace,model_name=model_name)
        model.download(version=version,output_folder=output_folder, expand=True)
    else:
        logger.info(msg=f"Le modèle {model_id=} est déjà téléchargé dans {output_folder}.")
        
    subdirs = [d for d in output_folder.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        model_dir = subdirs[0]
    else:
        raise RuntimeError(
            f"Il doit être un seul repertoire dans le dossier du modèle. \
                Vérifier le modèle téléchargé dans : {output_folder}"
        )
    logger.info(msg=f"Le modèle {model_id=} téléchargé de Comet ML dans {model_dir}")
    
    return model_dir


class StopOnTokens(StoppingCriteriaList):
    """
    Critère d'arrêt qui arrête la génération lorsqu'un token spécifique est produit.
    Args:
        stop_ids(List[int]):Une liste de tokens qui déclenchent l'arrêt de la génération.
    
    """
    def __init__(self,stop_ids:List[int]):
        super().__init__()
        self._stop_ids = stop_ids
        
    def __call__(self,input_ids:torch.LongTensor, scores:torch.FloatTensor,**kwargs)->bool:
        
        """
        Vérifie si le dernier token généré est dans la liste stop_ids.
        Args:
            input_ids (torch.LongTensor): Les ids des inputs tokens.
            scores (torch.FloatTensor): Les scores des tokens générés.

        Returns:
            bool: True si le dernier token généré est dans la liste stop_ids, False sinon.
        """
        for stop_id in self._stop_ids:
            if input_ids[0][-1] == stop_id:
                return True

        return False


def build_huggingface_pipeline(
    llm_model_id: str,
    llm_lora_model_id: str,
    max_new_tokens: int = constants.LLM_INFERNECE_MAX_NEW_TOKENS,
    temperature: float = constants.LLM_INFERENCE_TEMPERATURE,
    gradient_checkpointing: bool = False,
    use_streamer: bool = False,
    cache_dir: Optional[Path] = None,) -> Tuple[HuggingFacePipeline, Optional[TextIteratorStreamer]]:
    
    """
    Construits un pipeline pour la génération de texte en utilisant un LLM personnalisé + un 
    checkpoint fine tuné.

    Args:
        llm_model_id (str): Le ID ou le chemin du modèle LLM.
        llm_lora_model_id (str): TLe ID ou le chemin du modèle LLM LoRA.
        max_new_tokens (int, optional): Le nombre maximum de nouveaux tokens à générer. 128 par défaut.
        temperature (float, optional): La temperature à utiliser pour le sampling. 0.7 par défaut.
        gradient_checkpointing (bool, optional): Indique si un checkpointing du gradient est utilisé ou pas. False par défaut.
        cache_dir (Optional[Path], optional): Le reprtoire utilisé pour la cache. None par défaut.
        use_streamer (bool, optional): Indique si un iterateur de texte streamer est utilisé ou pas. False par défaut.

    Returns:
        Tuple[HuggingFacePipeline, Optional[TextIteratorStreamer]]: Un tuple contenant un pipeline HuggingFace 
        et un iterateur de texte streamer (si utilisé).
    """
    model, tokenizer, _ = build_qlora_model(
        pretrained_model_name_or_path=llm_model_id,
        peft_pretrained_model_name_or_path=llm_lora_model_id,
        gradient_checkpointing=gradient_checkpointing,
        cache_dir=cache_dir,)
    
    model.eval()
    
    if use_streamer:
        streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        stop_on_tokens = StopOnTokens(stop_ids=[tokenizer.eos_token_id])
        stopping_criteria = StoppingCriteriaList([stop_on_tokens])
    else:
        streamer = None
        stopping_criteria = StoppingCriteriaList([])

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    return hf, streamer



def build_qlora_model(
    pretrained_model_name_or_path:str="tiiuae/falcon-40b",
    peft_pretrained_model_name_or_path:Optional[str]=None,
    gradient_checkpointing:bool=True,
    cache_dir: Optional[Path]=None) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:

    """
    Cette fonction crée un modèle LLM QLora sur la base du nom du modèle de HuggingFace.
    1. Crée et prépare la configuration bitsandbytes pour la quantization de QLora.
    2. Télécharge, charge et quantize à la volée le modèle "tiiuae/falcon-40b".
    3. Crée et prépare la configuration Lora.
    4. Charge et configure le tokenizer de "tiiuae/falcon-40b".

    Args:
        - pretrained_model_name_or_path (str): le nom ou le chemin du modèle pre-entrainé de HuggingFace
        - peft_pretrained_model_name_or_path(str): le nom ou le chemin du modèle pre-entrainé de HuggingFace à utiliser pour 
        la Peft
        - gradient_checkpointing(bool): pour indiquer si on fait ou pas le gradient_checkpointing.
        - cache_dir(Path): le repertoire de la cache où sera enregistré le modèle.
    Returns:
        - Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig] : un tuple contenant le modèle construit, le tokenizer et le PeftConfig
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        revision='main',
        quanization_config=bnb_config,
        load_in_4bit=True,
        device_map='auto',
        trust_remote_code=False,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=False,
        truncation=True,
        cache_dir=str(cache_dir) if cache_dir else None,)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        with torch.no_grad():
            model.resize_token_embeddings(len(tokenizer))
        model.conig.pad_token_id = tokenizer.pad_token_id
    
    if peft_pretrained_model_name_or_path:
        is_model_name = not os.path.isdir(peft_pretrained_model_name_or_path)
        if is_model_name:
            logger.info(
                f"Téléchargement de {peft_pretrained_model_name_or_path} dépuis le registre de modèle de Comet ML"
            )
            peft_pretrained_model_name_or_path = download_from_model_registry(model=peft_pretrained_model_name_or_path,cache_dir=cache_dir)
        logger.info(msg=f"Chargement de la configuration Lora depuis {peft_pretrained_model_name_or_path}")
        lora_config =LoraConfig.from_pretrained(peft_pretrained_model_name_or_path)
        assert (
            lora_config.base_model_name_or_path == pretrained_model_name_or_path
        ), f"Modèle Lora entrained sur un modèle de base différent que celui demandé:\
            {lora_config.base_model_name_or_path} != {pretrained_model_name_or_path}"
        logger.info(msg=f"Chargement du modèle Peft à partir de {peft_pretrained_model_name_or_path}")
        model = PeftModel.from_pretrained(model=model, model_id=peft_pretrained_model_name_or_path)
    else:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=["query_key_value"]
        )
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache=(False, # Le caching n'est pas compatible avec le gradient_checkpointing
                                    )
        else:
            model.gradient_checkpointing_disable()
            model.config.use_cache=True, # C'est une bonne pratique de permettre le caching lorsque l'on fait de l'inférence

    return model, tokenizer, lora_config