import os
from typing import Tuple, Optional
from pathlib import Path
import logging
import torch
from comet_ml import API
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from training_pipeline import constants

logger = logging.getLogger(__name__)


def build_qlora_model(
    pretrained_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
    peft_pretrained_model_name_or_path: Optional[str] = None,
    gradient_checkpointing: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
    """
    Cette fonction crée un modèle LLM QLora sur la base du nom du modèle de HuggingFace.
    1. Crée et prépare la configuration bitsandbytes pour la quantization de QLora.
    2. Télécharge, charge et quantize à la volée le modèle "mistralai/Mistral-7B-Instruct-v0.2".
    3. Crée et prépare la configuration Lora.
    4. Charge et configure le tokenizer de "mistralai/Mistral-7B-Instruct-v0.2".

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
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        revision="main",
        quantization_config=bnb_config,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=False,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=False,
        truncation=True,
        cache_dir=str(cache_dir) if cache_dir else None,
        padding_side = 'right'
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        with torch.no_grad():
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    if peft_pretrained_model_name_or_path:
        is_model_name = not os.path.isdir(peft_pretrained_model_name_or_path)
        if is_model_name:
            logger.info(
                f"Téléchargement de {peft_pretrained_model_name_or_path} dépuis le registre de modèle de Comet ML"
            )
            peft_pretrained_model_name_or_path = download_from_model_registry(
                model=peft_pretrained_model_name_or_path, cache_dir=cache_dir
            )
        logger.info(
            msg=f"Chargement de la configuration Lora depuis {peft_pretrained_model_name_or_path}"
        )
        lora_config = LoraConfig.from_pretrained(peft_pretrained_model_name_or_path)
        assert (
            lora_config.base_model_name_or_path == pretrained_model_name_or_path
        ), f"Modèle Lora entrainé sur un modèle de base différent que celui demandé:\
            {lora_config.base_model_name_or_path} != {pretrained_model_name_or_path}"
        logger.info(
            msg=f"Chargement du modèle Peft à partir de {peft_pretrained_model_name_or_path}"
        )
        model = PeftModel.from_pretrained(
            model=model, model_id=peft_pretrained_model_name_or_path
        )
    else:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
        )
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = (
                False,  # Le caching n'est pas compatible avec le gradient_checkpointing
            )
        else:
            model.gradient_checkpointing_disable()
            model.config.use_cache = (
                True,
            )  # C'est une bonne pratique de permettre le caching lorsque l'on fait de l'inférence

    return model, tokenizer, lora_config


def download_from_model_registry(model_id: str, cache_dir: Optional[Path] = None):
    """
    Cette fonction télécharge un modèle à partir du registre de modèle de Comet ML.
    Args:
        - model_id(str): ID du modèle à télécharger sous la forme "workspace/model_name:version"
        - cache_dir(Path): le repertoire de la cache du modèle téléchargé
    Returns:
        - Le chemin du modèle téléchargé
    """
    if cache_dir is None:
        cache_dir = constants.CACHE_DIR
    output_folder = cache_dir / "models" / model_id
    already_downloaded = output_folder.exists()
    if not already_downloaded:
        workspace, model_id = model_id.split("/")
        model_name, version = model_id.split(":")

        api = API()
        model = api.get_model(workspace=workspace, model_name=model_name)
        model.download(version=version, output_folder=output_folder, expand=True)
    else:
        logger.info(msg=f"Le modèle {model_id} est déjà téléchargé à {output_folder}")

    subdirs = [d for d in output_folder.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        model_dir = subdirs[0]
    else:
        raise RuntimeError(
            f"Il doit avoir un seul repertoire dans le dossier du modèle. Vérifier le modèle téléchargé dans {output_folder}"
        )
    logger.info(
        msg=f"Le modèle {model_id} est téléchargé depuis le registre dans {model_dir}"
    )

    return model_dir


def prompt(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    device: str = "cuda:0",
    return_only_answer: bool = False,
):
    """
    Cette fonction génére du texte basé sur celui fourni en entrée en utilisant le modèle et le tokenizer.
    Args:
        - model(transformers.PretrainedModel): le modèle à utiliser pour générer du texte.
        - tokenizer(transformers.PretrainedTokenizer): le tokenizer à utilser pour générer du texte.
        - input_text(str): le texte donné en entrée à partir duquel un nouveau texte sera généré.
        - max_new_tokens(int,optional): le nombre maximal de tokens à générer. La valeur par défaut est de 40.
        - temeperature(float, optional): la temperature à utilser pour générer du texte. La valeur par défaut est de 1.0.
        - device(str,optional): le périphérique à utiliser pour générer du texte. La valeur par défaut est cuda:0.
        - return_only_answer(bool, optional): C'est pour indiquer s'il faut renvoyer uniquement le texte généré ou toute la séquence générée.
    Returns:
        - le texte généré par le modèle.
    """
    inputs = tokenizer(
        input_text=input_text, return_tensors="pt", return_token_type_ids=False
    ).to(device)

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=temperature
    )
    output = outputs[0]
    if return_only_answer:
        input_ids = inputs.ids
        input_length = input_ids.shape[-1]
        output = outputs[input_length:]
    output = tokenizer.decode(output, skip_special_tokens=True)

    return output
