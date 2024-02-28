from typing import Any, Dict
from pathlib import Path
from dataclasses import dataclass

from transformers import TrainingArguments

from training_pipeline.data.utils import load_yaml


@dataclass
class TrainingConfig:
    """
    Une classe de configuration utilisée pour charger et stocker la configuration d'entrainement.
    Attributs:
        - training(TrainingArguments) : arguments d'entrainement utilisés pour entrainer le modèle.
        - model(Dict[str, Any]): dictionnaire contenant la configuration du modèle.
    """

    training: TrainingArguments
    model: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path, output_dir: Path) -> "TrainingConfig":
        """
        Récupère un fichier de configuration à partir du lien indiqué.
        Args:
            - config_path(Path): le chemin du fichier à récupérer.
            - output_dir(Path): le chemin de l'enregistrement du fichier.
        Returns:
            - TrainingConfig: object de configuration d'entrainement.
        """

        config = load_yaml(config_path)
        config["training"] = cls._dict_to_training_arguments(
            training_config=config["training"], output_dir=output_dir
        )

        return cls(**config)

    @classmethod
    def _dict_to_training_arguments(
        cls, training_config: dict, output_dir: Path
    ) -> TrainingArguments:
        """
        Construit un object TrainingArguments à partir d'un dictionnaire de configuration.
        Args:
            - training_config(dict): le dictionnaire contenant la configuration d'entrainement.
            - output_dir(Path): le chemin de sauvegarde de l'output.
        Returns:
            - TrainingArguments: l'objet TrainingArguments
        """

        return TrainingArguments(
            output_dir=str(output_dir),
            logging_dir=str(output_dir / "logs"),
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            eval_accumulation_steps=training_config["eval_accumulation_steps"],
            optim=training_config["optim"],
            save_steps=training_config["save_steps"],
            logging_steps=training_config["logging_steps"],
            learning_rate=training_config["learning_rate"],
            fp16=training_config["fp16"],
            max_grad_norm=training_config["max_grad_norm"],
            num_train_epochs=training_config["num_train_epochs"],
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            evaluation_strategy=training_config["evaluation_strategy"],
            eval_steps=training_config["eval_steps"],
            report_to=training_config["report_to"],
            seed=training_config["seed"],
            load_best_model_at_end=training_config["load_best_model_at_end"],
        )


@dataclass
class InferenceConfig:
    """
    Une classe représentant la configuration de l'inférence.
    Attributs:
        - model(Dict[str, Any]): un dictionnaire contenant la configuration du modèle.
        - peft_model(Dict[str, Any]): un dictionnaire contenant la configuration du modèle PEFT.
        - setup_config(Dict[str, Any]):  un dictionnaire contenant la configuration d'installation.
        - dataset(Dict[str, str]): un dictionnaire contenant la configuration du dataset.
    """

    model: Dict[str, Any]
    peft_model: Dict[str, Any]
    setup: Dict[str, Any]
    dataset: Dict[str, str]

    @classmethod
    def from_yaml(cls, config_path: Path):
        """
        Télécharge un fichier de configuration à partir du chemin inidiqué.
        """

        config = load_yaml(config_path)

        return cls(**config)
