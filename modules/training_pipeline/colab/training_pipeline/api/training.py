import logging
from pathlib import Path
from typing import Optional, Tuple

import comet_ml
from datasets import Dataset
from peft import PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from trl import SFTTrainer

import sys
sys.path.append("..")

from training_pipeline import constants, metrics, models
from training_pipeline.configs import TrainingConfig
from training_pipeline.data import qa

logger = logging.getLogger(name=__name__)


class BestModelToModelRegistryCallback(TrainerCallback):
    """
    C'est un callback qui enregistre la meilleure version du modèle dans le registre de modèle comet.ml
    Args:
        - model_id(str): l'ID du modèle à enregistrer dans le registre de modèle.

    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    @property
    def model_name(self) -> str:
        """
        Renvoie le nom du modèle à enregister dans le registre de modèle.
        """
        return f"french_news_assistant/{self.model_id}"

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        C'est un événement appelé à la fin de chaque époque.
        Enregistre la meilleure version du modèle dans le registre comet.ml.

        """
        best_model_checkpoint = state.best_model_checkpoint
        has_best_model_checkpoint = best_model_checkpoint is not None
        if has_best_model_checkpoint:
            best_model_checkpoint = Path(best_model_checkpoint)
            logger.info(
                msg=f"Enregistrement du melleur modèle à partir de {best_model_checkpoint} dans le registre de modèle."
            )
        else:
            logger.warning(
                msg="Pas de meileure version trouvée. L'enregistrement dans le regsitre n'est pas effectué."
            )

    def to_model_registry(self, checkpoint_dir: Path):
        """
        Enregistre un checkpoint donné du model dans le registre Comet.ml
        Args:
            - checkpoint_dir(Path): le chemin du repertoire contenant le checkpoint du modèle.

        """
        checkpoint_dir = checkpoint_dir.resolve()

        assert (
            checkpoint_dir.exists()
        ), f"Le repertoire du checkpoint {checkpoint_dir} n'existe pas."

        # Recupérer l'expérience obsolète à partir du contexte
        # global afin d'obtenir la clé d'API et l'ID de l'expérience.
        stale_experiment = comet_ml.get_global_experiment()
        # Reprendre l'expérience en utilisant sa clé API et son ID d'expérience
        experiment = comet_ml.ExistingExperiment(
            api_key=stale_experiment.api_key, experimet_key=stale_experiment.id
        )
        logger.info(
            msg=f"Début de l'enregistrement du checkpoint du modèle @ {self.model_name}"
        )
        experiment.log_model(name=self.model_name, file_or_folder=str(checkpoint_dir))
        experiment.end()
        logger.info(
            msg=f"Fin de l'enregistrement du checkpoint du modèle @ {self.model_name}"
        )


class TrainingAPI:
    """
    Une classe pour l'entrainement d'un modèle QLora.
    Args:
        -root_dataset_dir(Path): le repertoire racine du dataset
        -model_id(str): l'identifiant du modèle à utiliser.
        -template_name(str):le nom du template à utiliser.
        -training_arguments(TrainingArguments): les arguments d'entrainement.
        -name(str,optional):le nom de l'API d'entrainement. Le nom par défaut est "training-api".
        -max_seq_length(int,optional): la longueur de séquence maximale. La valeur par défaut est de 1024.
        -model_cache_dir(Path, optional): le repertoire de cache du modèle. la valeur par défaut est
        constants.CACHE_DIR
    """

    def __init__(
        self,
        root_dataset_dir: Path,
        model_id: str,
        template_name: str,
        training_arguments: TrainingArguments,
        name: str = "training-api",
        max_seq_length: int = 1024,
        model_cache_dir: Path = constants.CACHE_DIR,
    ):
        self._root_dataset_dir = root_dataset_dir
        self._model_id = model_id
        self._template_name = template_name
        self._training_arguments = training_arguments
        self._name = name
        self._max_seq_length = max_seq_length
        self._model_cache_dir = model_cache_dir

        self._training_dataset, self._validation_dataset = self.load_data()
        self._model, self._tokenizer, self._peft_config = self.load_model()

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        root_dataset_dir: Path,
        model_cache_dir: Optional[Path] = None,
    ):
        """
        Crée une instance de TrainingApi à partir d'un objet TrainingConfig.
        Args:
            -config(TrainingConfig):la configuration d'entrainement.
            -root_dataset_dir(Path): le repertoire racine du dataset.
            -model_cache_dir(Path,optional): le repertoire de cache du modèle. La valeur par défaut est None.
        Returns:
            - TrainingAPI:une instance de TrainingAPI.
        """
        return cls(
            root_dataset_dir=root_dataset_dir,
            model_id=config.model["id"],
            template_name=config.model["template"],
            training_arguments=config.training,
            max_seq_length=config.model["max_seq_length"],
            model_cache_dir=model_cache_dir,
        )

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Charge les datasets d'entrainement et de validation.
        Returns:
            -Tuple[Dataset, Dataset]: un tuple contenant les datasets d'entrainement et de validation.
        """
        logger.info(
            msg=f"Chargement des Datasets de QR à partir {self._root_dataset_dir=}"
        )
        training_dataset = qa.NewsDataset(
            data_path=self._root_dataset_dir / "training_data.json",
            template=self._template_name,
            scope=constants.Scope.TRAINING,
        ).to_huggingface()

        validation_dataset = qa.NewsDataset(
            data_path=self._root_dataset_dir / "testing_data.json",
            scope=constants.Scope.TRAINING,
            template=self._template_name,
        ).to_huggingface()
        logger.info(msg=f"Training dataset size: {len(training_dataset)}")
        logger.info(msg=f"Validation dataset size:{len(validation_dataset)}")

        return training_dataset, validation_dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        """
        Recupère le modèle.
        Returns:
            -Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:un tuple contenant le modèle
            le tokenizer, et la PeftConfig.
        """
        logger.info(msg=f"Chargement du modèle en utilisant {self._model_id=}")
        model, tokenizer, peft_config = models.build_qlora_model(
            pretrained_model_name_or_path=self._model_id,
            gradient_checkpointing=True,
            cache_dir=self._model_cache_dir,
        )

        return model, tokenizer, peft_config

    def train(self) -> SFTTrainer:
        """
        Entraine le modèle.
        Returns:
            -SFTTrainer:le modèle entrainé.
        """
        logger.info(msg="Entrainement du modèle...")

        trainer = SFTTrainer(
            model=self._model,
            train_dataset=self._training_dataset,
            eval_dataset=self._validation_dataset,
            peft_config=self._peft_config,
            dataset_text_field="prompt",
            max_seq_length=self._max_seq_length,
            args=self._training_arguments,
            packing=True,
            compute_metrics=self.compute_metrics,
            callbacks=[BestModelToModelRegistryCallback(model_id=self._model_id)],
        )

        trainer.train()

        return trainer

    def compute_metrics(self, eval_pred: EvalPrediction):
        """
        Calcule la métrique perplexité.
        Args:
            - eval_pred(EvalPrediction):l'évaluation de la prédiction.
        Returns:
            - dict:Un dictionnaire contenant la métrique perplexité.
        """
        return {"perplexity": metrics.compute_perplexity(eval_pred.predictions)}
