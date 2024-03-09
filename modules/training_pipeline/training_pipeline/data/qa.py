from dataclasses import asdict, dataclass  # ,field
from typing import Dict, List, Optional
from pathlib import Path

from datasets import Dataset
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

from training_pipeline.constants import Scope
from training_pipeline.data.utils import load_json
from training_pipeline.prompt_templates.prompter import get_llm_template


@dataclass(frozen=True)
class DataSample:
    """
    Un échantillon de données pour le modèle de question et réponse.
    Args
        - news_context (str): Le contexte des nouvelles pour la question.
        - chat_history (str): L'historique des chats pour la question.
        - question (str): La question qu'il faut répondre.
        - answer (str): La réponse à la question.
    """

    news_category: str = ""
    news_context: str = ""
    chat_history: str = ""
    question: str = ""
    answer: str = ""


class NewsDataset:
    def __init__(
        self,
        data_path: Path,
        scope: Scope = Scope.TRAINING,
        template: str = "mistral",
        max_samples: Optional[int] = None,
    ):
        """
        Une classe représentant un Dataset des articles de presse.
        Args:
            - data_path(Path) : le chemin du fichier des données.
            - scope(Scope, optionnel):  la portée des données. La valeur par défaut est Scope.TRAINING.
            - template(str, optionnel): le template à utiliser pour les données. La valeur par défaut est mistral.
            - max_samples(Optional[int], optionnel): le nombre maximal d'observations à utiliser. La valeur par défaut est None.
        """
        self._data_path = data_path
        self._scope = scope
        self._template = get_llm_template(template)
        self._max_samples = max_samples
        self._raw_data = self.load(data_path=data_path)

    def load(self, data_path: Path) -> List[DataSample]:
        """
        Récupère des données à partir du chemin indiqué.
        Args:
            - data_path(Path): le chemin à partir duquel les données seront récupérées.
        Returns:
            - List(DataSample): les données récupérées.

        """
        data = load_json(data_path)
        if self._max_samples is not None:
            data = data[: self._max_samples]

        return self.deserialize(data)

    def deserialize(self, data: List[dict]) -> List[DataSample]:
        """
        Desérialise les données.
        Args:
            - data(List[dict]): les données à desérialiser.
        Returns:
            - List[DataSample]: les données desérialisées.
        """

        if self._scope == Scope.TRAINING:
            return [
                DataSample(
                    news_category=sample["news_category"],
                    news_context=sample["news_context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                    answer=sample["response"],
                )
                for sample in data
            ]
        else:
            return [
                DataSample(
                    news_category=sample["news_category"],
                    news_context=sample["news_context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                )
                for sample in data
            ]

    def to_huggingface(self) -> Dataset:
        """
        Pretraite les données et renvoie un Dataset HuggingFace.
        Returns:
            - Dataset: un Dataset HuggingFace.
        """
        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        if self._scope == Scope.TRAINING:
            template_maping_func = self._template.format_train
        else:
            template_maping_func = self._template.format_infer
        dataset = dataset.map(self.clean)
        dataset = dataset.map(template_maping_func, remove_columns=dataset.column_names)

        return dataset

    def clean(self, samples: Dict[str, str]) -> Dict[str, str]:
        """
        Nettoie les données.
        Args:
            - samples(Dict[str, str]): les données à nettoyer.
        Returns:
            - Dict[str, str]: les données nettoyées.
        """
        for key, sample in samples.items():
            cleaned_sample = clean_extra_whitespace(sample)
            cleaned_sample = group_broken_paragraphs(cleaned_sample)

            samples[key] = cleaned_sample

        return samples
