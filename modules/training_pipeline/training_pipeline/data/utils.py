import yaml
import json
from pathlib import Path
from typing import List, Union


def load_json(path: Path) -> dict:
    """Charge des données sous format json à partir d'un fichier.
    Args:
        - path(Path): le chemin du fichier json
    Returns:
        - dict : Les données json en dictionnaire
    """

    with path.open("r") as f:
        data = json.load(f)

    return data


def write_json(data: Union[dict, List[dict]], path: Path) -> None:
    """
    Enregistre un dictionnaire ou une liste de dictionnaires en  fichier json.
    Args:
        - data(Union[dict, List[dict]]): les données à enregistrer.
        - path(Path): le chemin du fichier à enregister.
    Returns:
        - None
    """

    with path.open("w") as f:
        json.dump(obj=data, fp=f, indent=4)


def load_yaml(path: Path) -> dict:
    """
    Récupère un fichier YAML à partir du lien indiqué et renvoie son contenu sous forme de dictionnaire.
    Args:
        - path(Path): Le chemin du fichier à récupérer.
    Returns:
        - dict: Contenu du fichier YAML sous forme de dictionnaire.
    """
    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config
