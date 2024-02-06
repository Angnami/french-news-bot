import numpy as np


def compute_perplexity(predictions: np.ndarray) -> float :

    return np.exp(predictions.mean()).item()