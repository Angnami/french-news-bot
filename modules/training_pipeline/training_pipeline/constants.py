from enum import Enum
from pathlib import Path

class Scope(Enum):

    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'test'
    INFERENCE = 'inference'


CACHE_DIR = Path.home() /".cache" / "french-news-bot"


print(Path.home())