from pathlib import Path


PARENT_DIR = Path(__file__).parent.resolve().parent

DATA_DIR  = PARENT_DIR / 'data'

Path.mkdir(DATA_DIR, exist_ok=True, parents=True)
