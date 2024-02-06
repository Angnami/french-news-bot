from pathlib import Path

# == Embeddings model ==
EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_MAX_INPUT_LENGTH = 384

# == VECTOR Database ==
VECTOR_DB_OUTPUT_COLLECTION_NAME = "french_news_bot"
VECTOR_DB_SEARCH_TOPK = 1

# == LLM Model ==
LLM_MODEL_ID = "tiiuae/falcon-40b"
LLM_QLORA_CHECKPOINT = "angnami/news-falcon-40b-lora:1.0.0"


LLM_INFERNECE_MAX_NEW_TOKENS = 500
LLM_INFERENCE_TEMPERATURE = 1.0


# == Prompt Template ==
TEMPLATE_NAME = "falcon"

# === Misc ===
CACHE_DIR = Path.home() / ".cache" / "french-news-bot"