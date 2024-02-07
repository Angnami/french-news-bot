from transformers import AutoModel, AutoTokenizer

# Tokenizer et model LMM pour calculer les embeddings des documets de text

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')