from typing import Dict, Optional, List
import hashlib
from pydantic import BaseModel
from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean,clean_non_ascii_chars, replace_unicode_quotes
from unstructured.staging.huggingface import chunk_by_attention_window
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from src.vector_db_api import get_qdrant_client, init_collection
from src.logger import get_console_logger
from src.paths import DATA_DIR


NEWS_FILE = DATA_DIR/"news_01_02_2024.json"
QDRANT_COLLECTION_NAME = "newsdata"
QDRANT_VECTOR_SIZE = 384

logger = get_console_logger(__name__)

# Tokenizer et LLM à utiliser pour calculer les embeddings des documents de texte
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# Initialiser le client Qdrant et la Collection où les documents seront enregistrés
qdrant_client = get_qdrant_client()
qdrant_client = init_collection(
    qdrant_client=qdrant_client,
    collection_name=QDRANT_COLLECTION_NAME,
    vector_size=QDRANT_VECTOR_SIZE,
)

class Document(BaseModel):
    id:str
    group_key:Optional[str]=None
    metadata:Optional[dict]={}
    text:Optional[list]=[]
    chunks:Optional[list]=[]
    embeddings:Optional[list]=[]

def parse_document(_data:Dict)->Document:
    
    document_id = hashlib.md5(_data["content"].encode()).hexdigest()
    document = Document(id=document_id)
    article_elements = partition_html(text=_data["content"])
    _data['content']=clean_non_ascii_chars(
        replace_unicode_quotes(
            clean(" ".join(str(x) for x in article_elements)
                  )
            )
        )
    _data["title"] = clean_non_ascii_chars(
        replace_unicode_quotes(
            clean(_data["news_category"]+': '+_data["title"]
                  )
                             )
                                        )
    _data["description"] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data["description"])))
    
    document.text=[_data["title"],_data["description"],_data['content']]
    document.metadata["news_category"] = _data['news_category']
    document.metadata['title'] = _data['title']
    document.metadata['date'] = _data['pubDate']
    document.metadata['description'] = _data.get("description","")
    document.metadata['link'] = _data.get("link","")
    document.metadata['article_id']=_data.get('article_id',"")
    
    return document


def chunk(document:Document)->Document:
    chunks = []
    for text in document.text:
        chunks += chunk_by_attention_window(
            text=text,
            tokenizer=tokenizer,
            max_input_size=QDRANT_VECTOR_SIZE
        )
    document.chunks = chunks
    
    return document

# Créer un embedding et le stocker dans la base des données de vecteur
def embedding(document:Document)->Document:
    for chunk in document.text:
        inputs = tokenizer(
            chunk,
            padding =True,
            truncation=True,
            return_tensors='pt',
            max_length=QDRANT_VECTOR_SIZE,
        )
        
        result = model(**inputs)
        
        embeddings = result.last_hidden_state[:,0,:].cpu().detach().numpy()
        lst = embeddings.flatten().tolist()
        document.embeddings.append(lst)
        
    return document

def build_payloads(doc:Document)->List:
    payloads = []
    ids = []
    for chunk in doc.chunks:
        payload = doc.metadata
        payload.update({"text":chunk})
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        ids.append(chunk_id)
        payloads.append(payload)
    
    return ids, payloads

def push_document_to_qdrant(document:Document)->None:
    from qdrant_client.models import PointStruct
    ids, _payloads = build_payloads(doc=document)
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=idx,
                vector=vector,
                payload=_payload
            )
            for idx, vector, _payload in zip(ids, document.embeddings, _payloads)
        ]
    )


def process_one_document(_data:Dict)->Document:
    doc = parse_document(_data=_data)
    doc = chunk(doc)
    doc= embedding(doc)
    push_document_to_qdrant(doc)
    
    return doc


def embed_news_into_qdrant(news_data:List(Dict),n_processes:int=1)->None:
    results = []
    if n_processes ==1:
        #Séquentiel
        for _data in tqdm(news_data):
            result = process_one_document(_data=_data)
            results.append(result)
    else:
        # En parallèle
        import multiprocessing
        
        # Créer un pool de multiprocessing
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(func=process_one_document,iterable=news_data),
                     total=len(news_data),
                     desc='Processing',
                     unit='news'
                     )
                )

if __name__ == "__main__":
    import json
    with open(NEWS_FILE,"w") as json_file:
        news_data = json.load(fp=json_file)
    
    embed_news_into_qdrant(
        news_data=news_data,
        n_processes=1
    )