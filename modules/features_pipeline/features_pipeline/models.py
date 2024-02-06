from typing import List, Optional, Tuple
import hashlib
from datetime import datetime

from pydantic import BaseModel
from unstructured.cleaners.core import (
    clean,
    replace_unicode_quotes,
    clean_non_ascii_chars
)
from unstructured.partition.html import partition_html
from unstructured.staging.huggingface import chunk_by_attention_window
from features_pipeline.embeddings import EmbeddingModelSingleton

class NewsArticle(BaseModel):
    """
    Représente un article de presse.
    Attributes:
        -article_id(str):l'identifiant de l'article de presse.
        -title(str):le titre de l'article de presse.
        -category(str):la rubrique d'information à laquelle appartient l'article.
        -content(str):le contenu de l'article.
        -pubDate(datetime):la date de publication de l'article.
        -link(str):le lien url de l'article.
        -description(str):une descrption sommaire de l'article.
    """
    title:str
    category:List[str]
    content:str
    pubDate:datetime
    link:Optional[str]
    description:str
    article_id:str
    
    def to_document(self)->'Document':
        """
        Convertit l'article de presse en un objet Document.
        Returns:
            -Document:un obejet Document représentant l'article de presse.
        """
        document_id = hashlib.md5(self.content.encode()).hexdigest()
        document=Document(id = document_id)
        article_elements = partition_html(text=self.content)
        cleaned_content = clean_non_ascii_chars(
            replace_unicode_quotes(
                clean(" ".join([str(x) for x in article_elements])
                      )
                )
            )
        cleaned_description = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.description))
        )
        cleaned_title = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.category[0]+': '+self.title))
            )
        
        document.text = [cleaned_title, cleaned_description,cleaned_content]
        document.metadata["news_category"] = self.category[0]
        document.metadata['title'] = self.title
        document.metadata['date'] = self.pubDate
        document.metadata['description'] = self.description
        document.metadata['link'] = self.link
        document.metadata['article_id']=self.article_id
        document.metadata['content']=cleaned_content
        return document
         
    
class Document(BaseModel):
    """
    Un modèle Pydantic représentant un Document.
    Attributes:
        -id(str):l'identifiant du document.
        -group_key(str):l'identifiant du groupe du document.
        -metadata(dict):les meta-données du document.
        -text(str):le contenu du document.
        -chunks(list):les chunks du document.
        -embeddings(list):les embeddings du document.
    Methods:
        -to_payloads: Retourne les payloads du document.
        -compute_chunks: Calcule les chunks du document.
        -compute_embeddings: Calcule les embeddings du document.
    """
    id:str
    group_key:Optional[str]=None
    metadata:dict={}
    text:list=[]
    chunks:list=[]
    embeddings:list=[]
    
    def to_payloads(self) -> Tuple[List[str], List[dict]]:
        """
        Retourne les payloads du document.

        Returns:
            Tuple[List[str], List[dict]]: Un tuple contenant les IDs et payloads du document.
        """
        payloads = []
        ids = []
        for chunk in self.chunks:
            payload = self.metadata
            payload.update({"text": chunk})
            # Crée l'ID du chunk en utilisant le hash pour éviter le stockage des doublons.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            payloads.append(payload)
            ids.append(chunk_id)

        return ids, payloads
    
    
    def compute_chunks(self, model: EmbeddingModelSingleton) -> "Document":
        """
        Calcule les chunks du document.

        Args:
            model (EmbeddingModelSingleton): Le modèle d'embedding à utiliser pour calculer les chunks.

        Returns:
            Document: L'objet document avec les chunks calculés.
        """

        for item in self.text:
            chunked_item = chunk_by_attention_window(
                item, model.tokenizer, max_input_size=model.max_input_length
            )

            self.chunks.extend(chunked_item)

        return self
    
    def compute_embeddings(self, model: EmbeddingModelSingleton) -> "Document":
        """
        Calcule les embeddings pour chaque chunk dans le document en utilisant le modèle d'embedding spécifié.

        Args:
            model (EmbeddingModelSingleton): Le modèle d'embedding à utiliser pour calculer les embeddings.

        Returns:
            Document: L'objet document avec les embeddings aclculés.
        """

        for chunk in self.chunks:
            embedding = model(chunk, to_list=True)

            self.embeddings.append(embedding)

        return self