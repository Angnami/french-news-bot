import time
from typing import Any, Dict, List, Optional

import qdrant_client
from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
    group_broken_paragraphs,
    clean_extra_whitespace
    )

from news_bot.template import PromptTemplate
from news_bot.embeddings import EmbeddingModelSingleton

class StatelessMemorySequentialChain(chains.SequentialChain):
    """
    Une chaine séquentielle qui utilise une mémoire sans état pour conserver les contextes entre deux invocations.
    Cette chaine écrase les méthodes _call et prep_outputs pour charger et effacer la mémoire avant et après chaque 
    invocation, respectivement.
    
    """
    history_input_key:str='to_load_history'
    
    def _call(self,inputs:Dict[str,str],**kwargs)-> Dict[str,str]:
        """
        Ecrase _call pour charger l'historique avant l'invocation de chain.
        Cette méthode charge l'historique à partir du dictionnaire d'input et le stocke
        dans une mémoire non permanente (sans état). Elle actualise les dictionnaires d'inputs avec les valeurs
        de la mémoire et enlève la clé de l'historique d'input. Elle appelle la méthode parent _call avec
        les inputs actualisés et retourne les résultats.
        """
        to_load_history = inputs[self.history_input_key]
        
        for (human,ai) in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key :human},
                outputs={self.memory.output_key:ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)
        del inputs[self.history_input_key]
        
        return super()._call(inputs=inputs,**kwargs)
    
    def prep_outputs(
        self,
        inputs:Dict[str,str],
        outputs:Dict[str,str],
        return_only_outputs:bool=False,)-> Dict[str,str]:
        """
        Ecrase prep_outputs pour effacer la mémoire interne après chaque invocation.
        Cette méthode appelle la méthode parent prep_outputs pour obtenir les résultats, ensuite
        efface la mémoire non permanente et retire la cle de la mémoire du dictionnaire des 
        résultats. Elle retourne les résultats actualisés.
        """
        results = super().prep_outputs(inputs=inputs,outputs=outputs,return_only_outputs=return_only_outputs)
        
        #Effacer la mémoire interne
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key]=''
        
        return results
    
    
class ContextExtractorChain(Chain):
    """
    Encode la question, recherche les k plus proches vecteurs et retourne les articles de presse de la collection 
    des documents de News Data.   
    
    Attributes:
    -----------
    top_k : int
        Le nombre des meilleures correspondances à récupérer de la base de vecteurs.
    embedding_model : EmbeddingModelSingleton
        Le modèle d'embedding à utiliser pour encoder la question.
    vector_store : qdrant_client.QdrantClient
        Le stockage de vecteurs pour la recherche des correspondances.
    vector_collection : str
        Le nom de la collection de recherche dans le stockage de vecteurs. 
    """
    top_k: int = 1
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str
    
    
    @property
    def input_keys(self)-> List[str]:
        return ['news_category', 'question']
    
    @property
    def output_keys(self)->List[str]:
        return ['context']
    
    
    def _call(self,inputs:Dict[str,Any])->Dict[str,str]:
        
        _, quest_key = self.input_keys
        question_str = inputs[quest_key]

        cleaned_question = self.clean(question_str)
        # TODO: Instead of cutting the question at 'max_input_length', chunk the question in 'max_input_length' chunks,
        # pass them through the model and average the embeddings.
        cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
        embeddings = self.embedding_model(cleaned_question)
        
        matches = self.vector_store.search(
            query_vector=embeddings,
            k=self.top_k,
            collection_name=self.vector_collection,
        )
        
        context = ""
        for match in matches:
            context += match.payload["content"] + "\n"

        return {
            "context": context,
        }
    
    
    def clean(self, question: str) -> str:
        """
        Nettoie la question d'input en éliminant les caractères non désirés.

        Parameters:
        -----------
        question : str
            La question d'input à néttoyer.

        Returns:
        --------
        str
            La question nettoyée.
        """
        question = clean(question)
        question = replace_unicode_quotes(question)
        question = clean_non_ascii_chars(question)

        return question


class NewsBotQAChain(Chain):
    """
    Cette chaine personnalisée gère la génération sur le prompt donné.
    """
    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate
    
    
    @property
    def input_keys(self) -> List[str]:
        """Retourne une liste de clés d'input pour la chaine"""

        return ["context"]

    @property
    def output_keys(self) -> List[str]:
        """Retourne une liste de clés d'output pour la chaine"""

        return ["answer"]
    
    
    
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,) -> Dict[str, Any]:
        """Invoque la chaine avec les inputs donnés et retourne le output"""

        inputs = self.clean(inputs)
        prompt = self.template.format_infer(
            {
                "news_category": inputs["news_category"],
                "news_context": inputs["context"],
                "chat_history": inputs["chat_history"],
                "question": inputs["question"],
            }
        )

        start_time = time.time()
        response = self.hf_pipeline(prompt["prompt"])
        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        if run_manager:
            run_manager.on_chain_end(
                outputs={
                    "answer": response,
                },
                # TODO: Count tokens instead of using len().
                metadata={
                    "prompt": prompt["prompt"],
                    "prompt_template_variables": prompt["payload"],
                    "prompt_template": self.template.infer_raw_template,
                    "usage.prompt_tokens": len(prompt["prompt"]),
                    "usage.total_tokens": len(prompt["prompt"]) + len(response),
                    "usage.actual_new_tokens": len(response),
                    "duration_milliseconds": duration_milliseconds,
                },
            )

        return {"answer": response}
        
    def clean(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Nettoie les inputs en éliminant les espaces inutiles et regroupant les paragraphes découpés."""

        for key, input in inputs.items():
            cleaned_input = clean_extra_whitespace(input)
            cleaned_input = group_broken_paragraphs(cleaned_input)

            inputs[key] = cleaned_input

        return inputs