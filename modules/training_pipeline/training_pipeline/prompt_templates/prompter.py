"""
Ce script définit une classe PromptTemplate qui aide dans la génération des templates conversation/prompt. Il facilite le formatage
des prompts pour l'inférence et l'entrainement en combinant différents éléments de contexte et des inputs de l'utilisateur.
"""

from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class PromptTemplate:
    """
    Une classe qui gére les templates de prompt.
    """
    name:str
    system_template:str ="{system_message}"
    context_template:str = "{news_category}:{news_context}"
    chat_history_template:str = "{chat_history}"
    question_template:str = "{question}"
    answer_template:str = "{answer}"
    system_message:str = ""
    sep:str = "\n"
    eos:str = ""

    @property
    def input_variables(self) -> List[str]:
        """ Renvoie une liste de variables d'input pour le template du prompt."""
        return ["news_category","news_context", "chat_history", "question", "answer"]
    
    @property
    def train_raw_template(self):
        """Renvoie le format du template d'entrainement"""
        system = self.system_template.format(system_message=self.system_message)
        context = f"{self.sep}{self.context_template}"
        chat_history = f"{self.sep}{self.chat_history_template}"
        question = f"{self.sep}{self.question_template}"
        answer = f"{self.sep}{self.answer_template}"

        return f"{system}{context}{chat_history}{question}{answer}{self.eos}"


    @property
    def infer_raw_template(self):
        """Renvoie le format du template d'inférence"""
        system = self.system_template.format(system_message=self.system_message)
        context = f"{self.sep}{self.context_template}"
        chat_history = f"{self.sep}{self.chat_history_template}"
        question = f"{self.sep}{self.question_template}"

        return f"{system}{context}{chat_history}{question}{self.eos}"

    def format_train(self, sample:Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """ Formate l'échantillon de données en un échantillon d'entrainement"""
        prompt = self.train_raw_template.format(
            news_category = sample["news_category"],
            news_context = sample['news_context'],
            chat_history = sample.get('chat_history',""),
            question = sample['question'],
            answer = sample['answer'],
        )
        return {"prompt":prompt, "payload":sample}
    
    def format_infer(self, sample:Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """ Formate l'échantillon de données en un échantillon de test"""
        prompt = self.train_raw_template.format(
            news_category = sample["news_category"],
            news_context = sample['news_context'],
            chat_history = sample.get('chat_history',""),
            question = sample['question'],
        )
        return {"prompt":prompt, "payload":sample}


# Registre Général des Templates
templates:Dict[str, PromptTemplate] = {}

def register_llm_template(template:PromptTemplate):
    """ Enregistre un nouveau template dans le registre général des templates """
    templates[template.name] = template

def get_llm_template(name:str)-> PromptTemplate:
    """ Récupère le template correspond à un nom donné à partir du registre général"""

    return templates[name]



##### Registre des Templates #####


register_llm_template(
    PromptTemplate(
        name='falcon',
        system_template=">>INTRODUCTION<<{system_message}",
        system_message="Vous êtes un assistant très utile, connaissant toutes les actualités de la France.",
        context_template=">>DOMAINE<<{news_category}:{news_context}",
        chat_history_template=">>RESUME<<{chat_history}",
        question_template=">>QUESTION<<{qustion}",
        answer_template=">>REPONSE<<{answer}",
        sep="\n",
        eos="<|findetexte|>",
    )
)