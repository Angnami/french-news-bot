import os
from typing import Dict
from pathlib import Path

import argparse

import openai
from src.logger import get_console_logger
from src.paths import DATA_DIR
from tqdm import tqdm
from config import set_environment
import json
from unstructured.cleaners.core import group_broken_paragraphs


set_environment()

logger = get_console_logger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]

def parseargs()->argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande pour la génération des données d'entrainement.
    Returns:
        argparse.Namespace:un objet contenant les arguments analysés.
    """
    parser = argparse.ArgumentParser(description='Training data génération') 
    
    parser.add_argument(
        '--only-q',
        type=str,
        default='false',
        help='Générer uniquement les questions et contextes'
    )
    parser.add_argument(
        '--only-a',
        type=str,
        default='false',
        help='Générer uniquement les réponses'
    )
    
    parser.add_argument(
        '--q-and-a',
        type=str,
        default='true',
        help='Générer les questions et les réponses'
    )
    
    parser.add_argument(
        '--new-examples-per-category',
        type=int,
        default=15,
        help="Définir le nombre de nouveaux exemples à générer par rubrique d'information."
    )
    
    return parser.parse_args()

# Récupérer les valeurs des arguments
args = parseargs()
# Transformer les arguments en booléens
only_q = True if args.only_q.lower() =='true' else False
only_a = True if args.only_a.lower() =='true' else False
q_and_a = True if args.q_and_a.lower() =='true' else False
new_examples_num = args.new_examples_per_category

# Créer des templates de prompt
Q_PROMPT_TEMPLATE = """
Je vous donne un exemple comprenant une question concernant les actualités de la France dans le domaine {NEWS_CATEGORY} et un texte 
représentant le contexte de la question. Veuillez me générer {NEW_EXAMPLES_NUM} autres différents exemples de la même forme. Je souhaite avoir chaque exemple sous 
la forme d'un dictionnaire python avec les clés news_category, question et news_context. Merci de regrouper l'ensemble des exemples dans une liste.
# NEWS_CONTEXT
{NEWS_CONTEXT}

# QUESTION
{QUESTION}
"""


PROMPT_TEMPLATE = """
Vous avez toutes les actualités de la France. Je vous donne une rubrique d'information avec quelques éléments de contexte
et vous allez répondre à ma question.

# QUESTION
{QUESTION}

# CONTEXTE
{NEWS_CATEGORY}:{NEWS_CONTEXT}

Veuillez donner une réponse précise en moins de 100 tokens à partir des informations du contexte.
"""

def build_prompt(example:Dict,q_or_a:str)->str:
    if q_or_a =='a':
        return PROMPT_TEMPLATE.format(NEWS_CATEGORY=example["news_category"],
                                   NEWS_CONTEXT = example["news_context"],
                                   QUESTION=example["question"],
                                   )
    else:
        return Q_PROMPT_TEMPLATE.format(NEWS_CATEGORY = example['news_category'],
                                        NEWS_CONTEXT = example['news_context'],
                                        QUESTION=example["question"],
                                        NEW_EXAMPLES_NUM = new_examples_num
                                        )
        
def generate_questions()->Path:
    """
    Récupère une liste de dictionnaires contenant un exemple (question, contexte) par catégorie (rubrique d'informations) et 
    génère new_examples_num autres exemples de la même forme pour chaque catégorie. 
    Returns:
        Path:Le chemin du fichier json contenant les questions générées.
            
    """
    given_examples_path = DATA_DIR/'given_examples.json'
    # Vérifier si le fichier given_examples_path existe ou pas. 
    try:
        with open(given_examples_path,'r') as json_file:
            examples = json.load(fp=json_file) 
    except FileNotFoundError:
        logger.error(msg=f"Le fichier {given_examples_path} n'existe pas. Veuillez le créer afin\
                         de pouvoir générer de nouveaux exemples.")
    # Génération de nouveaux exemples.
    output = []  
    for example in tqdm(examples):
        client = openai.OpenAI()
        prompt = build_prompt(example=example, q_or_a='q')
        logger.info(msg=f"{prompt=}")
        response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role":"user","content":prompt}],
                        temperature=0,
                        )
        response = response.choices[0].message.content
        logger.info(msg=f"{response=}")
        output.extend(eval(group_broken_paragraphs(response)))
    # Sauvegarder le résultat sous forme de fichier json
    context_question_path = DATA_DIR / "context_question.json"
    
    with open(context_question_path, "w") as f:
        json.dump(obj=output, fp=f)
    
    return context_question_path   

def generate_answers()->None:
    
    if q_and_a:
        context_question_path = generate_questions()
    else:
        context_question_path = DATA_DIR/"context_question.json"
        
    # Vérifier si le fichier context_question_path existe déjà.
    try:
        with open(context_question_path, 'r') as json_file:
            EXAMPLES = json.load(fp=json_file)
    except FileNotFoundError:
        logger.info(msg=f"Le fichier {context_question_path} n'existe pas. Le fichier par défaut sera utilisé pour générer les réponses.")
        with open(DATA_DIR.parent/"context-question-reformated.json","r") as file:
            EXAMPLES = json.load(file)
    
    # Génération des réponses
    output = []
    for example in tqdm(EXAMPLES):
        client = openai.OpenAI()
        prompt = build_prompt(example=example,q_or_a='a')
        logger.info(f"{prompt=}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            max_tokens=100,
        )
        response = response.choices[0].message.content
        logger.info(f"{response=}")

        output.append({**example, "response": response})

    # Sauvegarder le résultat sous forme de fichier json

    with open(DATA_DIR / "training_data.json", "w") as f:
        json.dump(output, f, indent=4)
        
        
        
if __name__=="__main__": 
    if (not only_a) and (not only_q) and (not q_and_a) :
        q_and_a =True
        generate_answers()
    elif q_and_a :
        only_q = False
        only_a = False
        generate_answers()
    elif only_q :
        only_a = False
        generate_questions()
    elif only_a :
        only_q = False
        generate_answers()