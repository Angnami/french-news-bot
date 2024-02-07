import os
from typing import Dict

from jinaai import JinaAI
import openai
from src.logger import get_console_logger
from src.paths import DATA_DIR
from tqdm import tqdm
from config import set_environment
import json


with open(DATA_DIR.parent/"context-question-reformated.json","r") as file:
    EXAMPLES = json.load(file)

set_environment()

logger = get_console_logger(__name__)

PROMPTE_TEMPLATE = """
Vous avez toutes les actualités de la France. Je vous donne une rubrique d'information avec quelques éléments de contexte
et vous allez répondre à ma question.

# QUESTION
{QUESTION}

# CONTEXTE
{RUBRIQUE}:{CONTEXTE}

Veuillez donner une réponse précise en moins de 100 tokens à partir des informations du contexte.
"""
jinachat_api_key = os.environ["JINACHAT_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

def build_prompt(example:Dict)->str:
    return PROMPTE_TEMPLATE.format(RUBRIQUE=example["news_category"],
                                   CONTEXTE = example["news_context"],
                                   QUESTION=example["question"]
                                   )

def run(api:str='openai'):
    output = []
    if api =='jinaai':
        for example in tqdm(EXAMPLES):
            promt = build_prompt(example=example)
            logger.info(msg=f"{promt=}")
            
            
            jinaai = JinaAI(
                secrets={
                        'jinachat-secret':jinachat_api_key
                        }
                    )
            response = jinaai.generate(promt, 
                                    options={"temperature":0,
                                                "max_tokens":100
                                                }
                                    )
            response = response["output"]
            logger.info(f"{response=}")
            
            output.append({**example, "response": response})
        # Enregistrement du résultat en json
        with open(DATA_DIR/"training_data.json","w") as file:
            json.dump(output,file, indent=4)
            
    elif api =='openai':
        for example in tqdm(EXAMPLES):
            client = openai.OpenAI()
            prompt = build_prompt(example)
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

        # save output as json file

        with open(DATA_DIR / "training_data.json", "w") as f:
            json.dump(output, f, indent=4)
        
        
        
if __name__=="__main__":
    run(api='openai')