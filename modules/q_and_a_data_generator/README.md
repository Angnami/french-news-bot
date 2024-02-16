# Génération semi-automatique des données d'entrainement et téléchargement des données historiques de l'API NewsData    
## Contenu du module
Ce module contient des fonctions permettant de:
* générer un dataset des Q&A pour faire le fine-tuning d'un modèle de base;
* télécharger des données historiques à partir de l'API NewsData pour une période donnée;
* Formater les articles de presse téléchargés, calculer leurs embeddings et les charger avec leurs metadonnées dans Qdrant.   
## Définition des varibles environnementales nécessaires à l'utilisation des services externes  
Le fichier example_config.py est un modèle de fichier python à compléter avec les infromations personnelles requises pour l'utilisation des services externes, les clés des API par exemple.   
Après avoir complété le fichier avec les informatons nécessaires, il suffit d'exécuter la commande:   
```bat
make env-variables
```   
## Génération des données d'entrainement   
Pour générer les données d'entrainement, il est possible d'utiliser un LLM comme gpt-3.5 et un prompt adapté au type des données désirées.  
Dans le cas de ce projet, deux prompts ont été utilisés. L'un permet de générer les questions et les éléments de contexte et l'autre sert à obtenir des réponses correspondant aux questions précédemment obtenues.   
Pour générer des questions,contextes et réponses en même temps,il suffit d'exéuter la commande suivante(on peut changer la valeur de new_examples_per_category ):   
```bat
new_examples_per_category=15 make training-data-q-and-a
```   
Pour générer des questions uniquement,il suffit d'exéuter cette commande:   
```bat
new_examples_per_category=15 make training-data-questions-only
```   

Pour générer uniquement des réponses,il suffit d'exéuter la commande suivante(on peut changer la valeur de new_examples_per_category ):   
```bat
make training-data-responses-only
```  

