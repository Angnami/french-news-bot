# Pipeline d'entrainement/Fine-Tuning   

Ce module permet de : 
- Charger les données (questions et réponses) ;
- Fine-tuner un modèle de base HugginFace en utilisant la méthode QLora ;
- Enregistrer les experiences dans l'outil de suivi des expériences de Comet ML et les résultats de l'inférence dans le tableau de bord de LLMOps de Comet ML ;
- Sauvegarder le meilleur modèle dans le registre de modèle de Comet ML .   

## Table des matières  

- [1. Installation](#1-install)
    - [1.1 Dépendances](#1.1-dependancies)
    - [1.2 Beam](#1.2-beam)
- [2. Utilisation](#2-usage)
    - [2.1 Entrainement](#2.1-train)
    - [2.2 Inférence](#2.2-inference)
    - [2.3 Linting & formatage](#2.3-linting--formatting)
