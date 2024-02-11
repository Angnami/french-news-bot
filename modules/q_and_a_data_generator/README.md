# Génération semi-automatique d'un dataset de Q&A  
# Contenu du module
Ce module contient des fonctions permettant de:
* générer un dataset des Q&A pour faire le fine-tuning d'un modèle de base;
* télécharger des données historiques à partir de l'API NewsData pour une période donnée;
* Formater les articles de presse télécharger, calculer leurs embeddings et les charger avec leurs metadonnées dans Qdrant;   
# Définition des varibles environnementales nécessaires à l'utilisation des services externes  
Le fichier example_config.py est un modèle de fichier python à compléter avec les infromations personnelles requises pour l'utilisation des services externes, les clés des API par exemple.   
Après avoir complété le fichier avec les informatons nécessaires, il suffit d'exécuter la commande:   
```bat
make set-env-var
```

