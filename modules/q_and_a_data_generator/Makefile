.PHONY: init set-env-var download embed training-data

# Installer poetry et les dépendances
init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

# Définir les variables environnementales nécessaires à l'utilisation des services externes
set-env-var:
	mv /example_config.py  /config.py

# Télécharger les articles de presse dépuis l'API NEWSDATA et les enregister dans un fichier JSON
download:
	@echo "Téléchargement des données dépuis NEWSDATA..."
	poetry run python scripts/download_news_from_newsdata.py
	
# Exécuter le feature-pipeline et envoyer le feature dans le registre de feature
embed:
	@echo "Embedding des chunks de texte dans Qdrant..."
	poetry run python scripts/embed_news_into_qdrant.py

# Compléter les données d'entrainement en générant des réponses aux questions (avec gpt-3.5-turbo)
training-data:
	poetry run python scripts/generate_training_data.py