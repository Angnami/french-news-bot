.PHONY: init env-variables download embed training-data-responses-only training-data-questions-only training-data-q-and-a split-and-shuffle

# Installer poetry et les dépendances
init:
	@echo "Installation de Poetry ..."
	curl -sSL https://install.python-poetry.org | python3 -
	@echo "Configuration de Poetry: création de l'environnement virtuel dans le projet ..."
	poetry config virtualenvs.in-project true
	@echo "Installation des dépendances"
	poetry install

# Définir les variables environnementales nécessaires à l'utilisation des services externes
env-variables:
	mv example_config.py  config.py

# Télécharger les articles de presse dépuis l'API NEWSDATA et les enregister dans un fichier JSON
download:
	@echo "Téléchargement des données dépuis NEWSDATA..."
	poetry run python scripts/download_news_from_newsdata.py
	
# Exécuter le feature-pipeline et envoyer le feature dans le registre de feature
embed:
	@echo "Embedding des chunks de texte dans Qdrant..."
	poetry run python scripts/embed_news_into_qdrant.py

# Compléter les données d'entrainement en générant uniquement les réponses aux questions (avec gpt-3.5-turbo)
training-data-responses-only:
	@echo "Générer des réponses aux questions ..."
	poetry run python scripts/generate_training_data.py --only-q=false --only-a=true --q-and-a=false
# Générer de nouvelles données (questions et contextes) avec gpt-3.5-turbo à partir d'un exemple par rubrique d'information 
training-data-questions-only:
	@echo "Générer de nouvelles questions ..."
	poetry run python scripts/generate_training_data.py --only-q=true --only-a=false --q-and-a=false --new-examples-per-category=$(new_examples_per_category)
# Générer de nouvelles données (questions,contextes et réponses) avec gpt-3.5-turbo à partir d'un exemple( question, contexte) par rubrique d'information
training-data-q-and-a:
	@echo "Générer des questions et avec leurs réponses..."
	poetry run python scripts/generate_training_data.py --only-q=false --only-a=false --q-and-a=true --new-examples-per-category=$(new_examples_per_category)

# Diviser les données entre l'entrainement et l'évaluation et mélanger les résultats
split-and-shuffle:
	@echo "Répartir et mélanger les données ..."
	poetry run python scripts/train_test_split_and_shuffle.py --test-samples-per-category=$(test_samples_per_category)