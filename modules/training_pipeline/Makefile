.PHONY: help install add  add_dev install_dev export_requirements  train_beam  infer_local infer_beam lint_check lint_fix format_check format_fix

# === Install ===

install:
	@echo "Installation du pipeline d'entrainement.."
	
	poetry env use $(shell which python3.10) && \
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install && poetry run pip install torch==2.2.0


install_dev: install
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install --with dev


add:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add $(package)


add_dev:
	PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry add --group dev $(package)


# === Beam ===

export_requirements:
	@echo "Exportation des dépendances..."

	if [ -f requirements.txt ]; then rm requirements.txt; fi
	poetry export -f requirements.txt --output requirements.txt --without-hashes

upload_dataset_to_beam:
	@echo "Chargement du volume des données dans Beam..."
	
	beam volume upload qa_dataset dataset


# === Training ===
dev_train_beam: export_requirements
	@echo "Exécution du pipeline d'entrainement sur Beam en utilisant la configuration de production production .."
	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_run.py:train -d '{"config_file": "configs/dev_training_config.yaml", "output_dir": "./output", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'

train_beam: export_requirements
	@echo "Exécution du pipeline d'entrainement sur Beam en utilisant la configuration de production production ..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train_run.py:train -d '{"config_file": "configs/training_config.yaml", "output_dir": "./output", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'

# === Inference ===

infer_beam: export_requirements
	@echo "Exécution du pipeline d'inférence sur Beam en utilisation la configuration de production ..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/inference_run.py:infer -d '{"config_file": "configs/inference_config.yaml", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'


# === PEP8 ===
# S'assurer d'installer les dépendances dev préalablement #

lint_check:
	@echo "Vérification des problèmes de linting..."

	poetry run ruff check .

lint_fix:
	@echo "Correction des problèmes de linting..."

	poetry run ruff check --fix .

format_check:
	@echo "Vérification des problèmes de formatage..."

	poetry run black --check .

format_fix:
	@echo "Formatage du code..."

	poetry run black .