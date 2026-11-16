# Raccourcis VéloPrévu. Sous Windows sans make, les commandes équivalentes sont dans le README.
SOURCE ?= reel
SCENARIO ?= hiver
API ?= http://localhost:8000

.PHONY: installer qualite format tests pipeline api pile arret derive reentrainement rollback rejouer agent dashboard

installer:
	pip install -r requirements.txt

qualite:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

tests:
	pytest

pipeline:          ## de la donnée brute au modèle enregistré, en une commande
	python -m src.pipeline --source $(SOURCE)

api:               ## API seule, sans Docker
	uvicorn src.serving.api:creer_app --factory --port 8000

pile:              ## pile complète (MLflow, API, Prometheus, Grafana, Loki)
	docker compose up -d --build

arret:
	docker compose down

derive:            ## simulation contrôlée + rapport de dérive
	python -m src.monitoring.derive --scenario $(SCENARIO)

reentrainement:    ## détection puis réentraînement automatique si la performance est dégradée
	python -m src.monitoring.reentrainement --scenario $(SCENARIO)

rollback:          ## retour à la version précédente du modèle
	python -m src.training.registre rollback
	curl -fsS -X POST $(API)/admin/reload -H "X-API-Key: $$VELOPREVU_API_KEY"

rejouer:           ## envoie du trafic à l'API (alimente Grafana)
	python scripts/rejouer_trafic.py --scenario $(SCENARIO) --url $(API)

agent:
	python -m src.llm_agents.agent "Fais le point sur l'état du modèle en production."

dashboard:
	python monitoring/generer_dashboard.py
