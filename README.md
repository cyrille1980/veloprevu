# VéloPrévu

Prévision de la demande horaire BIXI par station à Montréal, de la donnée brute au modèle
surveillé en production. Projet intégrateur « Mise en production et déploiement de solutions IA (MLOps) ».

**Problème métier.** Des stations BIXI se retrouvent vides ou pleines. Prévoir le nombre de départs
de l'heure à venir, station par station, permet d'anticiper le rééquilibrage des vélos.

| Exigence | Réalisation |
|---|---|
| Extraction | `src/ingestion` : historique BIXI, météo Open-Meteo, GBFS ; nouvelles tentatives, manifeste SHA-256 |
| Préparation | `src/preprocessing`, `src/features` : agrégation horaire, validation Pandera, découpage chronologique |
| Modèles | Base saisonnière naïve, Ridge, LightGBM (`src/training`) |
| Suivi | MLflow Tracking + Model Registry (PostgreSQL + MinIO), alias `champion` |
| Versionnage | Git, DVC (`dvc.yaml`), configuration dans `configs/` |
| Tests | Pytest, 67 tests, couverture minimale de 80 % imposée |
| CI/CD | GitHub Actions : Ruff, tests, image Docker, déploiement de test, publication, déploiement |
| Service | FastAPI (`src/serving`), validation des entrées, clé d'API, version du modèle |
| Observabilité | Journaux JSON, Prometheus, Grafana, Loki |
| Dérive | PSI + Kolmogorov-Smirnov, rapport Evidently, 4 scénarios simulés (`src/monitoring`) |
| Bonus | Réentraînement automatique, rollback, authentification, agent de surveillance avec validation humaine |

## 1. Prérequis

- Python 3.12 ou plus récent
- Git
- Docker Desktop (ou Docker Engine + Compose v2) pour la pile complète
- Facultatif : [Ollama](https://ollama.com) pour l'agent

## 2. Installation

```bash
git clone <url-du-depot> veloprevu && cd veloprevu
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # puis remplacer toutes les valeurs
```

## 3. Premier essai sans réseau ni Docker (5 minutes)

```bash
python -m src.pipeline --source echantillon      # données synthétiques au format des sources réelles
uvicorn src.serving.api:creer_app --factory --port 8000
```

Ouvrir http://localhost:8000/docs et essayer `POST /predict`.

## 4. Données réelles

1. Ouvrir https://bixi.com/fr/donnees-ouvertes/ et copier le lien du fichier de l'année voulue.
2. Le coller dans `configs/config.yaml`, clé `sources.bixi.url_historique`, et ajuster `periode`.
3. Lancer le pipeline complet, en une commande :

```bash
python -m src.pipeline --source reel       # ou : dvc repro
```

Le schéma du CSV BIXI a changé en 2022. Les deux sont acceptés ; si BIXI renomme encore une colonne,
ajouter le nouveau nom dans `sources.bixi.colonnes`.

| Source | Organisme | Accès | Licence |
|---|---|---|---|
| Historique des déplacements | BIXI Montréal | CSV annuel, bixi.com/fr/donnees-ouvertes | Creative Commons avec attribution (à confirmer sur la page) |
| État des stations | BIXI Montréal | Flux GBFS temps réel | Données ouvertes BIXI |
| Météo horaire | Open-Meteo | API REST sans clé | CC BY 4.0, usage non commercial |

## 5. Pile complète

```bash
docker compose up -d --build
export MLFLOW_TRACKING_URI=http://localhost:5000        # Windows PowerShell : $env:MLFLOW_TRACKING_URI="..."
export VELOPREVU_API_URL=http://localhost:8000
export VELOPREVU_API_KEY=<valeur du .env>
python -m src.pipeline --source reel
curl -X POST http://localhost:8000/admin/reload -H "X-API-Key: $VELOPREVU_API_KEY"
```

| Service | Adresse |
|---|---|
| API (documentation interactive) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana (tableau « VéloPrévu : production ») | http://localhost:3000 |
| Prometheus (onglet Alerts) | http://localhost:9090 |
| MinIO | http://localhost:9001 |

## 6. Commandes utiles

| Action | Commande |
|---|---|
| Tests et couverture | `pytest` |
| Qualité du code | `ruff check . && ruff format --check .` |
| Envoyer du trafic à l'API | `python scripts/rejouer_trafic.py --scenario aucune --n 300` |
| Simuler une dérive et produire le rapport | `python -m src.monitoring.derive --scenario hiver` |
| Dérive sur le trafic réellement servi | `python -m src.monitoring.derive --journal` |
| Détection puis réentraînement automatique | `python -m src.monitoring.reentrainement --scenario hiver` |
| Versions du modèle | `python -m src.training.registre lister` |
| Rollback | `python -m src.training.registre rollback` puis `POST /admin/reload` |
| Agent de surveillance | `python -m src.llm_agents.agent "Y a-t-il un incident ?"` |
| Valider une proposition de l'agent | `python -m src.llm_agents.valider approuver <id>` |

Scénarios de dérive : `aucune` (contrôle), `hiver` (dérive des données), `capteur` (température bloquée à 0 °C),
`concept` (mêmes entrées, demande divisée par deux).

## 7. Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -H "X-API-Key: $VELOPREVU_API_KEY" \
  -d '{"station": "<nom exact, voir GET /model>", "horodatage": "2025-07-15T17:00:00",
       "temperature": 24.5, "precipitation": 0, "vent": 12,
       "demande_h_1": 9, "demande_h_2": 7, "demande_h_24": 10, "demande_h_168": 11, "moyenne_24h": 4.2}'
```

Réponse : `{"id_prediction": "...", "prediction": 11.01, "version_modele": "1", "unite": "départs par heure"}`

## 8. Déroulé de la démonstration

1. `python -m src.pipeline --source reel` : extraction, préparation, entraînement.
2. MLflow : comparer les runs, montrer la version `champion` dans le registre.
3. `pytest`, puis un `git push` et l'exécution dans l'onglet Actions de GitHub.
4. `/docs` : une prédiction valide, une entrée invalide (422), une station inconnue.
5. `python scripts/rejouer_trafic.py --scenario aucune` : Grafana se remplit.
6. `python scripts/rejouer_trafic.py --scenario hiver` puis `python -m src.monitoring.derive --journal` :
   l'alerte de dérive apparaît dans Grafana et Prometheus ; ouvrir le rapport HTML dans `reports/derive/`.
7. `python -m src.monitoring.reentrainement --journal` : version 2 promue, API rechargée.
8. `python -m src.training.registre rollback` : retour à la version 1.
9. Agent : rapport d'incident, proposition, validation humaine.

## 9. Structure

```
configs/            configuration (aucun secret) et scénarios d'évaluation de l'agent
data/               raw, processed, features (hors Git, suivis par DVC)
notebooks/          exploration et expérimentation
src/
  common/           configuration, journalisation JSON
  ingestion/        extraction et échantillon synthétique
  preprocessing/    nettoyage, agrégation, validation
  features/         variables et découpage temporel
  training/         entraînement, registre, rollback
  evaluation/       métriques et analyse des erreurs
  serving/          API FastAPI
  monitoring/       dérive, simulation, réentraînement
  llm_agents/       agent de surveillance, outils, validation humaine, évaluation
tests/              tests unitaires et d'intégration
monitoring/         Prometheus, alertes, Grafana, Promtail
scripts/            trafic rejoué, test de fumée
.github/workflows/  pipeline CI/CD
```

## 10. Limites connues

- L'API reçoit les retards de demande dans la requête ; elle ne les calcule pas à partir du flux GBFS.
- Après un rollback, le jeu de référence de la dérive reste celui du dernier modèle promu.
- Les scénarios de dérive sont simulés ; un hiver réel de données BIXI les remplacerait avantageusement.
- La sélection des stations les plus actives porte sur toute la période étudiée.
- L'image Docker tourne avec l'utilisateur root ; à durcir pour une vraie production.

## 11. Sources externes et usage de l'IA générative

- Données : BIXI Montréal, Open-Meteo (voir section 4).
- Bibliothèques : voir `requirements.txt`.
- IA générative : <à compléter par l'équipe : ce qui a été généré, relu, modifié>.
