# Raccourcis VéloPrévu pour Windows (équivalent du Makefile).
# Usage, dans PowerShell avec l'environnement virtuel activé :
#     .\taches.ps1 tests
#     .\taches.ps1 pipeline -Source echantillon
#     .\taches.ps1 derive -Scenario hiver
# Si PowerShell refuse d'exécuter le script :
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
param(
    [Parameter(Position = 0)]
    [ValidateSet("aide", "installer", "qualite", "format", "tests", "pipeline", "api", "pile", "arret",
                 "derive", "reentrainement", "rollback", "rejouer", "agent", "dashboard", "env")]
    [string]$Tache = "aide",
    [ValidateSet("reel", "echantillon")]
    [string]$Source = "reel",
    [ValidateSet("aucune", "hiver", "capteur", "concept")]
    [string]$Scenario = "hiver",
    [string]$Api = "http://localhost:8000"
)

$env:PYTHONUTF8 = "1"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

# Arrête le script si la dernière commande externe a échoué.
function Verifier { if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

switch ($Tache) {
    "installer"      { pip install -r requirements.txt }
    "qualite"        { ruff check .; Verifier; ruff format --check . }
    "format"         { ruff check --fix .; ruff format . }
    "tests"          { pytest }
    "pipeline"       { python -m src.pipeline --source $Source }
    "api"            { uvicorn src.serving.api:creer_app --factory --port 8000 }
    "pile" {
        # Dossiers partagés avec les conteneurs, puis pile complète.
        foreach ($d in "logs", "reports", "models", "data\features") { New-Item -ItemType Directory -Force $d | Out-Null }
        docker compose up -d --build
    }
    "arret"          { docker compose down }
    "derive"         { python -m src.monitoring.derive --scenario $Scenario }
    "reentrainement" { python -m src.monitoring.reentrainement --scenario $Scenario }
    "rollback" {
        python -m src.training.registre rollback; Verifier
        Invoke-RestMethod -Method Post -Uri "$Api/admin/reload" -Headers @{ "X-API-Key" = $env:VELOPREVU_API_KEY }
    }
    "rejouer"        { python scripts/rejouer_trafic.py --scenario $Scenario --url $Api }
    "agent"          { python -m src.llm_agents.agent "Fais le point sur l'état du modèle en production." }
    "dashboard"      { python monitoring/generer_dashboard.py }
    "env" {
        # Variables du poste pour parler à la pile Docker : à lancer avec un point devant,
        #     . .\taches.ps1 env
        # pour qu'elles restent définies dans le terminal. La clé d'API est lue dans .env.
        $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
        $env:VELOPREVU_API_URL = $Api
        $ligne = Get-Content .env -ErrorAction SilentlyContinue | Where-Object { $_ -match "^VELOPREVU_API_KEY=" }
        if ($ligne) { $env:VELOPREVU_API_KEY = ($ligne -split "=", 2)[1] }
        Write-Host "MLFLOW_TRACKING_URI, VELOPREVU_API_URL et VELOPREVU_API_KEY sont définies."
    }
    default {
        Write-Host "Tâches : installer, qualite, format, tests, pipeline, api, pile, arret, derive,"
        Write-Host "         reentrainement, rollback, rejouer, agent, dashboard, env"
        Write-Host "Options : -Source reel|echantillon   -Scenario aucune|hiver|capteur|concept   -Api <url>"
    }
}
