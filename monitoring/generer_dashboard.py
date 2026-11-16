"""Génère monitoring/grafana/dashboards/veloprevu.json.

Le tableau de bord est décrit ici en Python (lisible, versionnable) plutôt qu'édité à la
main dans Grafana. Relancer ce script après toute modification :
    python monitoring/generer_dashboard.py
"""

import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}
_id = 0


def panneau(titre, type_, x, y, w, h, cibles, source=PROM, unite=None, **options):
    global _id
    _id += 1
    p = {
        "id": _id,
        "title": titre,
        "type": type_,
        "datasource": source,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"refId": chr(65 + i), "expr": expr, "legendFormat": legende, "datasource": source}
            for i, (expr, legende) in enumerate(cibles)
        ],
        "fieldConfig": {"defaults": {"unit": unite} if unite else {}, "overrides": []},
    }
    p.update(options)
    return p


def ligne(titre, y):
    global _id
    _id += 1
    return {"id": _id, "title": titre, "type": "row", "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "collapsed": False}


P95 = 'histogram_quantile(0.95, sum(rate(veloprevu_latence_secondes_bucket{route="/predict"}[5m])) by (le))'
P50 = 'histogram_quantile(0.50, sum(rate(veloprevu_latence_secondes_bucket{route="/predict"}[5m])) by (le))'
ERREURS = 'sum(rate(veloprevu_requetes_total{code=~"5.."}[5m])) / sum(rate(veloprevu_requetes_total[5m]))'

panneaux = [
    ligne("Monitoring technique", 0),
    panneau("API disponible", "stat", 0, 1, 4, 4, [('up{job="veloprevu-api"}', "")]),
    panneau("Requêtes par minute", "stat", 4, 1, 4, 4, [("sum(rate(veloprevu_requetes_total[5m])) * 60", "")]),
    panneau("Taux d'erreurs serveur", "stat", 8, 1, 4, 4, [(ERREURS, "")], unite="percentunit"),
    panneau("Latence p95 de /predict", "stat", 12, 1, 4, 4, [(P95, "")], unite="s"),
    panneau(
        "CPU du processus", "stat", 16, 1, 4, 4, [("rate(process_cpu_seconds_total[1m])", "")], unite="percentunit"
    ),
    panneau("Mémoire résidente", "stat", 20, 1, 4, 4, [("process_resident_memory_bytes", "")], unite="bytes"),
    panneau(
        "Requêtes par code HTTP",
        "timeseries",
        0,
        5,
        12,
        7,
        [("sum(rate(veloprevu_requetes_total[1m])) by (code) * 60", "{{code}}")],
    ),
    panneau("Latence de /predict", "timeseries", 12, 5, 12, 7, [(P50, "p50"), (P95, "p95")], unite="s"),
    ligne("Monitoring du modèle", 12),
    panneau(
        "Version du modèle servie",
        "stat",
        0,
        13,
        4,
        4,
        [("veloprevu_modele_info", "v{{version}} ({{source}})")],
        options={"textMode": "name"},
    ),
    panneau("Prédictions servies", "stat", 4, 13, 4, 4, [("sum(veloprevu_predictions_total)", "")]),
    panneau("MAE glissante", "stat", 8, 13, 4, 4, [("veloprevu_mae_glissante", "")]),
    panneau("WAPE glissant", "stat", 12, 13, 4, 4, [("veloprevu_wape_glissant", "")], unite="percentunit"),
    panneau(
        "Part des variables dérivées",
        "stat",
        16,
        13,
        4,
        4,
        [("veloprevu_derive_part_variables", "")],
        unite="percentunit",
    ),
    panneau("Alerte de dérive", "stat", 20, 13, 4, 4, [("veloprevu_derive_alerte", "")]),
    panneau(
        "Prédictions par minute, par version",
        "timeseries",
        0,
        17,
        8,
        7,
        [("sum(rate(veloprevu_predictions_total[1m])) by (version) * 60", "v{{version}}")],
    ),
    panneau(
        "Sorties : prédiction moyenne",
        "timeseries",
        8,
        17,
        8,
        7,
        [("rate(veloprevu_prediction_valeur_sum[5m]) / rate(veloprevu_prediction_valeur_count[5m])", "départs/h")],
    ),
    panneau(
        "Entrées : température et retard 1 h moyens",
        "timeseries",
        16,
        17,
        8,
        7,
        [
            (
                "rate(veloprevu_entree_temperature_sum[5m]) / rate(veloprevu_entree_temperature_count[5m])",
                "température (°C)",
            ),
            (
                "rate(veloprevu_entree_demande_h_1_sum[5m]) / rate(veloprevu_entree_demande_h_1_count[5m])",
                "demande h-1",
            ),
        ],
    ),
    panneau(
        "Performance avec valeurs réelles",
        "timeseries",
        0,
        24,
        12,
        7,
        [("veloprevu_mae_glissante", "MAE"), ("veloprevu_wape_glissant", "WAPE")],
    ),
    panneau(
        "Distribution des prédictions",
        "bargauge",
        12,
        24,
        12,
        7,
        [("sum(increase(veloprevu_prediction_valeur_bucket[15m])) by (le)", "≤ {{le}}")],
        options={"displayMode": "gradient", "orientation": "vertical"},
    ),
    ligne("Observabilité : journaux centralisés", 31),
    panneau(
        "Avertissements et erreurs",
        "logs",
        0,
        32,
        12,
        9,
        [('{job="veloprevu", niveau=~"WARNING|ERROR"}', "")],
        source=LOKI,
    ),
    panneau("Tous les événements", "logs", 12, 32, 12, 9, [('{job="veloprevu"}', "")], source=LOKI),
]

tableau = {
    "uid": "veloprevu",
    "title": "VéloPrévu : production",
    "tags": ["veloprevu", "mlops"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"},
    "panels": panneaux,
}

sortie = Path(__file__).parent / "grafana" / "dashboards" / "veloprevu.json"
sortie.parent.mkdir(parents=True, exist_ok=True)
sortie.write_text(json.dumps(tableau, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"{sortie} : {len(panneaux)} panneaux")
