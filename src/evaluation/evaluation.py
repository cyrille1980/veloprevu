"""Métriques de régression et analyse des erreurs.

Le MAPE n'est pas utilisé : la demande vaut souvent 0 et le MAPE n'est alors pas défini.
Le WAPE (somme des erreurs absolues / somme de la demande) le remplace : il permet de
comparer des périodes dont le niveau de demande diffère (été contre hiver).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def metriques(y_vrai, y_pred) -> dict:
    y_vrai, y_pred = np.asarray(y_vrai, dtype=float), np.asarray(y_pred, dtype=float)
    total = float(np.abs(y_vrai).sum())
    return {
        "mae": float(mean_absolute_error(y_vrai, y_pred)),
        "rmse": float(root_mean_squared_error(y_vrai, y_pred)),
        "r2": float(r2_score(y_vrai, y_pred)) if len(y_vrai) > 1 else float("nan"),
        "wape": float(np.abs(y_vrai - y_pred).sum() / total) if total > 0 else float("nan"),
    }


def analyse_erreurs(df: pd.DataFrame, y_pred, cible: str = "demande") -> dict[str, pd.DataFrame]:
    """Erreur absolue moyenne et biais, ventilés par heure de la journée et par station."""
    d = df[["station", "heure_jour", cible]].copy()
    d["erreur"] = np.asarray(y_pred) - d[cible]
    d["erreur_abs"] = d["erreur"].abs()

    def resumer(cle: str) -> pd.DataFrame:
        return (
            d.groupby(cle)
            .agg(mae=("erreur_abs", "mean"), biais=("erreur", "mean"), demande_moy=(cible, "mean"))
            .reset_index()
        )

    return {"par_heure": resumer("heure_jour"), "par_station": resumer("station")}
