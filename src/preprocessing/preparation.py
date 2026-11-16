"""Préparation : nettoyage des déplacements, agrégation horaire par station, jointure météo.

Sortie : data/processed/demande_horaire.parquet, une ligne par (station, heure) avec
la demande (nombre de départs) et la météo de l'heure. Le résultat est validé par un
schéma Pandera avant d'être écrit.

Usage : python -m src.preprocessing.preparation
"""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.common.config import charger_config, dossier
from src.common.journal import obtenir_journal

journal = obtenir_journal("preprocessing.preparation")

SCHEMA = pa.DataFrameSchema(
    {
        "station": pa.Column(nullable=False),
        "heure": pa.Column(nullable=False),
        "demande": pa.Column(float, pa.Check.ge(0), nullable=False),
        "temperature": pa.Column(float, pa.Check.in_range(-50, 50), nullable=False),
        "precipitation": pa.Column(float, pa.Check.ge(0), nullable=False),
        "vent": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    unique=["station", "heure"],
    strict=True,
)


def _choisir(colonnes, candidats: list[str]) -> str:
    """Retourne la première colonne candidate présente (insensible à la casse)."""
    presentes = {c.lower(): c for c in colonnes}
    for c in candidats:
        if c.lower() in presentes:
            return presentes[c.lower()]
    raise ValueError(f"Aucune des colonnes {candidats} dans {list(colonnes)}")


def _lire_csv(source, cfg: dict) -> pd.DataFrame:
    """Lit un CSV BIXI (ancien ou nouveau schéma) et le ramène à (debut, station)."""
    col = cfg["sources"]["bixi"]["colonnes"]
    entete = pd.read_csv(source, nrows=0).columns
    if hasattr(source, "seek"):
        source.seek(0)
    c_debut, c_station = _choisir(entete, col["debut"]), _choisir(entete, col["station"])
    df = pd.read_csv(source, usecols=[c_debut, c_station])
    df = df.rename(columns={c_debut: "debut", c_station: "station"})
    if pd.api.types.is_numeric_dtype(df["debut"]):
        # Nouveau schéma : millisecondes depuis l'époque, en UTC -> heure locale de Montréal.
        debut = pd.to_datetime(df["debut"], unit="ms", utc=True, errors="coerce")
        df["debut"] = debut.dt.tz_convert(cfg["preparation"]["fuseau"]).dt.tz_localize(None)
    else:
        df["debut"] = pd.to_datetime(df["debut"], errors="coerce")
    df["station"] = df["station"].astype("string").str.strip()
    return df


def lire_deplacements(cfg: dict) -> pd.DataFrame:
    repertoire = dossier(cfg, "brut") / "bixi"
    morceaux = []
    for fichier in sorted(repertoire.glob("*")):
        if fichier.suffix.lower() == ".csv":
            morceaux.append(_lire_csv(fichier, cfg))
        elif fichier.suffix.lower() == ".zip":
            with zipfile.ZipFile(fichier) as z:
                for nom in z.namelist():
                    if nom.lower().endswith(".csv") and "station" not in Path(nom).stem.lower():
                        with z.open(nom) as f:
                            morceaux.append(_lire_csv(f, cfg))
    if not morceaux:
        raise FileNotFoundError(f"Aucun fichier de déplacements dans {repertoire}")
    return pd.concat(morceaux, ignore_index=True)


def nettoyer(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Retire les lignes incomplètes, les doublons exacts et ce qui sort de la période."""
    avant = len(df)
    df = df.dropna(subset=["debut", "station"])
    df = df[df["station"] != ""].drop_duplicates()
    debut, fin = pd.Timestamp(cfg["periode"]["debut"]), pd.Timestamp(cfg["periode"]["fin"])
    df = df[(df["debut"] >= debut) & (df["debut"] < fin + pd.Timedelta(days=1))]
    journal.info("nettoyage", lignes_avant=avant, lignes_apres=len(df))
    return df


def agreger_horaire(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compte les départs par station et par heure, pour les N stations les plus actives.

    La grille complète station x heure est reconstruite : une heure sans départ vaut 0,
    ce n'est pas une valeur manquante.
    """
    n = cfg["preparation"]["nb_stations"]
    stations = df["station"].value_counts().head(n).index
    df = df[df["station"].isin(stations)].assign(heure=lambda d: d["debut"].dt.floor("h"))
    comptes = df.groupby(["station", "heure"]).size().rename("demande")
    heures = pd.date_range(df["heure"].min(), df["heure"].max(), freq="h")
    grille = pd.MultiIndex.from_product([sorted(stations), heures], names=["station", "heure"])
    demande = comptes.reindex(grille, fill_value=0).reset_index()
    demande["demande"] = demande["demande"].astype(float)
    # Valeurs aberrantes (événements exceptionnels, erreurs) : plafonnement au quantile choisi.
    plafond = demande["demande"].quantile(cfg["preparation"]["plafond_quantile"])
    nb_plafonnees = int((demande["demande"] > plafond).sum())
    demande["demande"] = demande["demande"].clip(upper=plafond)
    journal.info("agregation", stations=len(stations), heures=len(heures), plafonnees=nb_plafonnees)
    return demande


def lire_meteo(cfg: dict) -> pd.DataFrame:
    brut = json.loads((dossier(cfg, "brut") / "meteo" / "meteo.json").read_text(encoding="utf-8"))
    h = brut["hourly"]
    meteo = pd.DataFrame(
        {
            "heure": pd.to_datetime(h["time"]),
            "temperature": h["temperature_2m"],
            "precipitation": h["precipitation"],
            "vent": h["wind_speed_10m"],
        }
    ).drop_duplicates("heure")
    return meteo.astype({"temperature": float, "precipitation": float, "vent": float})


def joindre_meteo(demande: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """Jointure sur l'heure ; les trous météo sont interpolés puis comblés aux extrémités."""
    heures = pd.DataFrame({"heure": pd.date_range(demande["heure"].min(), demande["heure"].max(), freq="h")})
    meteo = heures.merge(meteo, on="heure", how="left").set_index("heure")
    manquantes = int(meteo.isna().any(axis=1).sum())
    meteo = meteo.interpolate(limit_direction="both").reset_index()
    meteo["precipitation"] = meteo["precipitation"].clip(lower=0)
    meteo["vent"] = meteo["vent"].clip(lower=0)
    journal.info("jointure_meteo", heures_meteo_manquantes=manquantes)
    return demande.merge(meteo, on="heure", how="left")


def preparer(cfg: dict) -> pd.DataFrame:
    journal.info("preparation_demarree")
    try:
        deplacements = nettoyer(lire_deplacements(cfg), cfg)
        resultat = joindre_meteo(agreger_horaire(deplacements, cfg), lire_meteo(cfg))
        resultat["station"] = resultat["station"].astype(str)
        resultat = SCHEMA.validate(resultat)
    except Exception:
        journal.exception("preparation_erreur")
        raise
    sortie = dossier(cfg, "traite") / "demande_horaire.parquet"
    resultat.to_parquet(sortie, index=False)
    journal.info("preparation_terminee", lignes=len(resultat), fichier=str(sortie))
    return resultat


if __name__ == "__main__":
    preparer(charger_config())
