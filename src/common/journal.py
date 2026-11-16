"""Journalisation structurée en JSON : date, niveau, composant, événement, contexte.

Chaque ligne est un objet JSON, écrit sur la sortie standard et dans logs/veloprevu.jsonl
(fichier lu par Promtail puis centralisé dans Loki).
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Clés dont la valeur ne doit jamais apparaître dans un journal.
CLES_SENSIBLES = ("cle", "key", "secret", "token", "password", "mot_de_passe", "authorization")


def masquer(contexte: dict) -> dict:
    """Remplace la valeur des clés sensibles par ***."""
    return {k: "***" if any(s in k.lower() for s in CLES_SENSIBLES) else v for k, v in contexte.items()}


class FormatJson(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ligne = {
            "date": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "niveau": record.levelname,
            "composant": record.name,
            "evenement": record.getMessage(),
            "contexte": masquer(getattr(record, "contexte", {})),
        }
        if record.exc_info:
            ligne["erreur"] = self.formatException(record.exc_info)
        return json.dumps(ligne, ensure_ascii=False, default=str)


class Journal:
    """Petit enrobage : journal.info("evenement", cle=valeur, ...)."""

    def __init__(self, composant: str):
        self._log = logging.getLogger(composant)
        if not self._log.handlers:
            self._log.setLevel(os.getenv("VELOPREVU_NIVEAU_LOG", "INFO"))
            self._log.propagate = False
            sortie = logging.StreamHandler(sys.stdout)
            sortie.setFormatter(FormatJson())
            self._log.addHandler(sortie)
            repertoire = Path(os.getenv("VELOPREVU_LOGS", "logs"))
            try:
                repertoire.mkdir(parents=True, exist_ok=True)
                fichier = logging.FileHandler(repertoire / "veloprevu.jsonl", encoding="utf-8")
                fichier.setFormatter(FormatJson())
                self._log.addHandler(fichier)
            except OSError:
                self._log.warning("journal_fichier_indisponible")

    def info(self, evenement: str, **contexte):
        self._log.info(evenement, extra={"contexte": contexte})

    def warning(self, evenement: str, **contexte):
        self._log.warning(evenement, extra={"contexte": contexte})

    def error(self, evenement: str, **contexte):
        self._log.error(evenement, extra={"contexte": contexte})

    def exception(self, evenement: str, **contexte):
        self._log.exception(evenement, extra={"contexte": contexte})


def obtenir_journal(composant: str) -> Journal:
    return Journal(composant)
