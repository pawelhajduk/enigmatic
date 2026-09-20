"""Presidio Analyzer + AnonymizerEngine with the placeholder operator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from enigmatic.config import EnigmaticConfig
from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.presidio_ops.placeholder import PlaceholderOperator
from enigmatic.presidio_ops.secrets import secret_recognizers

logger = logging.getLogger("enigmatic.presidio")

DEFAULT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_ITIN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "CRYPTO",
    "MEDICAL_LICENSE",
    "URL",
    "GITHUB_TOKEN",
    "OPENAI_KEY",
    "AWS_ACCESS_KEY",
    "AWS_SECRET_KEY",
    "PEM_KEY",
]


@dataclass
class Pipeline:
    analyzer: AnalyzerEngine
    anonymizer: AnonymizerEngine
    entities: list[str]
    language: str
    score_threshold: float

    def anonymize_text(self, text: str, mapping: SessionMapping) -> str:
        if not text:
            return text
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )
        if not results:
            return text
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig(
                    PlaceholderOperator.NAME,
                    {"mapping": mapping},
                )
            },
        )
        return anonymized.text


def _build_registry() -> RecognizerRegistry:
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"])
    for recognizer in secret_recognizers():
        registry.add_recognizer(recognizer)
    return registry


@lru_cache(maxsize=1)
def _cached_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    analyzer = AnalyzerEngine(registry=_build_registry())
    anonymizer = AnonymizerEngine()
    anonymizer.add_anonymizer(PlaceholderOperator)
    return analyzer, anonymizer


def build_pipeline(config: EnigmaticConfig) -> Pipeline:
    analyzer, anonymizer = _cached_engines()
    entities = list(config.enabled_entities) or list(DEFAULT_ENTITIES)
    return Pipeline(
        analyzer=analyzer,
        anonymizer=anonymizer,
        entities=entities,
        language=config.language,
        score_threshold=config.score_threshold,
    )


def spacy_status() -> dict[str, Any]:
    try:
        import spacy

        loaded = spacy.util.is_package("en_core_web_sm")
        return {"ok": loaded, "model": "en_core_web_sm", "installed": loaded}
    except Exception as exc:  # pragma: no cover - import diagnostics
        logger.warning("spaCy unavailable: %s", exc)
        return {"ok": False, "model": "en_core_web_sm", "installed": False, "error": str(exc)}
