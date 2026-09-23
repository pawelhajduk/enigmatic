"""Custom Presidio Operator: stable numbered placeholders."""

from __future__ import annotations

from typing import ClassVar

from presidio_anonymizer.operators import Operator, OperatorType

from enigmatic.presidio_ops.mapping import STORE, SessionMapping


class PlaceholderOperator(Operator):
    """Replace detected PII with `<ENTITY_N>` tokens that are stable per session."""

    NAME: ClassVar[str] = "placeholder"

    def operate(self, text: str = "", params: dict[str, object] | None = None) -> str:
        params = params or {}
        mapping = params.get("mapping")
        session_id = params.get("session_id")
        entity_type = params.get("entity_type")
        if not isinstance(entity_type, str) or not entity_type:
            entity_type = "ENTITY"
        session_map: SessionMapping
        if isinstance(mapping, SessionMapping):
            session_map = mapping
        elif isinstance(session_id, str) and session_id:
            session_map = STORE.get(session_id)
        else:
            session_map = SessionMapping()
        return session_map.placeholder_for(entity_type, text)

    def validate(self, params: dict[str, object] | None = None) -> None:
        return None

    def operator_name(self) -> str:
        return self.NAME

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize
