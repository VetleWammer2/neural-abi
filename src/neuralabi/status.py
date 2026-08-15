"""Stable status taxonomies used by the CLI and serialized reports."""

from __future__ import annotations

from enum import StrEnum


class LinkStatus(StrEnum):
    UNIQUE = "UNIQUE"
    AMBIGUOUS = "AMBIGUOUS"
    UNSAT = "UNSAT"
    UNSUPPORTED = "UNSUPPORTED"
    ARCHITECTURE_MISMATCH = "ARCHITECTURE_MISMATCH"


class ClaimStatus(StrEnum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Evidence(StrEnum):
    PROVEN_BY_STRUCTURE = "PROVEN_BY_STRUCTURE"
    CONSTRAINED = "CONSTRAINED"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"


class NeuralABIError(Exception):
    """Base error for expected, user-facing failures."""


class UnsupportedGraphError(NeuralABIError):
    """Raised when mandatory graph capture or recognition is unsupported."""


class ArchitectureMismatchError(NeuralABIError):
    """Raised when source and target logical signatures differ."""


class PlanValidationError(NeuralABIError):
    """Raised before execution when a declarative plan is invalid."""


class CheckpointError(NeuralABIError):
    """Raised for safe checkpoint schema, index, or fingerprint failures."""
