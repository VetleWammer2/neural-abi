"""Machine-readable compatibility certificate model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from neuralabi import __version__
from neuralabi.status import ClaimStatus
from neuralabi.util.canonical_json import pretty_dumps


@dataclass(frozen=True)
class VerificationClaim:
    claim: str
    status: ClaimStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"claim": self.claim, "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True)
class VerificationCertificate:
    source_adapter_id: str
    target_adapter_id: str
    source_graph_hash: str
    target_graph_hash: str
    source_state_schema_hash: str
    target_state_schema_hash: str
    source_checkpoint_fingerprint: str
    target_checkpoint_fingerprint: str
    plan_hash: str
    architecture_signature: dict[str, Any]
    name_hint_mode: str
    rewrite_rules_used: tuple[str, ...]
    assumptions: tuple[str, ...]
    unsupported_regions: tuple[str, ...]
    probe_seeds: tuple[int, ...]
    input_signatures: dict[str, Any]
    numeric_tolerances: dict[str, Any]
    state_coverage: dict[str, Any]
    forward_results: tuple[dict[str, Any], ...]
    intermediate_results: tuple[dict[str, Any], ...]
    gradient_results: tuple[dict[str, Any], ...]
    roundtrip_results: tuple[dict[str, Any], ...]
    first_divergence: str | None
    claims: tuple[VerificationClaim, ...]
    verification_outcome: ClaimStatus
    generated_file_hashes: dict[str, str]
    timestamp: str
    conversion_scope: dict[str, bool] = field(
        default_factory=lambda: {"parameter_state": True, "optimizer_state": False}
    )
    parameter_coverage: dict[str, Any] = field(default_factory=dict)
    optimizer_coverage: dict[str, Any] = field(default_factory=dict)
    optimizer_associations: tuple[dict[str, Any], ...] = ()
    optimizer_state_results: tuple[dict[str, Any], ...] = ()
    resume_results: tuple[dict[str, Any], ...] = ()
    numeric_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "neuralabi_version": __version__,
            "pytorch_version": torch.__version__,
            "source_adapter_id": self.source_adapter_id,
            "target_adapter_id": self.target_adapter_id,
            "source_graph_hash": self.source_graph_hash,
            "target_graph_hash": self.target_graph_hash,
            "source_state_schema_hash": self.source_state_schema_hash,
            "target_state_schema_hash": self.target_state_schema_hash,
            "source_checkpoint_fingerprint": self.source_checkpoint_fingerprint,
            "target_checkpoint_fingerprint": self.target_checkpoint_fingerprint,
            "plan_hash": self.plan_hash,
            "architecture_signature": self.architecture_signature,
            "name_hint_mode": self.name_hint_mode,
            "rewrite_rules_used": list(self.rewrite_rules_used),
            "assumptions": list(self.assumptions),
            "unsupported_regions": list(self.unsupported_regions),
            "conversion_scope": self.conversion_scope,
            "state_coverage": self.state_coverage,
            "parameter_coverage": self.parameter_coverage or self.state_coverage,
            "optimizer_coverage": self.optimizer_coverage,
            "optimizer_associations": list(self.optimizer_associations),
            "probe_seeds": list(self.probe_seeds),
            "input_signatures": self.input_signatures,
            "numeric_tolerances": self.numeric_tolerances,
            "numeric_contract": self.numeric_contract,
            "forward_results": list(self.forward_results),
            "intermediate_results": list(self.intermediate_results),
            "gradient_results": list(self.gradient_results),
            "roundtrip_results": list(self.roundtrip_results),
            "optimizer_state_results": list(self.optimizer_state_results),
            "resume_results": list(self.resume_results),
            "first_divergence": self.first_divergence,
            "claims": [claim.to_dict() for claim in self.claims],
            "verification_outcome": self.verification_outcome.value,
            "timestamp": self.timestamp,
            "generated_file_hashes": dict(sorted(self.generated_file_hashes.items())),
            "statement": "Structural compatibility under NeuralABI's documented canonicalization rules, exact coordinate-state verification where optimizer state is included, plus empirical verification over the recorded probe suite.",
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pretty_dumps(self.to_dict()), encoding="utf-8", newline="\n")


def certificate_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
