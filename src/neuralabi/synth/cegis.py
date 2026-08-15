"""Bounded counterexample-guided refinement for structurally valid candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from neuralabi.formats.plan import ConversionPlan
from neuralabi.status import LinkStatus, PlanValidationError


@dataclass(frozen=True)
class CandidateEvidence:
    plan_hash: str
    passed: bool
    first_divergence: str | None
    detail: str


@dataclass(frozen=True)
class RefinementResult:
    status: LinkStatus
    survivors: tuple[ConversionPlan, ...]
    evidence: tuple[CandidateEvidence, ...]
    evaluated_count: int
    cache_hits: int


class CandidateCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], CandidateEvidence] = {}

    def get(self, plan_hash: str, probe_id: str) -> CandidateEvidence | None:
        return self._entries.get((plan_hash, probe_id))

    def put(self, plan_hash: str, probe_id: str, evidence: CandidateEvidence) -> None:
        self._entries[(plan_hash, probe_id)] = evidence


def refine_candidates(
    candidates: Sequence[ConversionPlan],
    evaluator: Callable[[ConversionPlan], CandidateEvidence],
    *,
    probe_id: str,
    max_candidates: int = 128,
    cache: CandidateCache | None = None,
) -> RefinementResult:
    if max_candidates <= 0:
        raise PlanValidationError("max_candidates must be positive")
    unique = {candidate.plan_hash: candidate for candidate in candidates}
    ordered = tuple(unique[key] for key in sorted(unique))
    if len(ordered) > max_candidates:
        raise PlanValidationError(
            f"candidate budget exhausted: {len(ordered)} candidates exceeds {max_candidates}"
        )
    selected_cache = cache or CandidateCache()
    evidence: list[CandidateEvidence] = []
    survivors: list[ConversionPlan] = []
    cache_hits = 0
    evaluated = 0
    for candidate in ordered:
        item = selected_cache.get(candidate.plan_hash, probe_id)
        if item is None:
            item = evaluator(candidate)
            if item.plan_hash != candidate.plan_hash:
                raise PlanValidationError(
                    "candidate evaluator returned evidence for a different plan"
                )
            selected_cache.put(candidate.plan_hash, probe_id, item)
            evaluated += 1
        else:
            cache_hits += 1
        evidence.append(item)
        if item.passed:
            survivors.append(candidate)
    status = (
        LinkStatus.UNSAT
        if not survivors
        else LinkStatus.UNIQUE
        if len(survivors) == 1
        else LinkStatus.AMBIGUOUS
    )
    return RefinementResult(status, tuple(survivors), tuple(evidence), evaluated, cache_hits)
