"""
metric.py — Generiske kvalitetsmetrikker til prompt-optimering.

Hver metrik tager (example, prediction) og returnerer 0.0–1.0.
Kombinér med MetricComposer for vægtet total-score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ── Typer ─────────────────────────────────────────────────────────────────────

MetricFn = Callable[["Example", "Prediction"], float]


@dataclass
class Example:
    """Træningseksempel: input + forventet output."""
    inputs: Dict[str, Any]
    expected_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """LLM-output fra et enkelt kald."""
    raw: str
    parsed: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Basis-metrikker ───────────────────────────────────────────────────────────

def exact_match(example: Example, prediction: Prediction) -> float:
    """1.0 hvis prediction.raw == expected_output (case-insensitive strip)."""
    return float(
        prediction.raw.strip().lower() == example.expected_output.strip().lower()
    )


def contains_expected(example: Example, prediction: Prediction) -> float:
    """1.0 hvis forventet output er indeholdt i prediction."""
    return float(example.expected_output.strip() in prediction.raw)


def yaml_list_metric(example: Example, prediction: Prediction) -> float:
    """
    Scorer YAML-liste output: cut_at_indices: [N, N, N]
    Baseret på om predicted antal snit matcher forventet.
    """
    raw = prediction.raw.strip()
    try:
        nums = [int(x) for x in re.findall(r"\d+", raw)]
    except Exception:
        return 0.0

    expected_raw = example.expected_output
    try:
        expected_nums = [int(x) for x in re.findall(r"\d+", expected_raw)]
    except Exception:
        return 0.0

    if not expected_nums:
        return 1.0 if not nums else 0.5

    # Score: predicted antal snit tæt på forventet
    predicted_n = len(nums)
    expected_n = len(expected_nums)
    count_score = max(0.0, 1.0 - abs(predicted_n - expected_n) / max(expected_n, 1))

    # Bonus: rigtige indices
    overlap = len(set(nums) & set(expected_nums))
    overlap_score = overlap / max(len(expected_nums), 1)

    return round(count_score * 0.6 + overlap_score * 0.4, 3)


def chunk_quality_metric(example: Example, prediction: Prediction) -> float:
    """
    Scorer chunk-boundary output mod golden reference.
    Forventer example.metadata med:
      - 'golden_chunk_count': int  — forventet antal chunks
      - 'total_tokens': int        — dokumentets token-total
      - 'max_chunk_tokens': int    — max tokens per chunk
    """
    raw = prediction.raw.strip()
    try:
        # Filtrer 0 ud — ugyldigt snit-punkt
        all_nums = [int(x) for x in re.findall(r"\d+", raw)]
        has_zero = 0 in all_nums
        nums = sorted(set(n for n in all_nums if n > 0))
    except Exception:
        nums = []
        has_zero = False

    meta = example.metadata
    golden_n = meta.get("golden_chunk_count", 6)
    total = meta.get("total_tokens", 0)
    max_tok = meta.get("max_chunk_tokens", 600)

    # Score 1: Antal chunks tæt på golden (dominerende faktor)
    predicted_chunks = len(nums) + 1
    diff = abs(predicted_chunks - golden_n)
    if diff == 0:
        count_score = 1.0
    elif diff == 1:
        count_score = 0.85
    else:
        count_score = max(0.0, 1.0 - diff / max(golden_n, 1) * 0.6)

    # Score 2: Ingen snit ved index 0 (straf — 0 er allerede fjernet fra nums)
    zero_penalty = -0.4 if has_zero else 0.0

    # Score 3: Snit er spredde
    if len(nums) >= 2 and total > 0:
        spread = (nums[-1] - nums[0]) / total
        spread_score = min(1.0, spread * 2)
    else:
        spread_score = 0.5

    total_score = count_score * 0.8 + spread_score * 0.2 + zero_penalty
    return round(max(0.0, total_score), 3)


# ── MetricComposer ────────────────────────────────────────────────────────────

class MetricComposer:
    """
    Kombinerer flere metrikker med vægte.

    Eksempel:
        composer = MetricComposer([
            (yaml_list_metric, 0.6),
            (chunk_quality_metric, 0.4),
        ])
        score = composer(example, prediction)
    """

    def __init__(self, metrics: List[tuple[MetricFn, float]]):
        total_weight = sum(w for _, w in metrics)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Vægte skal summere til 1.0, fik {total_weight:.3f}"
            )
        self.metrics = metrics

    def __call__(self, example: Example, prediction: Prediction) -> float:
        score = sum(fn(example, prediction) * w for fn, w in self.metrics)
        return round(score, 3)

    def describe(self) -> str:
        lines = ["MetricComposer:"]
        for fn, w in self.metrics:
            lines.append(f"  {fn.__name__:<30} vægt={w:.2f}")
        return "\n".join(lines)
