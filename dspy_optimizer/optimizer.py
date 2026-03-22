"""
optimizer.py — DSPy-baseret prompt-optimering.

Tager en LLM-opgave beskrevet som (Signature, metric, trainset)
og returnerer en optimeret instruktion klar til deployment.

Understøtter:
  - Ollama (lokal)
  - OpenAI-kompatible endpoints
  - DSPy MIPROv2 og BootstrapFewShot
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .metric import Example, Prediction


# ── Konfiguration ─────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """Konfiguration til OptimizationRunner."""
    model: str = "qwen2.5:7b"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str = "dummy"
    max_tokens: int = 64
    temperature: float = 0.0
    iterations: int = 10
    optimizer_type: str = "miprov2"   # "miprov2" | "bootstrap"
    num_threads: int = 1
    verbose: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: Path) -> "OptimizerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ── DSPy-integration ──────────────────────────────────────────────────────────

def _get_dspy():
    """Lazy import af DSPy — fejler med klar besked hvis ikke installeret."""
    try:
        import dspy
        return dspy
    except ImportError:
        raise ImportError(
            "DSPy ikke installeret.\n"
            "Installer med: pip install dspy-ai\n"
            "Dokumentation: https://dspy.ai"
        )


def build_lm(config: OptimizerConfig):
    """Bygger DSPy LM fra config. Understøtter Ollama og OpenAI-kompatible."""
    dspy = _get_dspy()

    # Ollama via LiteLLM
    if "ollama" in config.base_url or "11434" in config.base_url:
        model_str = f"ollama_chat/{config.model}"
        lm = dspy.LM(
            model_str,
            api_base=config.base_url,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    else:
        # OpenAI-kompatibel (inkl. vLLM, LM Studio, etc.)
        lm = dspy.LM(
            config.model,
            api_base=config.base_url,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    dspy.configure(lm=lm)
    return lm


# ── Resultat ──────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Resultat fra en optimerings-kørsel."""
    baseline_score: float
    optimized_score: float
    improvement: float
    optimized_instruction: str
    original_instruction: str
    iterations_run: int
    config: OptimizerConfig

    @property
    def improved(self) -> bool:
        return self.optimized_score > self.baseline_score

    def summary(self) -> str:
        arrow = "↑" if self.improved else "→"
        return (
            f"Score: {self.baseline_score:.3f} {arrow} {self.optimized_score:.3f} "
            f"(+{self.improvement:+.3f}) efter {self.iterations_run} iterationer"
        )

    def save(self, path: Path) -> None:
        """Gem resultat som JSON."""
        import dataclasses
        data = {
            "baseline_score": self.baseline_score,
            "optimized_score": self.optimized_score,
            "improvement": self.improvement,
            "optimized_instruction": self.optimized_instruction,
            "original_instruction": self.original_instruction,
            "iterations_run": self.iterations_run,
            "config": self.config.to_dict(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── OptimizationRunner ────────────────────────────────────────────────────────

class OptimizationRunner:
    """
    Kører DSPy prompt-optimering givet:
      - En DSPy Signature-klasse
      - Et sæt træningseksempler (List[Example])
      - En metrik-funktion (Example, Prediction) → float
      - En OptimizerConfig

    Eksempel::

        from dspy_optimizer import OptimizationRunner, OptimizerConfig
        from dspy_optimizer.metric import chunk_quality_metric

        runner = OptimizationRunner(
            signature=MySignature,
            trainset=examples,
            metric=chunk_quality_metric,
            config=OptimizerConfig(model="qwen2.5:7b", iterations=10),
        )
        result = runner.run()
        print(result.summary())
    """

    def __init__(
        self,
        signature,             # DSPy Signature klasse
        trainset: List[Example],
        metric: Callable[[Example, Prediction], float],
        config: Optional[OptimizerConfig] = None,
    ):
        self.signature = signature
        self.trainset = trainset
        self.metric = metric
        self.config = config or OptimizerConfig()

    def _wrap_metric(self):
        """Pakker vores (Example, Prediction) metric ind i DSPy's format."""
        metric = self.metric

        def dspy_metric(example, prediction, trace=None) -> float:
            # Konverter DSPy example/prediction til vores typer
            our_example = Example(
                inputs=dict(example.inputs()),
                expected_output=getattr(example, "expected_output", ""),
                metadata=dict(getattr(example, "metadata", {}) or {}),
            )
            raw = ""
            for field in prediction.__dict__:
                raw = str(getattr(prediction, field, ""))
                if raw:
                    break
            our_pred = Prediction(raw=raw)
            return metric(our_example, our_pred)

        return dspy_metric

    def _to_dspy_examples(self):
        """Konverterer vores Example liste til DSPy Examples."""
        dspy = _get_dspy()
        result = []
        for ex in self.trainset:
            d = dict(ex.inputs)
            d["expected_output"] = ex.expected_output
            if ex.metadata:
                d["metadata"] = ex.metadata
            input_keys = list(ex.inputs.keys())
            result.append(dspy.Example(**d).with_inputs(*input_keys))
        return result

    def baseline_score(self) -> float:
        """Beregn baseline score uden optimering."""
        dspy = _get_dspy()
        lm = build_lm(self.config)
        program = dspy.Predict(self.signature)
        dspy_examples = self._to_dspy_examples()
        dspy_metric = self._wrap_metric()

        scores = []
        for ex in dspy_examples:
            try:
                pred = program(**dict(ex.inputs()))
                scores.append(dspy_metric(ex, pred))
            except Exception:
                scores.append(0.0)

        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def run(self) -> OptimizationResult:
        """Kør optimering og returner OptimizationResult."""
        dspy = _get_dspy()
        config = self.config

        if config.verbose:
            print(f"\n{'='*56}")
            print(f"  DSPy Prompt Optimizer")
            print(f"{'='*56}")
            print(f"  Model:      {config.model}")
            print(f"  Optimizer:  {config.optimizer_type}")
            print(f"  Iterationer:{config.iterations}")
            print(f"  Eksempler:  {len(self.trainset)}")
            print(f"{'='*56}\n")

        lm = build_lm(config)
        program = dspy.Predict(self.signature)
        dspy_examples = self._to_dspy_examples()
        dspy_metric = self._wrap_metric()

        # Baseline
        if config.verbose:
            print("Beregner baseline...")
        baseline = self.baseline_score()
        if config.verbose:
            print(f"  Baseline score: {baseline:.3f}\n")

        # Udtræk original instruktion
        original_instruction = ""
        try:
            state = program.dump_state()
            for k, v in state.items():
                if "instructions" in str(v).lower():
                    original_instruction = str(v)
                    break
        except Exception:
            original_instruction = self.signature.__doc__ or ""

        # Kør optimizer
        if config.verbose:
            print(f"Starter {config.optimizer_type.upper()} optimering "
                  f"({config.iterations} iterationer)...")

        if config.optimizer_type == "bootstrap":
            optimizer = dspy.BootstrapFewShot(
                metric=dspy_metric,
                max_bootstrapped_demos=3,
                max_labeled_demos=3,
            )
            optimized = optimizer.compile(program, trainset=dspy_examples)
        else:
            # MIPROv2 (default)
            optimizer = dspy.MIPROv2(
                metric=dspy_metric,
                auto="light",
                num_threads=config.num_threads,
                verbose=config.verbose,
            )
            optimized = optimizer.compile(
                program,
                trainset=dspy_examples,
                num_trials=config.iterations,
                minibatch=False,
                requires_permission_to_run=False,
            )

        # Evaluer optimeret program
        opt_scores = []
        for ex in dspy_examples:
            try:
                pred = optimized(**dict(ex.inputs()))
                opt_scores.append(dspy_metric(ex, pred))
            except Exception:
                opt_scores.append(0.0)

        opt_score = round(sum(opt_scores) / len(opt_scores), 3) if opt_scores else 0.0

        # Udtræk optimeret instruktion
        optimized_instruction = ""
        try:
            state = optimized.dump_state()
            for k, v in state.items():
                if isinstance(v, dict) and "instructions" in v:
                    optimized_instruction = v["instructions"]
                    break
            if not optimized_instruction:
                for k, v in state.items():
                    if "instruction" in k.lower():
                        optimized_instruction = str(v)
                        break
        except Exception:
            optimized_instruction = original_instruction

        result = OptimizationResult(
            baseline_score=baseline,
            optimized_score=opt_score,
            improvement=round(opt_score - baseline, 3),
            optimized_instruction=optimized_instruction,
            original_instruction=original_instruction,
            iterations_run=config.iterations,
            config=config,
        )

        if config.verbose:
            print(f"\n{result.summary()}")

        return result
