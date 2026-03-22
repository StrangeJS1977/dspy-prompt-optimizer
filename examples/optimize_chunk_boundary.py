"""
examples/optimize_chunk_boundary.py
————————————————————————————————————
Eksempel: Optimer chunk-boundary prompt til semantisk_chunking.

Viser hvordan dspy_optimizer bruges med en eksisterende LLM-pipeline
der producerer YAML cut_at_indices output.

Krav:
  - Ollama kørende med qwen2.5:7b
  - dspy-ai installeret: pip install dspy-ai
  - semantisk_chunking projektet i ../semantic chunk/semantisk_chunking/

Kørsel:
  python examples/optimize_chunk_boundary.py --iterations 10
  python examples/optimize_chunk_boundary.py --dry-run
  python examples/optimize_chunk_boundary.py --score-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Tilføj dspy_optimizer til path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dspy_optimizer import (
    Example,
    MetricComposer,
    OptimizationRunner,
    OptimizerConfig,
    PythonPromptPatcher,
    chunk_quality_metric,
    yaml_list_metric,
)

# Sti til semantisk_chunking projektet
CHUNKER_ROOT = Path(__file__).parent.parent.parent / "semantic chunk" / "semantisk_chunking"
CHUNK_PDF_PY = CHUNKER_ROOT / "chunk_pdf.py"


# ── Signature ─────────────────────────────────────────────────────────────────

def make_chunk_boundary_signature():
    """Lav DSPy Signature til chunk-boundary opgaven."""
    try:
        import dspy
    except ImportError:
        raise ImportError("Installer DSPy: pip install dspy-ai")

    class ChunkBoundarySignature(dspy.Signature):
        """
        INSTRUCTION: You are a YAML generator. Output EXACTLY one line of YAML.

        YOUR ONLY JOB:
        - Read the unit list and detected heading positions
        - Choose which positions to cut at
        - Output EXACTLY: cut_at_indices: [N, N, N]

        RULES:
        - Cut ONLY at [§], [KAP] or [H] tagged positions
        - Never cut inside a list item or Stk.
        - Each chunk must be <= max_chunk_tokens (see cum= column)
        """
        unit_overview: str = dspy.InputField(
            desc="Text units: idx | tok | cum=cumulative | pg | tag | first line"
        )
        max_chunk_tokens: int = dspy.InputField(
            desc="Maximum tokens allowed per chunk"
        )
        ideal_chunks: int = dspy.InputField(
            desc="Target number of output chunks"
        )
        cut_at_indices: str = dspy.OutputField(
            desc="YAML: cut_at_indices: [N, N, N] — indices where new chunks begin"
        )

    return ChunkBoundarySignature


# ── Træningsdata ──────────────────────────────────────────────────────────────

def build_trainset(chunker_root: Path) -> list[Example]:
    """
    Byg træningseksempler fra semantisk_chunking projektets golden reference.
    Forventer: chunker_root/out/golden_reference.chunks.json
    """
    import json

    golden_path = chunker_root / "out" / "golden_reference.chunks.json"
    if not golden_path.exists():
        print(f"ADVARSEL: {golden_path} ikke fundet")
        print("Kør: PYTHONPATH=. python golden_reference.py i semantisk_chunking/")
        return _fallback_trainset()

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    chunk_count = golden.get("chunk_count", 6)

    # Byg unit_overview fra pipeline output (dry-run)
    sys.path.insert(0, str(chunker_root))
    try:
        from chunk_pdf import _extract_unit_signature, load_yaml, run_pipeline
        import tempfile

        config = load_yaml(chunker_root / "config.yaml")
        pdf = chunker_root / "test_output" / "dansk_juridisk_test.pdf"

        if not pdf.exists():
            print(f"ADVARSEL: {pdf} ikke fundet — bruger fallback")
            return _fallback_trainset()

        with tempfile.TemporaryDirectory() as tmpdir:
            _, units, _ = run_pipeline(
                pdf_path=pdf,
                result_dir=Path(tmpdir) / "out",
                config=config,
                dry_run=True,
            )

        cumulative = 0
        overview_lines = []
        for idx, u in enumerate(units[:50]):  # max 50 units
            cumulative += u.token_count
            overview_lines.append(_extract_unit_signature(u, idx, cumulative))

        unit_overview = "\n".join(overview_lines)
        max_tok = int(config["chunking"]["max_chunk_tokens"])

    except Exception as exc:
        print(f"Pipeline fejl: {exc} — bruger fallback træningsdata")
        return _fallback_trainset()

    return [
        Example(
            inputs={
                "unit_overview": unit_overview,
                "max_chunk_tokens": max_tok,
                "ideal_chunks": chunk_count,
            },
            expected_output=f"cut_at_indices: []",  # DSPy finder de rigtige
            metadata={
                "golden_chunk_count": chunk_count,
                "total_tokens": sum(u.token_count for u in units),
                "max_chunk_tokens": max_tok,
            },
        )
    ]


def _fallback_trainset() -> list[Example]:
    """Minimal fallback når pipeline ikke er tilgængeligt."""
    unit_overview = "\n".join([
        "idx=  0 tok=  45 cum=   45 pg=1       |  CIRKULÆRE 2025",
        "idx=  1 tok=  89 cum=  134 pg=3       |  Generelle bemærkninger",
        "idx=  2 tok= 120 cum=  254 pg=5 [§ 1] |  § 1. Aftalen gælder for",
        "idx=  3 tok=  98 cum=  352 pg=5 [§ 2] |  § 2. Den ansatte har ret",
        "idx=  4 tok= 150 cum=  502 pg=6 [§ 3] |  § 3. Den opsparede frihed",
        "idx=  5 tok=  88 cum=  590 pg=7 [§ 4] |  § 4. Tidspunktet for",
        "idx=  6 tok=  67 cum=  657 pg=8 [§ 5] |  § 5. Hvis den ansatte er syg",
    ])
    return [
        Example(
            inputs={
                "unit_overview": unit_overview,
                "max_chunk_tokens": 300,
                "ideal_chunks": 3,
            },
            expected_output="cut_at_indices: [2, 4]",
            metadata={
                "golden_chunk_count": 3,
                "total_tokens": 657,
                "max_chunk_tokens": 300,
            },
        )
    ]


# ── Metric ────────────────────────────────────────────────────────────────────

def build_metric():
    """Kombineret metric for chunk-boundary opgaven."""
    return MetricComposer([
        (chunk_quality_metric, 0.7),
        (yaml_list_metric, 0.3),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimer chunk-boundary prompt med DSPy"
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--optimizer", choices=["miprov2", "bootstrap"],
                        default="miprov2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("optimization_result.json"))
    args = parser.parse_args()

    config = OptimizerConfig(
        model=args.model,
        base_url=args.base_url,
        iterations=args.iterations,
        optimizer_type=args.optimizer,
        verbose=True,
    )

    print("Bygger træningsdata...")
    trainset = build_trainset(CHUNKER_ROOT)
    print(f"  {len(trainset)} eksempel(er) klar\n")

    if args.score_only:
        # Kun baseline score
        from dspy_optimizer.optimizer import build_lm
        try:
            import dspy
        except ImportError:
            print("DSPy ikke installeret: pip install dspy-ai")
            return

        lm = build_lm(config)
        signature = make_chunk_boundary_signature()
        program = dspy.Predict(signature)
        metric = build_metric()

        scores = []
        for ex in trainset:
            try:
                pred_obj = program(**ex.inputs)
                raw = getattr(pred_obj, "cut_at_indices", "")
                from dspy_optimizer.metric import Prediction
                pred = Prediction(raw=raw)
                scores.append(metric(ex, pred))
            except Exception as e:
                print(f"Fejl: {e}")
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Baseline score: {avg:.3f}")
        return

    if args.dry_run:
        print("=== DRY-RUN MODE ===")
        print(f"Ville køre {args.iterations} iterationer med {args.model}")
        print(f"Ville patche: {CHUNK_PDF_PY}")
        if not CHUNK_PDF_PY.exists():
            print(f"  ADVARSEL: {CHUNK_PDF_PY} ikke fundet")
        return

    # Kør optimering
    signature = make_chunk_boundary_signature()
    metric = build_metric()

    runner = OptimizationRunner(
        signature=signature,
        trainset=trainset,
        metric=metric,
        config=config,
    )

    result = runner.run()
    result.save(args.output)
    print(f"\nResultat gemt: {args.output}")

    # Patch hvis forbedring
    if result.improved and result.optimized_instruction and CHUNK_PDF_PY.exists():
        print(f"\nForbedring fundet ({result.improvement:+.3f}) — patcher chunk_pdf.py...")
        patcher = PythonPromptPatcher(CHUNK_PDF_PY, "SYSTEM_PROMPT")
        patcher.patch(result.optimized_instruction)
        print("Kør ./run_test_flow.sh for at validere forbedringen.")
    elif not result.improved:
        print("\nIngen forbedring — chunk_pdf.py uændret.")
    elif not CHUNK_PDF_PY.exists():
        print(f"\nAdvarsel: {CHUNK_PDF_PY} ikke fundet — kan ikke patche automatisk.")
        print("Optimeret instruktion er gemt i result.json.")


if __name__ == "__main__":
    main()
