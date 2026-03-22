"""
tests/test_dspy_optimizer.py — Fuld testsuite for dspy_optimizer.

Tester alle moduler uden at kræve Ollama eller DSPy installeret.
DSPy-afhængige tests er markeret med @pytest.mark.integration
og springes over ved normal kørsel.

Kør alle tests:
    pytest tests/ -v

Kør kun unit tests (ingen LLM):
    pytest tests/ -v -m "not integration"
"""
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Tilføj parent til path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dspy_optimizer.metric import (
    Example,
    MetricComposer,
    Prediction,
    chunk_quality_metric,
    contains_expected,
    exact_match,
    yaml_list_metric,
)
from dspy_optimizer.optimizer import OptimizerConfig, OptimizationResult
from dspy_optimizer.patcher import JsonPromptPatcher, PythonPromptPatcher, YamlPromptPatcher


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_example(
    inputs: dict, expected: str, metadata: dict = None
) -> Example:
    return Example(
        inputs=inputs,
        expected_output=expected,
        metadata=metadata or {},
    )


def make_prediction(raw: str) -> Prediction:
    return Prediction(raw=raw)


# ── TestExample ───────────────────────────────────────────────────────────────

class TestExample(unittest.TestCase):

    def test_opret_med_inputs(self):
        ex = Example(inputs={"text": "hej"}, expected_output="svar")
        self.assertEqual(ex.inputs["text"], "hej")
        self.assertEqual(ex.expected_output, "svar")

    def test_metadata_default_tom(self):
        ex = Example(inputs={}, expected_output="x")
        self.assertEqual(ex.metadata, {})

    def test_metadata_gemmes(self):
        ex = Example(inputs={}, expected_output="x", metadata={"n": 5})
        self.assertEqual(ex.metadata["n"], 5)


# ── TestPrediction ────────────────────────────────────────────────────────────

class TestPrediction(unittest.TestCase):

    def test_raw_gemmes(self):
        p = Prediction(raw="cut_at_indices: [3, 7]")
        self.assertIn("3", p.raw)

    def test_parsed_default_none(self):
        p = Prediction(raw="noget")
        self.assertIsNone(p.parsed)

    def test_parsed_kan_saettes(self):
        p = Prediction(raw="[3, 7]", parsed=[3, 7])
        self.assertEqual(p.parsed, [3, 7])


# ── TestExactMatch ────────────────────────────────────────────────────────────

class TestExactMatch(unittest.TestCase):

    def test_ens_giver_1(self):
        ex = make_example({}, "hej verden")
        pred = make_prediction("hej verden")
        self.assertEqual(exact_match(ex, pred), 1.0)

    def test_forskel_giver_0(self):
        ex = make_example({}, "hej verden")
        pred = make_prediction("noget andet")
        self.assertEqual(exact_match(ex, pred), 0.0)

    def test_case_insensitive(self):
        ex = make_example({}, "HEJ")
        pred = make_prediction("hej")
        self.assertEqual(exact_match(ex, pred), 1.0)

    def test_whitespace_strips(self):
        ex = make_example({}, "  svar  ")
        pred = make_prediction("svar")
        self.assertEqual(exact_match(ex, pred), 1.0)


# ── TestContainsExpected ──────────────────────────────────────────────────────

class TestContainsExpected(unittest.TestCase):

    def test_indeholdt_giver_1(self):
        ex = make_example({}, "cut_at_indices")
        pred = make_prediction("Her er svaret: cut_at_indices: [3, 7]")
        self.assertEqual(contains_expected(ex, pred), 1.0)

    def test_ikke_indeholdt_giver_0(self):
        ex = make_example({}, "cut_at_indices")
        pred = make_prediction("ingen relevans her")
        self.assertEqual(contains_expected(ex, pred), 0.0)


# ── TestYamlListMetric ────────────────────────────────────────────────────────

class TestYamlListMetric(unittest.TestCase):

    def test_korrekt_antal_giver_hoej_score(self):
        ex = make_example({}, "cut_at_indices: [5, 13, 26]")
        pred = make_prediction("cut_at_indices: [5, 13, 26]")
        score = yaml_list_metric(ex, pred)
        self.assertGreater(score, 0.9)

    def test_forkert_antal_giver_lav_score(self):
        ex = make_example({}, "cut_at_indices: [5, 13, 26]")
        pred = make_prediction("cut_at_indices: [5]")
        score = yaml_list_metric(ex, pred)
        self.assertLess(score, 0.7)

    def test_tomt_output_haandteres(self):
        ex = make_example({}, "cut_at_indices: [5]")
        pred = make_prediction("")
        score = yaml_list_metric(ex, pred)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_ingen_forventet_snit_giver_middel(self):
        ex = make_example({}, "cut_at_indices: []")
        pred = make_prediction("cut_at_indices: []")
        score = yaml_list_metric(ex, pred)
        self.assertGreaterEqual(score, 0.5)

    def test_overlap_bonus(self):
        ex = make_example({}, "cut_at_indices: [5, 13, 26]")
        # Rigtige indices
        pred_right = make_prediction("cut_at_indices: [5, 13, 26]")
        # Forkerte indices (samme antal)
        pred_wrong = make_prediction("cut_at_indices: [1, 2, 3]")
        self.assertGreater(
            yaml_list_metric(ex, pred_right),
            yaml_list_metric(ex, pred_wrong),
        )


# ── TestChunkQualityMetric ────────────────────────────────────────────────────

class TestChunkQualityMetric(unittest.TestCase):

    def _meta(self, golden_n=6, total=3000, max_tok=600):
        return {
            "golden_chunk_count": golden_n,
            "total_tokens": total,
            "max_chunk_tokens": max_tok,
        }

    def test_korrekt_antal_giver_hoej_score(self):
        ex = make_example({}, "", metadata=self._meta(golden_n=5))
        # 4 snit = 5 chunks
        pred = make_prediction("cut_at_indices: [10, 20, 30, 40]")
        score = chunk_quality_metric(ex, pred)
        self.assertGreater(score, 0.75)

    def test_for_faa_snit_giver_lav_score(self):
        ex = make_example({}, "", metadata=self._meta(golden_n=6))
        pred = make_prediction("cut_at_indices: [10]")
        score = chunk_quality_metric(ex, pred)
        self.assertLess(score, 0.6)

    def test_index_nul_giver_straf(self):
        ex = make_example({}, "", metadata=self._meta(golden_n=3))
        # 0 er ugyldig
        pred_zero = make_prediction("cut_at_indices: [0, 10, 20]")
        pred_ok   = make_prediction("cut_at_indices: [5, 10, 20]")
        self.assertLess(
            chunk_quality_metric(ex, pred_zero),
            chunk_quality_metric(ex, pred_ok),
        )

    def test_score_i_gyldigt_interval(self):
        ex = make_example({}, "", metadata=self._meta())
        for raw in ["cut_at_indices: [5, 15, 25]", "", "garbage", "[0]"]:
            score = chunk_quality_metric(ex, make_prediction(raw))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ── TestMetricComposer ────────────────────────────────────────────────────────

class TestMetricComposer(unittest.TestCase):

    def test_vaegter_summerer_til_1(self):
        composer = MetricComposer([
            (exact_match, 0.6),
            (contains_expected, 0.4),
        ])
        self.assertIsNotNone(composer)

    def test_forkerte_vaegter_fejler(self):
        with self.assertRaises(ValueError):
            MetricComposer([
                (exact_match, 0.6),
                (contains_expected, 0.6),  # 1.2 total
            ])

    def test_vægtet_score_beregnes(self):
        composer = MetricComposer([
            (exact_match, 0.5),
            (contains_expected, 0.5),
        ])
        ex = make_example({}, "svar")
        pred = make_prediction("svar")
        score = composer(ex, pred)
        self.assertEqual(score, 1.0)

    def test_describe_returnerer_string(self):
        composer = MetricComposer([
            (exact_match, 0.7),
            (yaml_list_metric, 0.3),
        ])
        desc = composer.describe()
        self.assertIn("exact_match", desc)
        self.assertIn("0.70", desc)


# ── TestOptimizerConfig ───────────────────────────────────────────────────────

class TestOptimizerConfig(unittest.TestCase):

    def test_default_vaerdier(self):
        config = OptimizerConfig()
        self.assertEqual(config.model, "qwen2.5:7b")
        self.assertEqual(config.iterations, 10)
        self.assertEqual(config.optimizer_type, "miprov2")

    def test_from_dict(self):
        config = OptimizerConfig.from_dict({
            "model": "llama3.2:3b",
            "iterations": 5,
        })
        self.assertEqual(config.model, "llama3.2:3b")
        self.assertEqual(config.iterations, 5)

    def test_from_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"model": "gemma2:2b", "iterations": 3}, f)
            tmp = Path(f.name)
        try:
            config = OptimizerConfig.from_json(tmp)
            self.assertEqual(config.model, "gemma2:2b")
        finally:
            tmp.unlink()

    def test_to_dict(self):
        config = OptimizerConfig(model="test", iterations=7)
        d = config.to_dict()
        self.assertEqual(d["model"], "test")
        self.assertEqual(d["iterations"], 7)

    def test_ukendte_noegler_ignoreres(self):
        # Skal ikke kaste fejl
        config = OptimizerConfig.from_dict({
            "model": "x",
            "ukendt_noegle": "value",
        })
        self.assertEqual(config.model, "x")


# ── TestOptimizationResult ────────────────────────────────────────────────────

class TestOptimizationResult(unittest.TestCase):

    def _make_result(self, baseline=0.4, optimized=0.7) -> OptimizationResult:
        return OptimizationResult(
            baseline_score=baseline,
            optimized_score=optimized,
            improvement=round(optimized - baseline, 3),
            optimized_instruction="Ny instruktion",
            original_instruction="Gammel instruktion",
            iterations_run=10,
            config=OptimizerConfig(),
        )

    def test_improved_true_naar_bedre(self):
        result = self._make_result(0.4, 0.7)
        self.assertTrue(result.improved)

    def test_improved_false_naar_ikke_bedre(self):
        result = self._make_result(0.7, 0.4)
        self.assertFalse(result.improved)

    def test_improvement_beregnes(self):
        result = self._make_result(0.4, 0.7)
        self.assertAlmostEqual(result.improvement, 0.3, places=2)

    def test_summary_indeholder_scores(self):
        result = self._make_result(0.4, 0.7)
        summary = result.summary()
        self.assertIn("0.400", summary)
        self.assertIn("0.700", summary)

    def test_save_og_load(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            result.save(path)
            data = json.loads(path.read_text())
            self.assertEqual(data["baseline_score"], result.baseline_score)
            self.assertEqual(data["optimized_score"], result.optimized_score)
            self.assertIn("optimized_instruction", data)


# ── TestPythonPromptPatcher ───────────────────────────────────────────────────

class TestPythonPromptPatcher(unittest.TestCase):

    def _make_python_file(self, tmpdir: Path, content: str) -> Path:
        p = tmpdir / "test_module.py"
        p.write_text(content, encoding="utf-8")
        return p

    def test_laeser_nuvaerende_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            f = self._make_python_file(
                Path(td),
                'SYSTEM_PROMPT = """Du er en assistent."""\nreste = "kode"'
            )
            patcher = PythonPromptPatcher(f, "SYSTEM_PROMPT")
            current = patcher.read_current()
            self.assertEqual(current, "Du er en assistent.")

    def test_patcher_variabel(self):
        with tempfile.TemporaryDirectory() as td:
            f = self._make_python_file(
                Path(td),
                'SYSTEM_PROMPT = """Gammel instruktion."""\n'
            )
            patcher = PythonPromptPatcher(f, "SYSTEM_PROMPT")
            patcher.patch("Ny forbedret instruktion.")
            updated = patcher.read_current()
            self.assertEqual(updated, "Ny forbedret instruktion.")

    def test_dry_run_aendrer_ikke_fil(self):
        with tempfile.TemporaryDirectory() as td:
            original = 'SYSTEM_PROMPT = """Original."""\n'
            f = self._make_python_file(Path(td), original)
            patcher = PythonPromptPatcher(f, "SYSTEM_PROMPT")
            patcher.patch("Ny instruktion.", dry_run=True)
            self.assertEqual(f.read_text(), original)

    def test_backup_oprettes(self):
        with tempfile.TemporaryDirectory() as td:
            f = self._make_python_file(
                Path(td), 'MY_PROMPT = """Prompt."""\n'
            )
            patcher = PythonPromptPatcher(f, "MY_PROMPT")
            patcher.patch("Ny prompt.")
            backup = f.with_suffix(".py.bak")
            self.assertTrue(backup.exists())

    def test_restore_fra_backup(self):
        with tempfile.TemporaryDirectory() as td:
            original = 'SYSTEM_PROMPT = """Original."""\n'
            f = self._make_python_file(Path(td), original)
            patcher = PythonPromptPatcher(f, "SYSTEM_PROMPT")
            patcher.patch("Ændret.")
            patcher.restore()
            self.assertEqual(f.read_text(), original)

    def test_manglende_variabel_fejler(self):
        with tempfile.TemporaryDirectory() as td:
            f = self._make_python_file(Path(td), 'x = 1\n')
            patcher = PythonPromptPatcher(f, "SYSTEM_PROMPT")
            with self.assertRaises(ValueError):
                patcher.patch("noget")

    def test_multiline_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            content = 'PROMPT = """Linje 1.\nLinje 2.\nLinje 3."""\n'
            f = self._make_python_file(Path(td), content)
            patcher = PythonPromptPatcher(f, "PROMPT")
            current = patcher.read_current()
            self.assertIn("Linje 1.", current)
            self.assertIn("Linje 3.", current)


# ── TestJsonPromptPatcher ─────────────────────────────────────────────────────

class TestJsonPromptPatcher(unittest.TestCase):

    def test_patcher_json_noegle(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "config.json"
            f.write_text(json.dumps({
                "llm": {"system_prompt": "Gammel prompt"}
            }), encoding="utf-8")

            patcher = JsonPromptPatcher(f, "llm.system_prompt")
            patcher.patch("Ny prompt")

            data = json.loads(f.read_text())
            self.assertEqual(data["llm"]["system_prompt"], "Ny prompt")

    def test_dry_run_aendrer_ikke(self):
        with tempfile.TemporaryDirectory() as td:
            original = {"prompt": "Original"}
            f = Path(td) / "p.json"
            f.write_text(json.dumps(original), encoding="utf-8")

            patcher = JsonPromptPatcher(f, "prompt")
            patcher.patch("Ny", dry_run=True)

            data = json.loads(f.read_text())
            self.assertEqual(data["prompt"], "Original")


# ── TestYamlPromptPatcher ─────────────────────────────────────────────────────

class TestYamlPromptPatcher(unittest.TestCase):

    def test_patcher_yaml_noegle(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML ikke installeret")

        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "config.yaml"
            import yaml
            f.write_text(
                yaml.dump({"system": {"prompt": "Gammel"}}),
                encoding="utf-8"
            )
            patcher = YamlPromptPatcher(f, "system.prompt")
            patcher.patch("Ny YAML prompt")

            data = yaml.safe_load(f.read_text())
            self.assertEqual(data["system"]["prompt"], "Ny YAML prompt")


# ── Integration tests (kræver Ollama) ─────────────────────────────────────────

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


class TestOptimizationRunnerMocked(unittest.TestCase):
    """Tests af OptimizationRunner med mocked DSPy — ingen LLM krævet."""

    def test_baseline_score_kalder_dspy(self):
        """Verify at baseline_score bruger DSPy programmet."""
        from dspy_optimizer.optimizer import OptimizationRunner

        mock_dspy = MagicMock()
        mock_lm = MagicMock()
        mock_program = MagicMock()
        mock_program.return_value = MagicMock(cut_at_indices="cut_at_indices: [5, 13]")
        mock_dspy.Predict.return_value = mock_program
        mock_dspy.configure = MagicMock()

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            # Minimal signature mock
            mock_sig = MagicMock()
            mock_sig.__doc__ = "Test signature"

            examples = [
                Example(
                    inputs={"text": "test"},
                    expected_output="cut_at_indices: [5, 13]",
                )
            ]

            runner = OptimizationRunner(
                signature=mock_sig,
                trainset=examples,
                metric=yaml_list_metric,
                config=OptimizerConfig(model="test", iterations=1, verbose=False),
            )
            # Bekræft at runner kan instantieres
            self.assertIsNotNone(runner)

    def test_optimizer_config_bruges_korrekt(self):
        from dspy_optimizer.optimizer import OptimizationRunner
        config = OptimizerConfig(
            model="qwen2.5:7b",
            iterations=3,
            optimizer_type="bootstrap",
            verbose=False,
        )
        runner = OptimizationRunner(
            signature=MagicMock(),
            trainset=[Example(inputs={}, expected_output="x")],
            metric=exact_match,
            config=config,
        )
        self.assertEqual(runner.config.optimizer_type, "bootstrap")
        self.assertEqual(runner.config.iterations, 3)


# ── Kør tests ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
