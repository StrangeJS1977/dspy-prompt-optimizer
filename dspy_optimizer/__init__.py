"""
dspy_optimizer — DSPy-baseret automatisk prompt-optimering.

Brug::

    from dspy_optimizer import OptimizationRunner, OptimizerConfig
    from dspy_optimizer.metric import chunk_quality_metric, MetricComposer
    from dspy_optimizer.patcher import PythonPromptPatcher

Hurtig start::

    import dspy
    from dspy_optimizer import OptimizationRunner, OptimizerConfig
    from dspy_optimizer.metric import Example, yaml_list_metric

    class MySignature(dspy.Signature):
        \"\"\"Du er en YAML-generator. Output kun: result: [svar]\"\"\"
        question: str = dspy.InputField()
        result: str = dspy.OutputField()

    examples = [
        Example(inputs={"question": "Hvad er 2+2?"}, expected_output="result: [4]"),
    ]

    runner = OptimizationRunner(
        signature=MySignature,
        trainset=examples,
        metric=yaml_list_metric,
        config=OptimizerConfig(model="qwen2.5:7b", iterations=5),
    )
    result = runner.run()
    print(result.summary())
"""

from .metric import (
    Example,
    MetricComposer,
    Prediction,
    chunk_quality_metric,
    contains_expected,
    exact_match,
    yaml_list_metric,
)
from .optimizer import OptimizationResult, OptimizationRunner, OptimizerConfig, build_lm
from .patcher import JsonPromptPatcher, PythonPromptPatcher, YamlPromptPatcher

__version__ = "0.1.0"
__all__ = [
    "OptimizationRunner", "OptimizerConfig", "OptimizationResult", "build_lm",
    "Example", "Prediction", "MetricComposer",
    "exact_match", "contains_expected", "yaml_list_metric", "chunk_quality_metric",
    "PythonPromptPatcher", "YamlPromptPatcher", "JsonPromptPatcher",
]
