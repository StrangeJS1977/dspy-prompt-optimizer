# dspy-prompt-optimizer

**Automatisk prompt-optimering til lokale og cloud LLMs via DSPy.**

Erstatter manuel prompt-engineering med en systematisk, evaluerings-drevet tilgang.
Du beskriver opgaven og hvad "godt output" betyder — optimizeren finder den bedste instruktion.

---

## Hvad er det?

`dspy-prompt-optimizer` er et standalone Python-bibliotek der bruger
[DSPy](https://dspy.ai) til automatisk at forbedre LLM system prompts og user prompts.

**Kernekonceptet:**

```
Nuværende prompt → LLM → output → metric → score
                                              ↓
                                    DSPy optimizer forsøger
                                    N instruktions-varianter
                                              ↓
                                    Bedste prompt → kildekoden
```

**Fordele frem for manuel prompt-engineering:**

| Manuel | dspy-prompt-optimizer |
|--------|----------------------|
| Rettelser baseret på mavefornemmelse | Rettelser baseret på målte forbedringer |
| Svær at gentage på nyt dokumenttype | Kør optimizer igen — ny baseline |
| Prompt og kode blandes | Prompts som parametre der kan optimeres |
| Én model, én prompt | Skift model → kør optimizer → ny optimal prompt |

---

## Installation

```bash
# Basis (ingen LLM-afhængighed — unit tests kører):
pip install dspy-prompt-optimizer

# Med DSPy til faktisk optimering:
pip install "dspy-prompt-optimizer[dspy]"

# Udvikling:
git clone https://github.com/jorgenstrangeolsen/dspy-prompt-optimizer
cd dspy-prompt-optimizer
pip install -e ".[dev]"
```

**Krav:**
- Python ≥ 3.10
- [Ollama](https://ollama.ai) med en model (anbefalet: `qwen2.5:7b`)
- DSPy til optimering: `pip install dspy-ai`

---

## Hurtig start

### 1. Definer din opgave som en DSPy Signature

```python
import dspy
from dspy_optimizer import OptimizationRunner, OptimizerConfig
from dspy_optimizer.metric import Example, yaml_list_metric

class MinSignature(dspy.Signature):
    """Du er en YAML-generator. Output kun én linje YAML. Intet andet."""
    dokument_tekst: str = dspy.InputField(desc="Tekst der skal opdeles")
    max_tokens: int = dspy.InputField(desc="Max tokens per sektion")
    cut_at_indices: str = dspy.OutputField(desc="YAML: cut_at_indices: [N, N]")
```

### 2. Lav træningseksempler med forventede outputs

```python
eksempler = [
    Example(
        inputs={
            "dokument_tekst": "§ 1. Formål\nNoget tekst\n§ 2. Definitioner\nMere tekst",
            "max_tokens": 200,
        },
        expected_output="cut_at_indices: [2]",
        metadata={"golden_chunk_count": 2, "total_tokens": 400, "max_chunk_tokens": 200},
    ),
]
```

### 3. Kør optimering

```python
runner = OptimizationRunner(
    signature=MinSignature,
    trainset=eksempler,
    metric=yaml_list_metric,
    config=OptimizerConfig(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        iterations=10,
    ),
)

result = runner.run()
print(result.summary())
# Score: 0.420 ↑ 0.780 (+0.360) efter 10 iterationer
```

### 4. Skriv forbedret prompt til din kildekode

```python
from dspy_optimizer import PythonPromptPatcher
from pathlib import Path

if result.improved:
    patcher = PythonPromptPatcher(
        source_file=Path("min_llm_pipeline.py"),
        variable_name="SYSTEM_PROMPT",
    )
    patcher.patch(result.optimized_instruction)
```

---

## Moduler

### `dspy_optimizer.metric` — Kvalitetsmetrikker

Alle metrikker tager `(Example, Prediction) → float` (0.0–1.0).

| Metrik | Beskrivelse |
|--------|-------------|
| `exact_match` | 1.0 hvis output matcher præcis |
| `contains_expected` | 1.0 hvis forventet string er i output |
| `yaml_list_metric` | Scorer YAML-liste output mod forventede indices |
| `chunk_quality_metric` | Scorer chunk-boundary output mod golden reference |
| `MetricComposer` | Kombinér metrikker med vægte |

```python
from dspy_optimizer.metric import MetricComposer, chunk_quality_metric, yaml_list_metric

# Vægtet kombination
metric = MetricComposer([
    (chunk_quality_metric, 0.7),
    (yaml_list_metric,     0.3),
])
```

### `dspy_optimizer.optimizer` — Optimerings-motor

```python
from dspy_optimizer import OptimizationRunner, OptimizerConfig

config = OptimizerConfig(
    model="qwen2.5:7b",              # Ollama model
    base_url="http://localhost:11434",
    iterations=10,                    # Antal optimerings-forsøg
    optimizer_type="miprov2",         # "miprov2" | "bootstrap"
    num_threads=1,                    # Parallelisme (M3 Max: 2-4)
    verbose=True,
)

# Fra JSON-fil
config = OptimizerConfig.from_json(Path("optimizer_config.json"))
```

**`OptimizationResult`:**

```python
result.baseline_score      # float: score før optimering
result.optimized_score     # float: score efter optimering
result.improvement         # float: forskel (positiv = forbedring)
result.improved            # bool
result.optimized_instruction  # str: den nye prompt-instruktion
result.summary()           # "Score: 0.42 ↑ 0.78 (+0.36) efter 10 iterationer"
result.save(Path("result.json"))
```

### `dspy_optimizer.patcher` — Prompt-injektion i kildekode

| Klasse | Formål |
|--------|--------|
| `PythonPromptPatcher` | Erstatter `VAR = """..."""` i Python-fil |
| `YamlPromptPatcher` | Erstatter nøgle i YAML config |
| `JsonPromptPatcher` | Erstatter nøgle i JSON config |

Alle patchere opretter automatisk `.bak` backup:

```python
patcher = PythonPromptPatcher(Path("pipeline.py"), "SYSTEM_PROMPT")

# Læs nuværende
current = patcher.read_current()

# Dry-run
patcher.patch(ny_prompt, dry_run=True)

# Patch
patcher.patch(ny_prompt)

# Gendan
patcher.restore()
```

---

## Eksempel: semantisk_chunking integration

Projektet inkluderer et komplet eksempel der optimerer chunk-boundary
prompts i [semantisk_chunking](https://github.com/jorgenstrangeolsen/semantisk_chunking):

```bash
# Se baseline score (ingen optimering):
python examples/optimize_chunk_boundary.py --score-only

# Kør 10 optimerings-iterationer (~2-3 min på M3 Max):
python examples/optimize_chunk_boundary.py --iterations 10

# Dry-run — hvad ville der ske:
python examples/optimize_chunk_boundary.py --dry-run

# Fuld kørsel med BootstrapFewShot i stedet for MIPROv2:
python examples/optimize_chunk_boundary.py --optimizer bootstrap --iterations 20
```

---

## Tidsestimater (Apple M3 Max, qwen2.5:7b)

| Kørsel | Iterationer | Estimeret tid |
|--------|-------------|---------------|
| Hurtig test | 3 | ~30 sek |
| Standard | 10 | ~2-3 min |
| Grundig | 50 | ~15-20 min |
| NVIDIA DGX Spark | 50 | ~3-5 min |

---

## Brug med andre projekter

`dspy-prompt-optimizer` er designet til at være **pipeline-agnostisk**.
Det kræver kun:

1. En **DSPy Signature** der beskriver din LLM-opgave
2. Et sæt **træningseksempler** med input og forventet output
3. En **metrik-funktion** der måler output-kvalitet

### Eksempel: RAG retrieval prompt

```python
class RAGSignature(dspy.Signature):
    """Retrieve the most relevant passage for the question."""
    question: str = dspy.InputField()
    passages: str = dspy.InputField(desc="Candidate passages")
    answer: str = dspy.OutputField(desc="Best matching passage")
```

### Eksempel: Klassifikation

```python
class ClassifySignature(dspy.Signature):
    """Klassificer tekst som positiv, neutral eller negativ."""
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField(
        desc="Exactly one of: positiv, neutral, negativ"
    )
```

### Eksempel: Struktureret udtræk

```python
class ExtractSignature(dspy.Signature):
    """Udtræk juridiske entiteter som JSON."""
    legal_text: str = dspy.InputField()
    entities: str = dspy.OutputField(
        desc='JSON: {"paragraphs": [], "parties": [], "dates": []}'
    )
```

---

## Tests

```bash
# Alle unit tests (ingen Ollama krævet):
pytest tests/ -v

# Med coverage:
pytest tests/ -v --cov=dspy_optimizer --cov-report=term-missing

# Kun integration tests (kræver Ollama):
pytest tests/ -v -m integration
```

**Test-oversigt:**

| Klasse | Tests | Beskrivelse |
|--------|-------|-------------|
| `TestExample` | 3 | Example dataclass |
| `TestPrediction` | 3 | Prediction dataclass |
| `TestExactMatch` | 4 | exact_match metrik |
| `TestContainsExpected` | 2 | contains_expected metrik |
| `TestYamlListMetric` | 5 | yaml_list_metric |
| `TestChunkQualityMetric` | 4 | chunk_quality_metric |
| `TestMetricComposer` | 4 | Vægtet metrik-kombination |
| `TestOptimizerConfig` | 5 | Config oprettelse og parsing |
| `TestOptimizationResult` | 5 | Resultat og persistering |
| `TestPythonPromptPatcher` | 7 | Python-fil patching |
| `TestJsonPromptPatcher` | 2 | JSON-fil patching |
| `TestYamlPromptPatcher` | 1 | YAML-fil patching |
| `TestOptimizationRunnerMocked` | 2 | Runner med mocked DSPy |
| **Total** | **47** | |

---

## Arkitektur

```
dspy_optimizer/
├── __init__.py          ← Public API
├── metric.py            ← Metrik-funktioner og MetricComposer
├── optimizer.py         ← OptimizationRunner og OptimizerConfig
└── patcher.py           ← PythonPromptPatcher, YamlPromptPatcher, JsonPromptPatcher

examples/
└── optimize_chunk_boundary.py  ← Komplet eksempel med semantisk_chunking

tests/
└── test_dspy_optimizer.py      ← 47 unit tests
```

### Dataflow

```
1. Bruger definerer:
   - Signature (hvad skal LLM gøre?)
   - Trainset (eksempler med forventet output)
   - Metric (hvad er "godt"?)

2. OptimizationRunner:
   - Beregner baseline score
   - Kører DSPy MIPROv2/BootstrapFewShot
   - Evaluerer optimeret program
   - Returnerer OptimizationResult

3. Patcher:
   - Finder SYSTEM_PROMPT/nøgle i kildekode
   - Opretter backup
   - Erstatter med optimeret instruktion
```

---

## DSPy optimizers

| Optimizer | Bedst til | Hastighed |
|-----------|-----------|-----------|
| `miprov2` | Instruktions-optimering (default) | Middel |
| `bootstrap` | Few-shot eksempler | Hurtig |

MIPROv2 (Multi-Prompt Instruction PRoposal Optimizer v2) er DSPy's
stærkeste optimizer — den genererer og evaluerer automatisk mange
instruktions-varianter og vælger den bedste.

---

## Licens

MIT — se [LICENSE](LICENSE)

---

## Relaterede projekter

- [DSPy](https://github.com/stanfordnlp/dspy) — Frameworket dette bygger på
- [semantisk_chunking](https://github.com/jorgenstrangeolsen/semantisk_chunking) — PDF semantic chunker der bruger dette bibliotek
- [Ollama](https://ollama.ai) — Lokal LLM server
