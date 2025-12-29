# Verification + Evaluation

This document describes the verification and evaluation system for the research agent, implemented as part of Prompt 7.

## Overview

The verification module validates research run answers against evidence, while the evaluation harness enables systematic testing across question datasets.

## Architecture

```
contextmine_core/research/
├── verification/
│   ├── __init__.py           # Public exports
│   ├── models.py             # VerificationResult, VerificationStatus, etc.
│   └── verifier.py           # AnswerVerifier class
├── eval/
│   ├── __init__.py           # Public exports
│   ├── models.py             # EvalQuestion, EvalDataset, EvalRun, etc.
│   ├── runner.py             # EvalRunner class
│   └── metrics.py            # EvalMetrics, calculate_metrics()
└── ...existing files...
```

## Verification Module

### VerificationStatus

The overall status of a verification check:

```python
class VerificationStatus(Enum):
    PASSED = "passed"    # All checks pass
    FAILED = "failed"    # Critical issues found
    WARNING = "warning"  # Minor issues found
```

### VerificationResult

Complete verification result for a research run:

```python
@dataclass
class VerificationResult:
    status: VerificationStatus
    citations: list[CitationVerification]
    evidence_support: EvidenceSupportScore
    confidence_calibration: ConfidenceCalibration
    issues: list[str]
    verified_at: str  # ISO timestamp
```

### CitationVerification

Checks whether citations in the answer reference actual evidence:

```python
@dataclass
class CitationVerification:
    citation_id: str           # e.g., "ev-abc-001"
    found: bool                # Whether evidence exists
    evidence_snippet: str | None  # First 100 chars if found
```

### EvidenceSupportScore

Scores how well evidence supports the answer:

```python
@dataclass
class EvidenceSupportScore:
    score: float               # 0.0-1.0
    reasoning: str             # Explanation
    supporting_evidence_ids: list[str]
```

Scoring factors:
- Evidence count (more evidence = higher score)
- Average relevance scores
- Provenance diversity (multiple sources = bonus)

### ConfidenceCalibration

Compares stated confidence against evidence quality:

```python
@dataclass
class ConfidenceCalibration:
    stated_confidence: float     # From answer text
    evidence_confidence: float   # Calculated from evidence
    calibration_delta: float     # Absolute difference
    is_calibrated: bool          # Within tolerance
```

### AnswerVerifier

The main verifier class:

```python
from contextmine_core.research import AnswerVerifier

verifier = AnswerVerifier()
result = verifier.verify(run)

if result.status == VerificationStatus.PASSED:
    print("Verification passed!")
else:
    print(f"Issues: {result.issues}")
```

## Evaluation Harness

### EvalQuestion

A single evaluation question:

```python
@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_answer: str | None = None
    expected_evidence_files: list[str] | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

### EvalDataset

Collection of evaluation questions:

```python
from contextmine_core.research import EvalDataset

# Load from YAML
dataset = EvalDataset.from_yaml("eval/questions.yaml")

# Or create programmatically
dataset = EvalDataset(
    id="my-dataset",
    name="My Test Dataset",
    questions=[
        EvalQuestion(id="q1", question="What is the main entry point?"),
        EvalQuestion(id="q2", question="How does auth work?"),
    ]
)

# Filter by tags
filtered = dataset.filter_by_tags(["auth", "security"])
```

### YAML Dataset Format

```yaml
id: code-understanding-basic
name: Basic Code Understanding
description: Test questions for code understanding capabilities
questions:
  - id: q001
    question: "What is the main entry point of this application?"
    expected_evidence_files:
      - "src/main.py"
      - "app/main.py"
    tags: ["entry-point", "basic"]

  - id: q002
    question: "How does authentication work in this codebase?"
    expected_evidence_files:
      - "src/auth.py"
    tags: ["auth", "architecture"]
```

### EvalRunner

Run evaluations on a dataset:

```python
from contextmine_core.research import EvalRunner, ResearchAgent

agent = ResearchAgent(llm_provider=provider)
runner = EvalRunner(agent=agent)

# Run dataset serially
eval_run = await runner.run_dataset(dataset, max_parallel=1)

# Or in parallel
eval_run = await runner.run_dataset(dataset, max_parallel=4)

# Run a single ad-hoc question
result = await runner.run_single("What does function X do?")
```

### EvalMetrics

Aggregate metrics from an evaluation:

```python
from contextmine_core.research import calculate_metrics

metrics = calculate_metrics(eval_run.results)

print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Avg confidence: {metrics.avg_confidence:.2f}")
print(f"Verification pass rate: {metrics.verification_pass_rate:.1%}")

# Generate markdown report
print(metrics.to_report_markdown())
```

Available metrics:
| Metric | Description |
|--------|-------------|
| `success_rate` | Fraction completed without error |
| `total_questions` | Total questions evaluated |
| `successful_questions` | Questions completed successfully |
| `avg_confidence` | Average stated confidence |
| `avg_evidence_count` | Average evidence items collected |
| `avg_action_count` | Average actions taken |
| `avg_duration_seconds` | Average time per question |
| `citation_validity_rate` | Fraction of valid citations |
| `calibration_score` | How well confidence matches evidence |
| `verification_pass_rate` | Fraction passing verification |
| `avg_evidence_recall` | Average recall of expected files |

## Configuration

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `verification_require_citations` | `true` | Require citations for pass |
| `verification_min_evidence_support` | `0.5` | Min support score for pass |
| `verification_confidence_tolerance` | `0.2` | Max calibration delta |

### Environment Variables

```bash
VERIFICATION_REQUIRE_CITATIONS=true
VERIFICATION_MIN_EVIDENCE_SUPPORT=0.5
VERIFICATION_CONFIDENCE_TOLERANCE=0.2
```

## Integration with ResearchRun

The `ResearchRun` dataclass now includes an optional verification field:

```python
@dataclass
class ResearchRun:
    # ...existing fields...
    verification: VerificationResult | None = None
```

To add verification to a run:

```python
from contextmine_core.research import AnswerVerifier

verifier = AnswerVerifier()
run.verification = verifier.verify(run)

# Verification is included in serialization
data = run.to_trace_dict()
assert "verification" in data
```

## Typical Workflow

### Manual Verification

```python
from contextmine_core.research import (
    run_research,
    AnswerVerifier,
    VerificationStatus,
)

# Run research
run = await run_research("How does authentication work?")

# Verify the result
verifier = AnswerVerifier()
result = verifier.verify(run)

if result.status == VerificationStatus.FAILED:
    print("Verification failed:")
    for issue in result.issues:
        print(f"  - {issue}")
else:
    print(f"Verification {result.status.value}")
    print(f"Citation validity: {result.citation_validity_rate:.1%}")
    print(f"Evidence support: {result.evidence_support.score:.2f}")
```

### Running an Evaluation

```python
from contextmine_core.research import (
    EvalDataset,
    EvalRunner,
    ResearchAgent,
    calculate_metrics,
)
from contextmine_core.research.llm import get_research_llm_provider

# Setup
provider = get_research_llm_provider()
agent = ResearchAgent(llm_provider=provider)
runner = EvalRunner(agent=agent)

# Load and run dataset
dataset = EvalDataset.from_yaml("eval/questions.yaml")
eval_run = await runner.run_dataset(dataset)

# Calculate metrics
metrics = calculate_metrics(eval_run.results)
print(metrics.to_report_markdown())
```

## Citation Format

The verifier recognizes citation IDs in the following formats:

- `[ev-abc-001]` - Hyphen-separated
- `[ev_abc_001]` - Underscore-separated

Citations are extracted from the answer text and validated against the evidence collection.

## Confidence Extraction

The verifier extracts confidence from answer text using these patterns:

| Pattern | Extracted Value |
|---------|-----------------|
| `Confidence: high` | 0.85 |
| `Confidence: medium` | 0.60 |
| `Confidence: low` | 0.35 |
| `Confidence: 80%` | 0.80 |
| `Confidence: 0.75` | 0.75 |
| No pattern | 0.50 (default) |

## Testing

The module includes comprehensive tests:

- `test_verification.py` - Tests for citation checking, support scoring, calibration
- `test_evaluation.py` - Tests for runner, metrics calculation, dataset loading

Run tests:
```bash
uv run pytest packages/core/tests/test_verification.py -v
uv run pytest packages/core/tests/test_evaluation.py -v
```
