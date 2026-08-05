"""All Pydantic record types. The contract between every module in `ladder`.

See architecture.md §2. Changing a field here is a deliberate decision,
not a casual edit — every other module depends on this shape.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Example(BaseModel):
    """A single dataset item, normalized to a dataset-agnostic shape.

    Produced by `ladder.datasets` loaders and consumed by `ladder.prompts.render`.
    Task-specific fields (question, choices, answer, etc.) live in `payload`
    so this record stays uniform across all datasets.

    Attributes:
        dataset: Registry name of the source dataset (e.g. "arc_easy").
        split: Dataset split the example was drawn from (e.g. "test", "fixture").
        example_id: Stable ID, either dataset-native or a content hash.
        payload: Task-specific fields (e.g. question/choices/answer_index for MC).
    """

    dataset: str
    split: str
    example_id: str  # stable ID (dataset-native or content hash)
    payload: dict  # task-specific fields


class RenderedRequest(BaseModel):
    """A prompt rendered from an `Example` and `PromptVariant`, ready to send to a `ModelClient`.

    Attributes:
        example_id: ID of the `Example` this request was rendered from.
        prompt_variant_id: ID of the `PromptVariant` used, or None for PPL runs.
        kind: Which `ModelClient` method this request targets.
        prompt: The rendered prompt text.
        continuations: One continuation per option (loglik requests only).
        gen_params: Generation parameters (dict form) for "generate" requests.
    """

    example_id: str
    prompt_variant_id: str | None  # None for PPL runs
    kind: Literal["loglik", "generate", "nll"]
    prompt: str
    continuations: list[str] | None  # loglik: one per option
    gen_params: dict | None


class Prediction(BaseModel):
    """Raw model output for a single `RenderedRequest`, before scoring.

    Attributes:
        request_hash: Hash identifying the originating `RenderedRequest`.
        model_id: Registry name of the model that produced this prediction.
        revision: Model checkpoint/revision used.
        logliks: Sum logprob per continuation (loglik requests only).
        token_nlls: Per-token NLLs, for PPL windows (nll requests only).
        generation: Generated text (generate requests only).
        n_bytes: UTF-8 byte count of the scored text, used to compute bits-per-byte.
    """

    model_config = ConfigDict(protected_namespaces=())

    request_hash: str
    model_id: str
    revision: str
    logliks: list[float] | None  # sum logprob per continuation
    token_nlls: list[float] | None  # for PPL windows
    generation: str | None
    n_bytes: int | None  # UTF-8 bytes of scored text (for bpb)


class ExampleResult(BaseModel):
    """Scored outcome for a single example within a run.

    Attributes:
        run_id: ID of the `RunRecord` this result belongs to.
        example_id: ID of the scored `Example`.
        correct: Whether the prediction was correct, or None for PPL (no notion of correctness).
        score: Numeric score for the example (e.g. 1.0/0.0 for accuracy, or an NLL-derived value).
        detail: Extra scoring detail (chosen option, extracted answer, or window NLLs).
    """

    run_id: str
    example_id: str
    correct: bool | None  # None for PPL
    score: float
    detail: dict  # chosen option / extracted answer / window nlls


class RunRecord(BaseModel):
    """Metadata and aggregate metrics for one evaluation run.

    Attributes:
        run_id: Unique identifier for this run.
        model_id: Registry name of the evaluated model.
        revision: Model checkpoint/revision used.
        dataset: Registry name of the dataset evaluated.
        split: Dataset split evaluated.
        prompt_variant_id: ID of the `PromptVariant` used, or None for PPL runs.
        evaluator: Evaluator that scored this run (loglik_mc | cloze | generative | perplexity).
        framework: Framework that executed the run ("ladder" | "lm_eval_harness").
        metrics: Aggregate metrics (e.g. accuracy, acc_norm) for the run.
        n_examples: Number of examples evaluated.
        seed: Random seed used for the run.
        code_version: Version/commit identifier of the code that produced this run.
        config: Full resolved configuration used to produce this run.
        status: Lifecycle state of the run.
        started_at: ISO 8601 timestamp when the run began, or None if not yet started.
        finished_at: ISO 8601 timestamp when the run reached "done"/"failed", or
            None while still running.
    """

    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    model_id: str
    revision: str
    dataset: str
    split: str
    prompt_variant_id: str | None
    evaluator: str  # loglik_mc | cloze | generative | perplexity
    framework: str  # "ladder" | "lm_eval_harness"
    metrics: dict[str, float]
    n_examples: int
    seed: int
    code_version: str
    config: dict  # full resolved config
    status: Literal["running", "done", "failed"]
    started_at: str | None = None
    finished_at: str | None = None
