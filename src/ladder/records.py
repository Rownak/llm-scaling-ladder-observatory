"""All Pydantic record types. The contract between every module in `ladder`.

See architecture.md §2. Changing a field here is a deliberate decision,
not a casual edit — every other module depends on this shape.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Example(BaseModel):
    dataset: str
    split: str
    example_id: str  # stable ID (dataset-native or content hash)
    payload: dict  # task-specific fields


class RenderedRequest(BaseModel):
    example_id: str
    prompt_variant_id: str | None  # None for PPL runs
    kind: Literal["loglik", "generate", "nll"]
    prompt: str
    continuations: list[str] | None  # loglik: one per option
    gen_params: dict | None


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_hash: str
    model_id: str
    revision: str
    logliks: list[float] | None  # sum logprob per continuation
    token_nlls: list[float] | None  # for PPL windows
    generation: str | None
    n_bytes: int | None  # UTF-8 bytes of scored text (for bpb)


class ExampleResult(BaseModel):
    run_id: str
    example_id: str
    correct: bool | None  # None for PPL
    score: float
    detail: dict  # chosen option / extracted answer / window nlls


class RunRecord(BaseModel):
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
