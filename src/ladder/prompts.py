"""PromptVariant model, YAML loading, and the pure renderer.

Variants are files, never inline strings (architecture.md §5). Editing a
template is a version bump — save a new file, the old one stays so stored
runs keep referencing it.
"""

import string
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from ladder.records import Example, RenderedRequest

PROMPTS_LIBRARY_DIR = Path(__file__).resolve().parents[2] / "prompts" / "library"

_OPTION_LETTERS = string.ascii_uppercase  # "A".."Z" — enough for any MC benchmark here


class PromptVariant(BaseModel):
    """A versioned prompt template loaded from `prompts/library/`.

    Attributes:
        id: Identifier for this variant (matches its file path under the library dir).
        task_family: Evaluation family the template targets.
        template: The prompt template string, formatted with `str.format` fields.
        continuation_style: For "mc", whether continuations are option letters
            or full option text. None for non-"mc" families.
        num_fewshot: Number of few-shot examples the template expects/embeds.
        max_new_tokens: For "generative", the generation token budget
            (architecture.md §6 — the variant, not the evaluator, owns this
            since it's a property of the prompt format/answer style, e.g. how
            much room a chain-of-thought answer cue needs). None for
            non-"generative" families.
        stop: For "generative", stop sequences ending generation early. None
            for non-"generative" families.
    """

    id: str
    task_family: Literal["mc", "cloze", "generative"]
    template: str
    continuation_style: Literal["letter", "option_text"] | None = None
    num_fewshot: int = 0
    max_new_tokens: int | None = None
    stop: list[str] | None = None


def load_variant(variant_id: str) -> PromptVariant:
    """Load a `PromptVariant` from the prompts library by id.

    variant_id is a path relative to prompts/library/, without the .yaml suffix.

    e.g. "arc_easy/mc_letter_v1" -> prompts/library/arc_easy/mc_letter_v1.yaml

    Args:
        variant_id: Relative path (no .yaml suffix) under `PROMPTS_LIBRARY_DIR`.

    Returns:
        The parsed and validated `PromptVariant`.
    """
    path = PROMPTS_LIBRARY_DIR / f"{variant_id}.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PromptVariant.model_validate(data)


def render(example: Example, variant: PromptVariant) -> RenderedRequest:
    """Pure function (Example, PromptVariant) -> RenderedRequest.

    `task_family in {"mc", "cloze", "generative"}` are all implemented
    (architecture.md §6).

    Args:
        example: The `Example` to render a prompt for.
        variant: The `PromptVariant` template/config to render with.

    Returns:
        A `RenderedRequest` with the formatted prompt and per-option
        continuations ("mc"), or a bare prompt with no continuations and
        `gen_params` set from the variant's own generation config ("cloze",
        "generative" — the evaluator generates instead of scoring options).

    Raises:
        NotImplementedError: If `variant.task_family` is none of "mc",
            "cloze", "generative".
    """
    if variant.task_family == "cloze":
        # Pass-through template (architecture.md §5): LAMBADA's payload is
        # already the exact context to condition generation on, so the
        # template exists for provenance (recording which variant id scored
        # a run) rather than to reformat anything.
        prompt = variant.template.format(context=example.payload["context"])
        return RenderedRequest(
            example_id=example.example_id,
            prompt_variant_id=variant.id,
            kind="generate",
            prompt=prompt,
            continuations=None,
            gen_params=None,
        )

    if variant.task_family == "generative":
        prompt = variant.template.format(question=example.payload["question"])
        # Unlike cloze (whose generation budget is derived per-example from
        # the target length, evaluators.cloze), a generative variant's
        # max_new_tokens/stop are fixed properties of the prompt/answer
        # format — e.g. how much room a chain-of-thought answer cue needs —
        # so they're read straight off the variant and carried on the
        # RenderedRequest itself, not recomputed by the evaluator per example.
        gen_params = {"max_new_tokens": variant.max_new_tokens, "stop": variant.stop, "temperature": 0.0}
        return RenderedRequest(
            example_id=example.example_id,
            prompt_variant_id=variant.id,
            kind="generate",
            prompt=prompt,
            continuations=None,
            gen_params=gen_params,
        )

    if variant.task_family != "mc":
        raise NotImplementedError(f"task_family={variant.task_family!r} is not a known task_family")

    choices: list[str] = example.payload["choices"]
    lettered_choices = "\n".join(
        f"{_OPTION_LETTERS[i]}. {choice}" for i, choice in enumerate(choices)
    )
    # Question-style payloads (ARC-Easy, MMLU) carry "question"; context-completion
    # payloads (HellaSwag) carry "context" instead — dispatch on whichever key the
    # loader populated (architecture.md §4).
    format_fields = {"lettered_choices": lettered_choices}
    if "context" in example.payload:
        format_fields["context"] = example.payload["context"]
    else:
        format_fields["question"] = example.payload["question"]
    prompt = variant.template.format(**format_fields)

    if variant.continuation_style == "option_text":
        continuations = [f" {choice}" for choice in choices]
    else:  # "letter" (default for mc)
        continuations = [f" {_OPTION_LETTERS[i]}" for i in range(len(choices))]

    return RenderedRequest(
        example_id=example.example_id,
        prompt_variant_id=variant.id,
        kind="loglik",
        prompt=prompt,
        continuations=continuations,
        gen_params=None,
    )
