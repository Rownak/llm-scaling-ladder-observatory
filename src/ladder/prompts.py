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
    """

    id: str
    task_family: Literal["mc", "cloze", "generative"]
    template: str
    continuation_style: Literal["letter", "option_text"] | None = None
    num_fewshot: int = 0


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

    Only `task_family == "mc"` is implemented in Sprint 1 (cloze/generative
    arrive with their evaluators in Sprint 3).

    Args:
        example: The `Example` to render a prompt for.
        variant: The `PromptVariant` template/config to render with.

    Returns:
        A `RenderedRequest` with the formatted prompt and per-option continuations.

    Raises:
        NotImplementedError: If `variant.task_family` is not "mc".
    """
    if variant.task_family != "mc":
        raise NotImplementedError(f"task_family={variant.task_family!r} arrives in a later sprint")

    choices: list[str] = example.payload["choices"]
    lettered_choices = "\n".join(
        f"{_OPTION_LETTERS[i]}. {choice}" for i, choice in enumerate(choices)
    )
    prompt = variant.template.format(
        question=example.payload["question"],
        lettered_choices=lettered_choices,
    )

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
