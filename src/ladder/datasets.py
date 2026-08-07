"""DatasetLoader ABC, registry, and all dataset loaders.

Registry access only — no module outside this file's own tests may import a
loader class directly. Always go through `get_loader(name)`.

Every loader has a two-tier source policy: Hugging Face (cached locally) or
the bundled JSONL fixture in `tests/fixtures/`. Tests use fixtures only.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Callable

from ladder.records import Example

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


class DatasetLoader(ABC):
    """Abstract interface every dataset loader implements.

    Yields dataset rows normalized to `Example`. Concrete implementations
    (e.g. `ArcEasyLoader`) are accessed only via the module-level registry
    (`get_loader`), never imported directly by other modules.

    Attributes:
        name: Registry name of the dataset (matches the `register` key).
    """

    name: str

    @abstractmethod
    def load(self, split: str, limit: int | None = None) -> Iterator[Example]:
        """Yield normalized `Example`s for the given split.

        Args:
            split: Dataset split to load (e.g. "train", "test", or "fixture"
                for the bundled offline JSONL fixture).
            limit: Maximum number of examples to yield, or None for no limit.

        Yields:
            `Example` instances in source order.
        """
        ...


_REGISTRY: dict[str, Callable[[], DatasetLoader]] = {}


def register(name: str, factory: Callable[[], DatasetLoader]) -> None:
    """Register a `DatasetLoader` factory under `name` in the module registry.

    Args:
        name: Dataset key callers will use with `get_loader`.
        factory: Zero-argument callable that constructs a `DatasetLoader`.

    Side Effects:
        Mutates the module-level `_REGISTRY` dict.
    """
    _REGISTRY[name] = factory


def get_loader(name: str) -> DatasetLoader:
    """Look up and construct a registered `DatasetLoader`.

    Args:
        name: Dataset key previously passed to `register`.

    Returns:
        A new `DatasetLoader` instance for the given dataset.

    Raises:
        KeyError: If `name` has no registered factory.
    """
    if name not in _REGISTRY:
        raise KeyError(f"No loader registered for dataset={name!r}. Known datasets: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def _load_fixture_jsonl(path: Path, split: str, limit: int | None) -> Iterator[Example]:
    """Load `Example`s from a bundled offline JSONL fixture file.

    Args:
        path: Path to the fixture JSONL file, one `Example` per line.
        split: Split label to pass through (fixtures don't encode their own split).
        limit: Maximum number of examples to yield, or None for no limit.

    Yields:
        `Example` instances parsed from the file, in line order.
    """
    with open(path, encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if limit is not None and count >= limit:
                return
            yield Example.model_validate_json(line)
            count += 1


class ArcEasyLoader(DatasetLoader):
    """ARC-Easy (allenai/ai2_arc, config 'ARC-Easy').

    Normalized MC payload: {question, choices, answer_index}.
    `answer_index` is found by locating the example's own `answerKey` within
    its own `choices.label` list — ARC-Easy's label alphabet is NOT fixed
    (some examples use "1".."4", others "A".."D" or "A".."E"), so the index
    must never be computed via a hardcoded letter→index map.
    """

    name = "arc_easy"

    def load(self, split: str, limit: int | None = None) -> Iterator[Example]:
        """See `DatasetLoader.load`. `split="fixture"` reads the bundled offline JSONL."""
        if split == "fixture":
            yield from _load_fixture_jsonl(FIXTURES_DIR / "arc_easy.jsonl", split, limit)
            return

        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
        count = 0
        for row in ds:
            if limit is not None and count >= limit:
                return
            yield self._to_example(row, split)
            count += 1

    @staticmethod
    def _to_example(row: dict, split: str) -> Example:
        """Convert one raw ai2_arc row into a normalized `Example`.

        Args:
            row: Raw row dict from the `allenai/ai2_arc` dataset.
            split: Split label to record on the resulting `Example`.

        Returns:
            An `Example` with payload {question, choices, answer_index}.
        """
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        answer_index = labels.index(row["answerKey"])
        return Example(
            dataset="arc_easy",
            split=split,
            example_id=f"arc_easy-{row['id']}",
            payload={
                "question": row["question"],
                "choices": texts,
                "answer_index": answer_index,
            },
        )


register("arc_easy", ArcEasyLoader)


class HellaSwagLoader(DatasetLoader):
    """HellaSwag (Rowan/hellaswag).

    Normalized MC payload: {context, choices, answer_index}. HellaSwag's
    native item is a sentence-completion task (`ctx` + `endings`), not a
    question, so it gets the `context`-keyed payload shape rather than
    `question` (architecture.md §4) — `prompts.render` dispatches on which
    key is present.
    """

    name = "hellaswag"

    def load(self, split: str, limit: int | None = None) -> Iterator[Example]:
        """See `DatasetLoader.load`. `split="fixture"` reads the bundled offline JSONL."""
        if split == "fixture":
            yield from _load_fixture_jsonl(FIXTURES_DIR / "hellaswag.jsonl", split, limit)
            return

        from datasets import load_dataset

        ds = load_dataset("Rowan/hellaswag", split=split)
        count = 0
        for row in ds:
            if limit is not None and count >= limit:
                return
            yield self._to_example(row, split)
            count += 1

    @staticmethod
    def _to_example(row: dict, split: str) -> Example:
        """Convert one raw Rowan/hellaswag row into a normalized `Example`.

        Args:
            row: Raw row dict from the `Rowan/hellaswag` dataset.
            split: Split label to record on the resulting `Example`.

        Returns:
            An `Example` with payload {context, choices, answer_index}.
        """
        return Example(
            dataset="hellaswag",
            split=split,
            example_id=f"hellaswag-{row['ind']}",
            payload={
                "context": row["ctx"],
                "choices": row["endings"],
                "answer_index": int(row["label"]),
            },
        )


register("hellaswag", HellaSwagLoader)


# Fixed 8-subject MMLU subset (architecture.md §4) — keeps the sweep small
# while spanning STEM, humanities, and social-science subjects. Iteration
# order is fixed so pooled example order is reproducible.
MMLU_SUBJECTS = [
    "astronomy",
    "college_biology",
    "college_computer_science",
    "high_school_mathematics",
    "high_school_world_history",
    "moral_scenarios",
    "professional_law",
    "world_religions",
]


class MmluLoader(DatasetLoader):
    """MMLU (cais/mmlu), fixed 8-subject subset.

    Normalized MC payload: {question, choices, answer_index, subject}.
    Pools rows across `MMLU_SUBJECTS` (in that fixed order) into a single
    stream so the loader interface stays `load(split, limit)` like every
    other loader; `subject` is kept in the payload for later per-subject
    breakdowns.
    """

    name = "mmlu"

    def load(self, split: str, limit: int | None = None) -> Iterator[Example]:
        """See `DatasetLoader.load`. `split="fixture"` reads the bundled offline JSONL."""
        if split == "fixture":
            yield from _load_fixture_jsonl(FIXTURES_DIR / "mmlu.jsonl", split, limit)
            return

        from datasets import load_dataset

        count = 0
        for subject in MMLU_SUBJECTS:
            ds = load_dataset("cais/mmlu", subject, split=split)
            for row in ds:
                if limit is not None and count >= limit:
                    return
                yield self._to_example(row, split)
                count += 1

    @staticmethod
    def _to_example(row: dict, split: str) -> Example:
        """Convert one raw cais/mmlu row into a normalized `Example`.

        Args:
            row: Raw row dict from a `cais/mmlu` subject config.
            split: Split label to record on the resulting `Example`.

        Returns:
            An `Example` with payload {question, choices, answer_index, subject}.
        """
        subject = row["subject"]
        # cais/mmlu rows carry no native row ID; hash the question text for a
        # stable, reproducible ID (Python's built-in hash() is salted per-process).
        content_hash = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()[:16]
        return Example(
            dataset="mmlu",
            split=split,
            example_id=f"mmlu-{subject}-{content_hash}",
            payload={
                "question": row["question"],
                "choices": row["choices"],
                "answer_index": row["answer"],
                "subject": subject,
            },
        )


register("mmlu", MmluLoader)
