"""DatasetLoader ABC, registry, and all dataset loaders.

Registry access only — no module outside this file's own tests may import a
loader class directly. Always go through `get_loader(name)`.

Every loader has a two-tier source policy: Hugging Face (cached locally) or
the bundled JSONL fixture in `tests/fixtures/`. Tests use fixtures only.
"""

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
