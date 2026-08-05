"""ARC-Easy loader: fixture-tier schema + stable IDs (architecture.md §4)."""

from ladder.datasets import ArcEasyLoader, get_loader
from ladder.records import Example


def test_get_loader_returns_arc_easy_loader():
    loader = get_loader("arc_easy")
    assert isinstance(loader, ArcEasyLoader)


def test_fixture_tier_yields_examples():
    loader = get_loader("arc_easy")
    examples = list(loader.load("fixture"))
    assert len(examples) == 20
    for ex in examples:
        assert isinstance(ex, Example)


def test_fixture_tier_schema():
    loader = get_loader("arc_easy")
    for ex in loader.load("fixture"):
        assert ex.dataset == "arc_easy"
        assert ex.split == "fixture"
        assert isinstance(ex.example_id, str) and ex.example_id.startswith("arc_easy-")
        assert set(ex.payload.keys()) == {"question", "choices", "answer_index"}
        assert isinstance(ex.payload["question"], str) and ex.payload["question"]
        choices = ex.payload["choices"]
        assert isinstance(choices, list) and 3 <= len(choices) <= 5
        assert all(isinstance(c, str) for c in choices)
        answer_index = ex.payload["answer_index"]
        assert isinstance(answer_index, int)
        assert 0 <= answer_index < len(choices)


def test_fixture_tier_stable_ids_are_unique():
    loader = get_loader("arc_easy")
    ids = [ex.example_id for ex in loader.load("fixture")]
    assert len(ids) == len(set(ids))


def test_fixture_tier_respects_limit():
    loader = get_loader("arc_easy")
    examples = list(loader.load("fixture", limit=5))
    assert len(examples) == 5


def test_fixture_tier_ids_are_stable_across_loads():
    loader = get_loader("arc_easy")
    ids_first = [ex.example_id for ex in loader.load("fixture")]
    ids_second = [ex.example_id for ex in loader.load("fixture")]
    assert ids_first == ids_second


def test_to_example_handles_non_letter_answer_key():
    # ARC-Easy's label alphabet isn't fixed: some rows use "1".."4" instead of "A".."D".
    row = {
        "id": "numeric-example",
        "question": "2 + 2 = ?",
        "choices": {"text": ["3", "4", "5", "6"], "label": ["1", "2", "3", "4"]},
        "answerKey": "2",
    }
    ex = ArcEasyLoader._to_example(row, "fixture")
    assert ex.payload["answer_index"] == 1
    assert ex.payload["choices"] == ["3", "4", "5", "6"]
