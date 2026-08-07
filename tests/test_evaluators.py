"""loglik_mc evaluator: render -> loglikelihood -> argmax -> ExampleResult (architecture.md §6)."""

from ladder.client import get_client
from ladder.datasets import get_loader
from ladder.evaluators import loglik_mc
from ladder.metrics import acc
from ladder.prompts import load_variant
from ladder.records import ExampleResult
from ladder.storage import PredictionCache, connect


def _cache(tmp_path, name="ladder.db") -> PredictionCache:
    return PredictionCache(connect(tmp_path / name))


def test_loglik_mc_yields_one_result_per_example(tmp_path):
    loader = get_loader("arc_easy")
    examples = list(loader.load("fixture", limit=5))
    variant = load_variant("arc_easy/mc_letter_v1")
    client = get_client("dummy", seed=0)

    results = list(loglik_mc("run-1", examples, variant, client, _cache(tmp_path), "dummy", "main"))

    assert len(results) == 5
    for r, ex in zip(results, examples):
        assert isinstance(r, ExampleResult)
        assert r.run_id == "run-1"
        assert r.example_id == ex.example_id
        assert r.correct in (True, False)
        assert r.score in (0.0, 1.0)


def test_loglik_mc_detail_contains_chosen_and_answer_index(tmp_path):
    loader = get_loader("arc_easy")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("arc_easy/mc_letter_v1")
    client = get_client("dummy", seed=0)

    result = next(loglik_mc("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.detail["answer_index"] == example.payload["answer_index"]
    n_choices = len(example.payload["choices"])
    assert 0 <= result.detail["chosen_index"] < n_choices
    assert len(result.detail["logliks"]) == n_choices
    assert len(result.detail["logliks_norm"]) == n_choices
    assert result.correct == (result.detail["chosen_index"] == result.detail["answer_index"])


def test_loglik_mc_is_deterministic_given_same_seed(tmp_path):
    loader = get_loader("arc_easy")
    examples = list(loader.load("fixture", limit=5))
    variant = load_variant("arc_easy/mc_letter_v1")

    results_a = list(
        loglik_mc(
            "run-1",
            examples,
            load_variant("arc_easy/mc_letter_v1"),
            get_client("dummy", seed=0),
            _cache(tmp_path, "a.db"),
            "dummy",
            "main",
        )
    )
    results_b = list(
        loglik_mc(
            "run-1",
            examples,
            variant,
            get_client("dummy", seed=0),
            _cache(tmp_path, "b.db"),
            "dummy",
            "main",
        )
    )

    assert [r.model_dump() for r in results_a] == [r.model_dump() for r in results_b]


def test_loglik_mc_results_feed_acc_metric(tmp_path):
    loader = get_loader("arc_easy")
    examples = list(loader.load("fixture"))
    variant = load_variant("arc_easy/mc_letter_v1")
    client = get_client("dummy", seed=0)

    results = list(loglik_mc("run-1", examples, variant, client, _cache(tmp_path), "dummy", "main"))
    accuracy = acc(results)

    assert 0.0 <= accuracy <= 1.0
    expected = sum(1 for r in results if r.correct) / len(results)
    assert accuracy == expected
