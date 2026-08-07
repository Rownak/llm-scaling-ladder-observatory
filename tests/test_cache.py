"""Prediction cache: content-hash key, get/put round-trip, evaluator wiring (architecture.md §7)."""

from ladder.client import LoglikResult, get_client
from ladder.datasets import get_loader
from ladder.evaluators import loglik_mc
from ladder.prompts import load_variant
from ladder.records import Prediction
from ladder.storage import PredictionCache, connect, get_prediction, prediction_cache_key, save_prediction


def test_prediction_cache_key_is_deterministic():
    key_a = prediction_cache_key("dummy", "main", "loglik", "prompt", [" A", " B"], None)
    key_b = prediction_cache_key("dummy", "main", "loglik", "prompt", [" A", " B"], None)
    assert key_a == key_b


def test_prediction_cache_key_changes_with_prompt():
    key_a = prediction_cache_key("dummy", "main", "loglik", "prompt one", [" A"], None)
    key_b = prediction_cache_key("dummy", "main", "loglik", "prompt two", [" A"], None)
    assert key_a != key_b


def test_prediction_cache_key_changes_with_any_field():
    base = ("dummy", "main", "loglik", "prompt", [" A"], None)
    key_base = prediction_cache_key(*base)

    assert prediction_cache_key("other-model", *base[1:]) != key_base
    assert prediction_cache_key(base[0], "step1000", *base[2:]) != key_base
    assert prediction_cache_key(*base[:2], "generate", *base[3:]) != key_base
    assert prediction_cache_key(*base[:4], [" B"], None) != key_base
    assert prediction_cache_key(*base[:5], {"temperature": 0.5}) != key_base


def test_get_put_prediction_round_trip(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    key = prediction_cache_key("dummy", "main", "loglik", "p", [" A"], None)

    assert get_prediction(conn, key) is None

    save_prediction(
        conn,
        Prediction(
            request_hash=key,
            model_id="dummy",
            revision="main",
            logliks=[-1.23],
            token_nlls=None,
            generation=None,
            n_bytes=None,
        ),
    )

    fetched = get_prediction(conn, key)
    assert fetched is not None
    assert fetched.logliks == [-1.23]
    assert fetched.model_id == "dummy"
    assert fetched.revision == "main"


def test_prediction_cache_get_counts_hits_and_misses(tmp_path):
    conn = connect(tmp_path / "ladder.db")
    cache = PredictionCache(conn)

    assert cache.get("dummy", "main", "loglik", "p", [" A"], None) is None
    assert cache.misses == 1 and cache.hits == 0

    cache.put(
        "dummy",
        "main",
        "loglik",
        "p",
        [" A"],
        None,
        Prediction(
            request_hash="",
            model_id="dummy",
            revision="main",
            logliks=[-1.0],
            token_nlls=None,
            generation=None,
            n_bytes=None,
        ),
    )

    assert cache.get("dummy", "main", "loglik", "p", [" A"], None) is not None
    assert cache.misses == 1 and cache.hits == 1


class _CountingDummyClient:
    """Wraps `DummyClient.loglikelihood`, counting calls (architecture.md §12's cache-hit spy)."""

    def __init__(self, inner):
        self._inner = inner
        self.call_count = 0

    def loglikelihood(self, prompt: str, continuations: list[str]) -> list[LoglikResult]:
        self.call_count += 1
        return self._inner.loglikelihood(prompt, continuations)

    def generate(self, prompt, params):
        return self._inner.generate(prompt, params)

    def token_nlls(self, text, context=""):
        return self._inner.token_nlls(text, context)


def test_loglik_mc_same_request_twice_makes_one_client_call(tmp_path):
    loader = get_loader("arc_easy")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("arc_easy/mc_letter_v1")
    spy = _CountingDummyClient(get_client("dummy", seed=0))
    cache = PredictionCache(connect(tmp_path / "ladder.db"))

    list(loglik_mc("run-1", [example], variant, spy, cache, "dummy", "main"))
    n_choices = len(example.payload["choices"])
    assert spy.call_count == n_choices
    assert cache.misses == n_choices
    assert cache.hits == 0

    # Re-run the identical request: every continuation should hit the cache,
    # so the client is not called again at all.
    list(loglik_mc("run-1", [example], variant, spy, cache, "dummy", "main"))
    assert spy.call_count == n_choices  # unchanged — zero new client calls
    assert cache.hits == n_choices


def test_loglik_mc_changed_prompt_is_a_cache_miss(tmp_path):
    loader = get_loader("arc_easy")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("arc_easy/mc_letter_v1")
    other_variant = load_variant("arc_easy/mc_letter_v1").model_copy(
        update={"template": "Something different: {question}\n{lettered_choices}\nAnswer:\n"}
    )
    spy = _CountingDummyClient(get_client("dummy", seed=0))
    cache = PredictionCache(connect(tmp_path / "ladder.db"))

    list(loglik_mc("run-1", [example], variant, spy, cache, "dummy", "main"))
    n_choices = len(example.payload["choices"])
    assert spy.call_count == n_choices

    # A different prompt template renders a different prompt string, so the
    # cache key changes and every continuation misses again.
    list(loglik_mc("run-1", [example], other_variant, spy, cache, "dummy", "main"))
    assert spy.call_count == 2 * n_choices
    assert cache.misses == 2 * n_choices
    assert cache.hits == 0
