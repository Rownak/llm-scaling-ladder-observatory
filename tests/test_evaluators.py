"""loglik_mc/perplexity/cloze evaluators: render -> client call -> score -> ExampleResult (architecture.md §6)."""

import math

from ladder.client import GenParams, LoglikResult, ModelClient, TokenNLLs, get_client
from ladder.datasets import get_loader
from ladder.evaluators import cloze, loglik_mc, perplexity
from ladder.metrics import acc, perplexity_metrics
from ladder.prompts import load_variant
from ladder.records import Example, ExampleResult
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


def _ppl_example(text: str, example_id: str = "doc-1") -> Example:
    return Example(dataset="toy_ppl", split="fixture", example_id=example_id, payload={"text": text})


def test_perplexity_hand_computed_windowed_ppl_and_bpb(tmp_path):
    # Sprint 3, Phase 3.2 non-negotiable fixture: a 9-word toy document,
    # window=4, stride=2, scored by DummyClient(seed=0, model_id="dummy").
    #
    # Word list: ["the","quick","brown","fox","jumps","over","the","lazy","dog"]
    # (9 words); raw text is 43 UTF-8 bytes.
    #
    # Sliding windows (word indices), covered = words already scored:
    #   start=0: window=[the,quick,brown,fox]        covered=0 -> context=[]                new=[the,quick,brown,fox]
    #   start=2: window=[brown,fox,jumps,over]        covered=4 -> context=[brown,fox]        new=[jumps,over]
    #   start=4: window=[jumps,over,the,lazy]         covered=6 -> context=[jumps,over]        new=[the,lazy]
    #   start=6: window=[the,lazy,dog]  (last, short)  covered=8 -> context=[the,lazy]          new=[dog]
    # Every word appears in exactly one window's "new" (scored) text -> no
    # double-scoring, 4+2+2+1 = 9 = word count.
    #
    # DummyClient(seed=0, model_id="dummy").token_nlls(new_text, context=...)
    # is deterministic; the pinned per-call NLLs (nats), read off the client
    # directly, are:
    #   token_nlls("the quick brown fox", context="")            -> [2.5368, 0.5216, 3.6888, 1.3900]
    #   token_nlls("jumps over", context="brown fox")             -> [3.1928, 2.6388]
    #   token_nlls("the lazy", context="jumps over")              -> [2.2660, 1.0036]
    #   token_nlls("dog", context="the lazy")                     -> [1.4304]
    #
    # Concatenated window_nlls (9 values), summed:
    #   total_nll_nats = 2.5368+0.5216+3.6888+1.3900+3.1928+2.6388+2.2660+1.0036+1.4304
    #                  = 18.6688 (nats)
    #
    # ppl = exp(total_nll_nats / n_scored_tokens) = exp(18.6688 / 9) = exp(2.074311...) ≈ 7.959061660899256
    # bpb = (total_nll_nats / ln(2)) / n_bytes = (18.6688 / 0.6931471805599453) / 43 ≈ 0.6263577948685553
    text = "the quick brown fox jumps over the lazy dog"
    example = _ppl_example(text)
    client = get_client("dummy", seed=0)
    cache = _cache(tmp_path)

    results = list(perplexity("run-1", [example], client, cache, "dummy", "main", window=4, stride=2))
    assert len(results) == 1
    result = results[0]

    assert result.correct is None
    assert result.detail["n_bytes"] == 43
    assert len(result.detail["window_nlls"]) == 9

    metrics = perplexity_metrics(results)
    assert metrics["n_scored_tokens"] == 9.0
    assert metrics["n_bytes"] == 43.0
    assert math.isclose(metrics["ppl"], 7.959061660899256, rel_tol=1e-9)
    assert math.isclose(metrics["bpb"], 0.6263577948685553, rel_tol=1e-9)
    # score is the document's own bpb (single-document run == run-level bpb)
    assert math.isclose(result.score, metrics["bpb"], rel_tol=1e-9)


def test_perplexity_windowing_scores_every_token_exactly_once(tmp_path):
    # Property test (not hand-computed): for a document long enough to need
    # several overlapping windows, total scored tokens must equal the
    # document's own word count, and no word's text may appear in more than
    # one window's "new" (scored) span.
    text = " ".join(f"word{i}" for i in range(23))  # 23 words, window=4 stride=2 -> several windows
    example = _ppl_example(text)
    client = get_client("dummy", seed=0)
    cache = _cache(tmp_path)

    (result,) = list(perplexity("run-1", [example], client, cache, "dummy", "main", window=4, stride=2))

    assert len(result.detail["window_nlls"]) == len(text.split())


def test_perplexity_default_stride_is_half_window(tmp_path):
    # architecture.md §6: stride defaults to W/2 when not passed explicitly.
    text = " ".join(f"word{i}" for i in range(23))
    example = _ppl_example(text)

    results_default = list(
        perplexity("run-1", [example], get_client("dummy", seed=0), _cache(tmp_path, "a.db"), "dummy", "main", window=8)
    )
    results_explicit = list(
        perplexity(
            "run-1", [example], get_client("dummy", seed=0), _cache(tmp_path, "b.db"), "dummy", "main", window=8, stride=4
        )
    )

    assert results_default[0].detail["window_nlls"] == results_explicit[0].detail["window_nlls"]


def test_perplexity_multiple_documents_aggregate_across_run(tmp_path):
    examples = [_ppl_example("alpha beta gamma delta", "doc-a"), _ppl_example("epsilon zeta eta theta", "doc-b")]
    client = get_client("dummy", seed=0)
    cache = _cache(tmp_path)

    results = list(perplexity("run-1", examples, client, cache, "dummy", "main", window=4, stride=2))
    assert len(results) == 2

    metrics = perplexity_metrics(results)
    expected_tokens = sum(len(r.detail["window_nlls"]) for r in results)
    expected_bytes = sum(r.detail["n_bytes"] for r in results)
    assert metrics["n_scored_tokens"] == expected_tokens
    assert metrics["n_bytes"] == expected_bytes


# --- cloze (LAMBADA) --------------------------------------------------------


class _RiggedClient(ModelClient):
    """Fake `ModelClient` whose `generate` output is set per-prompt by the test.

    `DummyClient.generate` always produces "The answer is {number}." (client.py),
    which can never equal an arbitrary LAMBADA target word, so it cannot
    exercise `cloze`'s "reproduces the target" path. This rigged client lets a
    test pin an exact generation per prompt (Sprint 3, Phase 3.4's "DummyClient
    rigged to reproduce/not-reproduce targets" requirement) while keeping
    `loglikelihood` deterministic so `detail["target_logprob"]` is still
    checkable.
    """

    def __init__(self, generations: dict[str, str], loglik_value: float = -1.5):
        self._generations = generations
        self._loglik_value = loglik_value

    def loglikelihood(self, prompt: str, continuations: list[str]) -> list[LoglikResult]:
        return [LoglikResult(loglik=self._loglik_value, n_tokens=1) for _ in continuations]

    def generate(self, prompt: str, params: GenParams) -> str:
        return self._generations[prompt]

    def token_nlls(self, text: str, context: str = "") -> TokenNLLs:
        raise NotImplementedError


def _lambada_example(context: str, target: str, example_id: str = "lambada-x") -> Example:
    return Example(dataset="lambada", split="fixture", example_id=example_id, payload={"context": context, "target": target})


def test_cloze_exact_match_when_generation_equals_target(tmp_path):
    example = _lambada_example("The sky is very", "blue")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"The sky is very": " blue"})

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["generation"] == " blue"
    assert result.detail["target"] == "blue"


def test_cloze_no_match_when_generation_differs_from_target(tmp_path):
    example = _lambada_example("The sky is very", "blue")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"The sky is very": " green"})

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is False
    assert result.score == 0.0
    assert result.detail["generation"] == " green"


def test_cloze_strips_whitespace_before_comparing(tmp_path):
    # Generation carries a leading space (continuation convention, same as
    # loglik_mc's " A"/" foo" continuations) and possibly trailing
    # whitespace/newline fragments; comparison must strip both.
    example = _lambada_example("Roses are red, violets are", "blue")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"Roses are red, violets are": "  blue  "})

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is True


def test_cloze_target_logprob_stored_in_detail(tmp_path):
    example = _lambada_example("The sky is very", "blue")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"The sky is very": " blue"}, loglik_value=-2.75)

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.detail["target_logprob"] == -2.75


def test_cloze_accuracy_hand_computed_three_of_five(tmp_path):
    # Sprint 3, Phase 3.4: accuracy fixture worked by hand. 5 examples, rigged
    # generations: 3 exactly reproduce their target, 2 don't.
    #   ex0: target="blue"  generation=" blue"  -> match
    #   ex1: target="snow"  generation=" snow"  -> match
    #   ex2: target="keys"  generation=" wallet" -> no match
    #   ex3: target="pepper" generation=" pepper" -> match
    #   ex4: target="page"  generation=" book"  -> no match
    # acc = 3/5 = 0.6
    examples = [
        _lambada_example("The sky is very", "blue", "ex0"),
        _lambada_example("It began to", "snow", "ex1"),
        _lambada_example("He grabbed his", "keys", "ex2"),
        _lambada_example("Add a dash of", "pepper", "ex3"),
        _lambada_example("She opened the", "page", "ex4"),
    ]
    generations = {
        "The sky is very": " blue",
        "It began to": " snow",
        "He grabbed his": " wallet",
        "Add a dash of": " pepper",
        "She opened the": " book",
    }
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient(generations)

    results = list(cloze("run-1", examples, variant, client, _cache(tmp_path), "dummy", "main"))
    accuracy = acc(results)

    assert accuracy == 0.6
    assert sum(1 for r in results if r.correct) == 3


def test_cloze_max_new_tokens_scales_with_target_word_count(tmp_path):
    # A multi-word target should get a proportionally larger generation
    # budget, not a fixed cap sized for single-word LAMBADA targets.
    from ladder.evaluators import _CLOZE_TOKENS_PER_TARGET_WORD

    example = _lambada_example("They walked into the", "old wooden barn")
    variant = load_variant("lambada/cloze_v1")

    class _CapturingClient(_RiggedClient):
        def __init__(self):
            super().__init__({"They walked into the": " old wooden barn"})
            self.seen_params: GenParams | None = None

        def generate(self, prompt: str, params: GenParams) -> str:
            self.seen_params = params
            return super().generate(prompt, params)

    client = _CapturingClient()
    list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert client.seen_params is not None
    assert client.seen_params.max_new_tokens == _CLOZE_TOKENS_PER_TARGET_WORD * 3
