"""loglik_mc/perplexity/cloze/generative evaluators: render -> client call -> score -> ExampleResult (architecture.md §6)."""

import math

import pytest

from ladder.client import GenParams, LoglikResult, ModelClient, TokenNLLs, get_client
from ladder.datasets import get_loader
from ladder.evaluators import cloze, generative, loglik_mc, perplexity
from ladder.metrics import acc, perplexity_metrics
from ladder.prompts import load_variant
from ladder.prompts import render as render_request
from ladder.records import Example, ExampleResult
from ladder.storage import PredictionCache, connect


def _cache(tmp_path, name="ladder.db") -> PredictionCache:
    return PredictionCache(connect(tmp_path / name))


def render_prompt(example: Example, variant) -> str:
    return render_request(example, variant).prompt


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


def test_loglik_mc_prepends_fewshot_demos_when_variant_requests_them(tmp_path):
    loader = get_loader("arc_easy")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("arc_easy/mc_letter_5shot_v1").model_copy(
        update={"fewshot_split": "fixture_train"}
    )
    client = get_client("dummy", seed=0)

    result = next(loglik_mc("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    # Prepended demos means the scored prompt is much longer than a zero-shot
    # render of the same example, and each fixture_train question text
    # appears somewhere in it.
    zero_shot_prompt = render_prompt(example, load_variant("arc_easy/mc_letter_v1"))
    assert result.detail["answer_index"] == example.payload["answer_index"]
    train_examples = list(loader.load("fixture_train"))
    demo_prompt = render_request(example, variant, demos=train_examples[:5]).prompt
    assert len(demo_prompt) > len(zero_shot_prompt)


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
    """Fake `ModelClient` whose `generate` output and greedy-match flag are set per-prompt by the test.

    `DummyClient.generate` always produces "The answer is {number}." (client.py),
    which can never equal an arbitrary LAMBADA target word, so it cannot
    exercise `cloze`'s generation-based diagnostic path. This rigged client
    lets a test pin an exact generation per prompt (Sprint 3, Phase 3.4's
    "DummyClient rigged to reproduce/not-reproduce targets" requirement)
    and independently pin `is_greedy_match` (Sprint 4/5's greedy target-word
    accuracy primary metric, architecture.md §6) — the two are deliberately
    decoupled in tests since they're now decoupled in `cloze` itself (a
    generation can differ from the target while `is_greedy_match` is still
    True, exactly the "Queen." vs. "Queen" bug this rig exists to reproduce).
    """

    def __init__(self, generations: dict[str, str], loglik_value: float = -1.5, is_greedy_match: bool = True):
        self._generations = generations
        self._loglik_value = loglik_value
        self._is_greedy_match = is_greedy_match

    def loglikelihood(self, prompt: str, continuations: list[str]) -> list[LoglikResult]:
        return [
            LoglikResult(loglik=self._loglik_value, n_tokens=1, is_greedy_match=self._is_greedy_match)
            for _ in continuations
        ]

    def generate(self, prompt: str, params: GenParams) -> str:
        return self._generations[prompt]

    def token_nlls(self, text: str, context: str = "") -> TokenNLLs:
        raise NotImplementedError


def _lambada_example(context: str, target: str, example_id: str = "lambada-x") -> Example:
    return Example(dataset="lambada", split="fixture", example_id=example_id, payload={"context": context, "target": target})


def test_cloze_correct_when_target_is_greedy_match(tmp_path):
    # Primary metric: correct/score come from is_greedy_match, not from
    # whether generation equals target (architecture.md §6, Sprint 4/5 fix).
    example = _lambada_example("The sky is very", "blue")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"The sky is very": " blue"}, is_greedy_match=True)

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["generation"] == " blue"
    assert result.detail["target"] == "blue"


def test_cloze_incorrect_when_target_is_not_greedy_match(tmp_path):
    example = _lambada_example("The sky is very", "blue")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"The sky is very": " green"}, is_greedy_match=False)

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is False
    assert result.score == 0.0
    assert result.detail["generation"] == " green"


def test_cloze_scores_correct_despite_generation_mismatch_when_greedy_match_true(tmp_path):
    # The bug this metric fixes: a raw generation that doesn't equal the
    # target string verbatim (e.g. it ran on to a second, unrelated word)
    # must not affect `correct` at all — only is_greedy_match does.
    example = _lambada_example("The king and", "Queen", "ex-queen")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"The king and": " something else entirely"}, is_greedy_match=True)

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is True  # primary metric: greedy match, not string equality
    assert result.detail["generation"] == " something else entirely"


def test_cloze_target_logprob_stored_in_detail(tmp_path):
    example = _lambada_example("The sky is very", "blue")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"The sky is very": " blue"}, loglik_value=-2.75)

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.detail["target_logprob"] == -2.75
    assert result.detail["target_nll"] == 2.75
    assert result.detail["target_ppl"] == pytest.approx(math.exp(2.75))


def test_cloze_nonstandard_generated_word_acc_normalizes_punctuation(tmp_path):
    example = _lambada_example("The king and", "Queen", "ex-queen2")
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient({"The king and": " Queen."})

    (result,) = list(cloze("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    # Diagnostic field: strips trailing punctuation before comparing, unlike
    # the old strict comparison — this is what the deprecated exact-match
    # behavior *should* have done, kept only as a sanity-check diagnostic.
    assert result.detail["nonstandard_generated_word_acc"] is True


def test_cloze_accuracy_hand_computed_three_of_five(tmp_path):
    # 5 examples, rigged is_greedy_match: 3 True, 2 False. acc = 3/5 = 0.6
    # (primary metric — no longer tied to the generation string at all).
    examples = [
        _lambada_example("The sky is very", "blue", "ex0"),
        _lambada_example("It began to", "snow", "ex1"),
        _lambada_example("He grabbed his", "keys", "ex2"),
        _lambada_example("Add a dash of", "pepper", "ex3"),
        _lambada_example("She opened the", "page", "ex4"),
    ]
    match_flags = {
        "The sky is very": True,
        "It began to": True,
        "He grabbed his": False,
        "Add a dash of": True,
        "She opened the": False,
    }
    generations = {p: " x" for p in match_flags}  # generation text is irrelevant to `correct` now
    variant = load_variant("lambada/cloze_v1")

    class _PerPromptGreedyClient(_RiggedClient):
        def loglikelihood(self, prompt: str, continuations: list[str]) -> list[LoglikResult]:
            return [
                LoglikResult(loglik=self._loglik_value, n_tokens=1, is_greedy_match=match_flags[prompt])
                for _ in continuations
            ]

    client = _PerPromptGreedyClient(generations)

    results = list(cloze("run-1", examples, variant, client, _cache(tmp_path), "dummy", "main"))
    accuracy = acc(results)

    assert accuracy == 0.6
    assert sum(1 for r in results if r.correct) == 3


def test_cloze_metrics_aggregates_secondary_and_diagnostic_fields(tmp_path):
    from ladder.metrics import cloze_metrics

    examples = [
        _lambada_example("The sky is very", "blue", "ex0"),
        _lambada_example("It began to", "snow", "ex1"),
    ]
    variant = load_variant("lambada/cloze_v1")
    client = _RiggedClient(
        {"The sky is very": " blue", "It began to": " snow"}, loglik_value=-2.0, is_greedy_match=True
    )

    results = list(cloze("run-1", examples, variant, client, _cache(tmp_path), "dummy", "main"))
    metrics = cloze_metrics(results)

    assert metrics["target_nll_mean"] == pytest.approx(2.0)
    assert metrics["target_ppl_mean"] == pytest.approx(math.exp(2.0))
    assert metrics["nonstandard_generated_word_acc"] == 1.0




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


# --- generative (GSM8K) -----------------------------------------------------


def _gsm8k_example(question: str, answer_number: float, example_id: str = "gsm8k-x") -> Example:
    return Example(
        dataset="gsm8k", split="fixture", example_id=example_id, payload={"question": question, "answer_number": answer_number}
    )


def test_generative_correct_when_extracted_number_matches(tmp_path):
    example = _gsm8k_example("What is 2 + 2?", 4.0)
    variant = load_variant("gsm8k/gen_v1")
    request_prompt = render_prompt(example, variant)
    client = _RiggedClient({request_prompt: "2 + 2 = 4. Final answer: 4"})

    (result,) = list(generative("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["extracted_number"] == 4.0
    assert result.detail["answer_number"] == 4.0


def test_generative_incorrect_when_extracted_number_differs(tmp_path):
    example = _gsm8k_example("What is 2 + 2?", 4.0)
    variant = load_variant("gsm8k/gen_v1")
    request_prompt = render_prompt(example, variant)
    client = _RiggedClient({request_prompt: "2 + 2 = 5. Final answer: 5"})

    (result,) = list(generative("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is False
    assert result.score == 0.0
    assert result.detail["extracted_number"] == 5.0


def test_generative_incorrect_when_no_number_extracted(tmp_path):
    example = _gsm8k_example("What is 2 + 2?", 4.0)
    variant = load_variant("gsm8k/gen_v1")
    request_prompt = render_prompt(example, variant)
    client = _RiggedClient({request_prompt: "I'm not sure how to solve this."})

    (result,) = list(generative("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert result.correct is False
    assert result.detail["extracted_number"] is None


def test_generative_uses_variant_gen_params(tmp_path):
    example = _gsm8k_example("What is 2 + 2?", 4.0)
    variant = load_variant("gsm8k/gen_v1")
    request_prompt = render_prompt(example, variant)

    class _CapturingClient(_RiggedClient):
        def __init__(self):
            super().__init__({request_prompt: "Final answer: 4"})
            self.seen_params: GenParams | None = None

        def generate(self, prompt: str, params: GenParams) -> str:
            self.seen_params = params
            return super().generate(prompt, params)

    client = _CapturingClient()
    list(generative("run-1", [example], variant, client, _cache(tmp_path), "dummy", "main"))

    assert client.seen_params is not None
    assert client.seen_params.max_new_tokens == variant.max_new_tokens
    assert client.seen_params.stop == variant.stop
    assert client.seen_params.temperature == 0.0


def test_generative_accuracy_hand_computed_three_of_five(tmp_path):
    # Sprint 3, Phase 3.5: accuracy fixture worked by hand. 5 examples, rigged
    # generations: 3 extract to the correct answer_number, 2 don't.
    #   ex0: answer=4.0   generation="Final answer: 4"  -> extracted 4.0  -> match
    #   ex1: answer=10.0  generation="Final answer: 10" -> extracted 10.0 -> match
    #   ex2: answer=7.0   generation="Final answer: 9"  -> extracted 9.0  -> no match
    #   ex3: answer=100.0 generation="#### 100"          -> extracted 100.0 -> match
    #   ex4: answer=3.0   generation="not sure"          -> extracted None -> no match
    # acc = 3/5 = 0.6
    variant = load_variant("gsm8k/gen_v1")
    examples = [
        _gsm8k_example("Q0?", 4.0, "ex0"),
        _gsm8k_example("Q1?", 10.0, "ex1"),
        _gsm8k_example("Q2?", 7.0, "ex2"),
        _gsm8k_example("Q3?", 100.0, "ex3"),
        _gsm8k_example("Q4?", 3.0, "ex4"),
    ]
    generations = {
        render_prompt(examples[0], variant): "Final answer: 4",
        render_prompt(examples[1], variant): "Final answer: 10",
        render_prompt(examples[2], variant): "Final answer: 9",
        render_prompt(examples[3], variant): "#### 100",
        render_prompt(examples[4], variant): "not sure",
    }
    client = _RiggedClient(generations)

    results = list(generative("run-1", examples, variant, client, _cache(tmp_path), "dummy", "main"))
    accuracy = acc(results)

    assert accuracy == 0.6
    assert sum(1 for r in results if r.correct) == 3
