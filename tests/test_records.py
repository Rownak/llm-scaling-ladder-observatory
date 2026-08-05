"""Round-trip every record type through JSON serialization (architecture.md §2)."""

from ladder.records import (
    Example,
    ExampleResult,
    Prediction,
    RenderedRequest,
    RunRecord,
)


def _round_trip(model_cls, instance):
    restored = model_cls.model_validate_json(instance.model_dump_json())
    assert restored == instance
    return restored


def test_example_round_trip():
    ex = Example(
        dataset="arc_easy",
        split="test",
        example_id="arc_easy-test-0",
        payload={"question": "What is 2+2?", "choices": ["3", "4"], "answer_index": 1},
    )
    _round_trip(Example, ex)


def test_rendered_request_round_trip_loglik():
    req = RenderedRequest(
        example_id="arc_easy-test-0",
        prompt_variant_id="arc_easy/mc_letter_v1",
        kind="loglik",
        prompt="Question: What is 2+2?\nA. 3\nB. 4\nAnswer:",
        continuations=[" A", " B"],
        gen_params=None,
    )
    _round_trip(RenderedRequest, req)


def test_rendered_request_round_trip_nll_none_variant():
    # PPL runs have no prompt_variant_id and no continuations.
    req = RenderedRequest(
        example_id="wikitext-test-0",
        prompt_variant_id=None,
        kind="nll",
        prompt="The quick brown fox jumps over the lazy dog.",
        continuations=None,
        gen_params=None,
    )
    _round_trip(RenderedRequest, req)


def test_prediction_round_trip():
    pred = Prediction(
        request_hash="deadbeef",
        model_id="pythia-70m",
        revision="main",
        logliks=[-3.21, -1.05],
        token_nlls=None,
        generation=None,
        n_bytes=None,
    )
    _round_trip(Prediction, pred)


def test_prediction_round_trip_generation_and_nlls():
    pred = Prediction(
        request_hash="cafef00d",
        model_id="pythia-70m",
        revision="main",
        logliks=None,
        token_nlls=[0.1, 0.2, 0.3],
        generation="42",
        n_bytes=128,
    )
    _round_trip(Prediction, pred)


def test_example_result_round_trip():
    res = ExampleResult(
        run_id="run-1",
        example_id="arc_easy-test-0",
        correct=True,
        score=1.0,
        detail={"chosen_option": 1},
    )
    _round_trip(ExampleResult, res)


def test_example_result_round_trip_correct_none():
    # PPL results have no correct/incorrect notion.
    res = ExampleResult(
        run_id="run-2",
        example_id="wikitext-test-0",
        correct=None,
        score=1.23,
        detail={"window_nlls": [0.1, 0.2]},
    )
    _round_trip(ExampleResult, res)


def test_run_record_round_trip():
    run = RunRecord(
        run_id="run-1",
        model_id="pythia-70m",
        revision="main",
        dataset="arc_easy",
        split="test",
        prompt_variant_id="arc_easy/mc_letter_v1",
        evaluator="loglik_mc",
        framework="ladder",
        metrics={"acc": 0.42},
        n_examples=200,
        seed=0,
        code_version="0.1.0",
        config={"limit": 200},
        status="done",
        started_at="2026-08-05T10:00:00",
        finished_at="2026-08-05T10:05:00",
    )
    _round_trip(RunRecord, run)


def test_run_record_timestamps_default_to_none():
    run = RunRecord(
        run_id="run-1",
        model_id="pythia-70m",
        revision="main",
        dataset="arc_easy",
        split="test",
        prompt_variant_id=None,
        evaluator="loglik_mc",
        framework="ladder",
        metrics={},
        n_examples=0,
        seed=0,
        code_version="0.1.0",
        config={},
        status="running",
    )
    assert run.started_at is None
    assert run.finished_at is None
    _round_trip(RunRecord, run)


def test_run_record_round_trip_all_statuses():
    for status in ("running", "done", "failed"):
        run = RunRecord(
            run_id=f"run-{status}",
            model_id="pythia-70m",
            revision="main",
            dataset="arc_easy",
            split="test",
            prompt_variant_id=None,
            evaluator="perplexity",
            framework="ladder",
            metrics={},
            n_examples=0,
            seed=0,
            code_version="0.1.0",
            config={},
            status=status,
        )
        _round_trip(RunRecord, run)
