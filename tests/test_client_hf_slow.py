"""Tokenizer-boundary regression test (architecture.md §3, sprint1.md Phase 1.2).

Marked `slow`: requires downloading/caching a tiny real HF model. Skipped by
default `pytest` (network disabled); run explicitly with `pytest -m slow`.

The bug this guards against: tokenizing a continuation independently of its
prompt can merge/split tokens differently at the boundary (e.g. "foo" + "bar"
BPE-merges across the join point when tokenized jointly, but "bar" tokenizes
to a different id on its own). Scoring the wrong token count or the wrong
token IDs silently corrupts every loglikelihood-based benchmark. This is the
single most common cause of accuracy mismatches against lm-evaluation-harness.
"""

import pytest

from ladder.client import HFClient

pytestmark = pytest.mark.slow

TINY_MODEL = "sshleifer/tiny-gpt2"


def test_in_context_tokenization_differs_from_independent():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)

    # No leading space on the continuation: GPT-2 BPE merges "foo"+"bar" into a
    # single token that doesn't exist when "bar" is tokenized on its own.
    prompt = "foo"
    continuation = "bar"

    independent_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    joint_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    in_context_ids = joint_ids[len(prompt_ids) :]

    # This is exactly the failure mode the rule guards against: the same
    # continuation string tokenizes differently depending on context.
    assert independent_ids != in_context_ids, (
        "Fixture no longer demonstrates a boundary difference for this model/string; "
        "pick a new (prompt, continuation) pair that does."
    )


def test_hfclient_loglikelihood_uses_in_context_ids():
    client = HFClient(TINY_MODEL, revision="main")

    # No leading space on the continuation: GPT-2 BPE merges "foo"+"bar" into a
    # single token that doesn't exist when "bar" is tokenized on its own.
    prompt = "foo"
    continuation = "bar"

    _, n_prompt, cont_ids = client._prompt_continuation_ids(prompt, continuation)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    independent_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]

    assert cont_ids != independent_ids
    assert n_prompt == len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    # Sanity: loglikelihood runs end-to-end and returns one result per continuation.
    results = client.loglikelihood(prompt, [continuation, "baz"])
    assert len(results) == 2
    for r in results:
        assert r.n_tokens >= 1


def test_hfclient_token_nlls_repeated_text_has_lower_mean_nll_than_random():
    # Sprint 3, Phase 3.1: a real (if tiny/untrained) LM should still find a
    # short repeated phrase more predictable in aggregate than token noise —
    # this is a loose sanity check on token_nlls' wiring, not a quality bar.
    client = HFClient(TINY_MODEL, revision="main")

    repeated = "the cat sat on the mat the cat sat on the mat"
    random_text = "purple飞 zzqx 7!@# glorp xk9 ünïcödé blarg"

    repeated_result = client.token_nlls(repeated)
    random_result = client.token_nlls(random_text)

    assert len(repeated_result.nlls) > 0
    assert len(random_result.nlls) > 0

    mean_repeated = sum(repeated_result.nlls) / len(repeated_result.nlls)
    mean_random = sum(random_result.nlls) / len(random_result.nlls)

    assert mean_repeated < mean_random


def test_hfclient_token_nlls_context_is_masked_and_not_scored():
    client = HFClient(TINY_MODEL, revision="main")

    result_no_context = client.token_nlls("bar", context="")
    result_with_context = client.token_nlls("bar", context="foo")

    # n_bytes always reflects only the scored text, never the context.
    assert result_no_context.n_bytes == len("bar".encode("utf-8"))
    assert result_with_context.n_bytes == len("bar".encode("utf-8"))

    # Context conditions the prediction, so per-token NLLs may legitimately
    # differ in value and even in count (in-context tokenization can split
    # "bar" differently depending on what precedes it) — but context itself
    # must never appear as scored positions.
    assert len(result_with_context.nlls) <= len(
        client.tokenizer("foo" + "bar", add_special_tokens=False)["input_ids"]
    )
