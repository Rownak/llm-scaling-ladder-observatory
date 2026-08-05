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
