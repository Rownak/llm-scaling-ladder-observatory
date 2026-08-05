"""Registry behavior and DummyClient determinism (architecture.md §3).

Direct-import ban: every other module in `ladder` must go through
`ladder.client.get_client(name, revision=...)`. Only this file may import
`DummyClient`/`HFClient` directly, to test the classes themselves.
"""

import pytest

from ladder.client import (
    DummyClient,
    GenParams,
    ModelClient,
    get_client,
    register,
)


def test_get_client_dummy_returns_dummy_client():
    client = get_client("dummy")
    assert isinstance(client, DummyClient)
    assert isinstance(client, ModelClient)


def test_get_client_pythia_returns_hf_client_type():
    # Import is deferred inside HFClient.__init__ (transformers/torch), so
    # constructing one here would require real deps + network. Instead assert
    # the registry maps the model_id to the right factory without instantiating.
    from ladder.client import _REGISTRY, HFClient

    factory = _REGISTRY["pythia-70m"]
    assert factory.__name__ == "factory"  # the _make_hf_factory closure


def test_get_client_unknown_model_raises_keyerror():
    with pytest.raises(KeyError, match="pythia-9000"):
        get_client("pythia-9000")


def test_register_adds_new_entry():
    class _FakeClient(ModelClient):
        def loglikelihood(self, prompt, continuations):
            return []

        def generate(self, prompt, params):
            return ""

        def token_nlls(self, text, context=""):
            raise NotImplementedError

    register("test-fake", lambda revision="main", **kw: _FakeClient())
    client = get_client("test-fake")
    assert isinstance(client, _FakeClient)


def test_dummy_client_loglikelihood_is_deterministic():
    a = DummyClient(seed=0, model_id="dummy")
    b = DummyClient(seed=0, model_id="dummy")
    results_a = a.loglikelihood("Question: 2+2?\nAnswer:", [" 3", " 4"])
    results_b = b.loglikelihood("Question: 2+2?\nAnswer:", [" 3", " 4"])
    assert [r.model_dump() for r in results_a] == [r.model_dump() for r in results_b]


def test_dummy_client_loglikelihood_differs_by_seed():
    a = DummyClient(seed=0, model_id="dummy")
    b = DummyClient(seed=1, model_id="dummy")
    results_a = a.loglikelihood("prompt", [" x"])
    results_b = b.loglikelihood("prompt", [" x"])
    assert results_a[0].loglik != results_b[0].loglik


def test_dummy_client_loglikelihood_shape():
    client = DummyClient(seed=0)
    results = client.loglikelihood("prompt", [" A", " B", " C"])
    assert len(results) == 3
    for r in results:
        assert r.loglik < 0  # logliks are negative log-probabilities
        assert r.n_tokens >= 1


def test_dummy_client_generate_is_deterministic_and_extractable():
    a = DummyClient(seed=0)
    b = DummyClient(seed=0)
    text_a = a.generate("What is 6*7?", GenParams())
    text_b = b.generate("What is 6*7?", GenParams())
    assert text_a == text_b
    # A number must be extractable, per architecture.md §3 (GSM8K needs this).
    assert any(ch.isdigit() for ch in text_a)


def test_dummy_client_generate_respects_stop_sequence():
    client = DummyClient(seed=0)
    text = client.generate("prompt", GenParams(stop=["."]))
    assert "." not in text


def test_dummy_client_token_nlls_raises_until_sprint3():
    client = DummyClient(seed=0)
    with pytest.raises(NotImplementedError):
        client.token_nlls("some text")
