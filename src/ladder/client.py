"""ModelClient ABC, registry, DummyClient, HFClient.

Registry access only — no module outside this file's own tests may import
`DummyClient` or `HFClient` directly. Always go through `get_client(name, revision=...)`
so every caller is agnostic to which client backs a model_id.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel


class LoglikResult(BaseModel):
    """Log-likelihood of a single continuation, scored in context.

    Attributes:
        loglik: Sum logprob of the continuation tokens, in context.
        n_tokens: Number of continuation tokens (used for length-normalized accuracy).
    """

    loglik: float  # sum logprob of the continuation, in context
    n_tokens: int  # continuation token count (for acc_norm)


class GenParams(BaseModel):
    """Generation parameters passed to `ModelClient.generate`.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        stop: Stop sequences; generation output is truncated at the first match.
        temperature: Sampling temperature; 0.0 selects greedy decoding.
    """

    max_new_tokens: int = 32
    stop: list[str] | None = None
    temperature: float = 0.0  # 0.0 == greedy


class TokenNLLs(BaseModel):
    """Per-token negative log-likelihoods for a scored span of text.

    Attributes:
        nlls: Per-token NLL in nats; context tokens are excluded.
        n_bytes: UTF-8 byte length of the scored (non-context) text.
    """

    nlls: list[float]  # per-token NLL (nats), context positions excluded
    n_bytes: int  # UTF-8 bytes of the scored (non-context) text


class ModelClient(ABC):
    """Abstract interface every model backend implements.

    Defines the three ways `ladder` evaluators interact with a model:
    scoring continuations, free-form generation, and per-token NLL scoring
    for perplexity. Concrete implementations (`DummyClient`, `HFClient`) are
    accessed only via the module-level registry (`get_client`), never
    imported directly by other modules.
    """

    @abstractmethod
    def loglikelihood(self, prompt: str, continuations: list[str]) -> list[LoglikResult]:
        """Score each continuation's log-likelihood conditioned on `prompt`.

        Args:
            prompt: Context the continuations are conditioned on.
            continuations: Candidate continuation strings to score.

        Returns:
            One `LoglikResult` per continuation, in the same order.
        """
        ...

    @abstractmethod
    def generate(self, prompt: str, params: GenParams) -> str:
        """Generate free-form text continuing from `prompt`.

        Args:
            prompt: Context to generate from.
            params: Generation parameters (max tokens, stop sequences, temperature).

        Returns:
            The generated text, truncated at the first matching stop sequence if any.
        """
        ...

    @abstractmethod
    def token_nlls(self, text: str, context: str = "") -> TokenNLLs:
        """Compute per-token NLLs of `text`, optionally conditioned on `context`.

        Args:
            text: The text span to score.
            context: Optional preceding context (not itself scored).

        Returns:
            Per-token NLLs and the UTF-8 byte length of `text`.
        """
        ...

    def unload(self) -> None:
        """Release any loaded model resources. No-op by default."""


_REGISTRY: dict[str, Callable[..., ModelClient]] = {}


def register(name: str, factory: Callable[..., ModelClient]) -> None:
    """Register a `ModelClient` factory under `name` in the module registry.

    Args:
        name: model_id key callers will use with `get_client`.
        factory: Callable that constructs a `ModelClient` given `revision=` and kwargs.

    Side Effects:
        Mutates the module-level `_REGISTRY` dict.
    """
    _REGISTRY[name] = factory


def get_client(name: str, revision: str = "main", **kwargs) -> ModelClient:
    """Look up and construct a registered `ModelClient`.

    Args:
        name: model_id previously passed to `register`.
        revision: Checkpoint/revision to instantiate the client with.
        **kwargs: Extra keyword arguments forwarded to the registered factory.

    Returns:
        A new `ModelClient` instance for the given model_id and revision.

    Raises:
        KeyError: If `name` has no registered factory.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"No client registered for model_id={name!r}. "
            f"Known model_ids: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](revision=revision, **kwargs)


# --- DummyClient --------------------------------------------------------


def _stable_hash(*parts: str) -> int:
    """Deterministic, process-independent hash (Python's built-in `hash()` is salted per-run).

    Args:
        *parts: Strings to combine into the hash input.

    Returns:
        A deterministic integer derived from the first 16 hex digits of the
        SHA-256 digest of the parts joined by "|".
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


class DummyClient(ModelClient):
    """Deterministic client with no dependencies. Used for offline tests, CI, and demos.

    Determinism contract: identical (seed, model_id, revision, prompt, continuations)
    always produces identical output, across processes and runs.
    """

    def __init__(self, revision: str = "main", seed: int = 0, model_id: str = "dummy"):
        """Initialize the deterministic dummy client.

        Args:
            revision: Nominal checkpoint/revision label (affects output determinism key).
            seed: Seed mixed into the determinism key.
            model_id: Nominal model_id label (affects output determinism key).
        """
        self.revision = revision
        self.seed = seed
        self.model_id = model_id

    def _rng_value(self, *parts: str) -> float:
        """Derive a deterministic pseudo-random float in [0, 1) from `parts` and instance state.

        Args:
            *parts: Additional strings (e.g. prompt, continuation) to mix into the key.

        Returns:
            A float in [0, 1), stable across processes and runs for the same inputs.
        """
        h = _stable_hash(str(self.seed), self.model_id, self.revision, *parts)
        return (h % 10_000) / 10_000.0  # deterministic float in [0, 1)

    def loglikelihood(self, prompt: str, continuations: list[str]) -> list[LoglikResult]:
        """See `ModelClient.loglikelihood`. Scores are synthetic but deterministic."""
        results = []
        for cont in continuations:
            u = self._rng_value(prompt, cont)
            # Negative loglik, scaled by continuation length so longer continuations
            # plausibly accumulate more negative log-probability.
            n_tokens = max(1, len(cont.split()))
            loglik = -(1.0 + 4.0 * u) * n_tokens
            results.append(LoglikResult(loglik=loglik, n_tokens=n_tokens))
        return results

    def generate(self, prompt: str, params: GenParams) -> str:
        """See `ModelClient.generate`. Produces a synthetic deterministic sentence."""
        u = self._rng_value(prompt)
        number = int(u * 1000)
        text = f"The answer is {number}."
        if params.stop:
            for stop_seq in params.stop:
                idx = text.find(stop_seq)
                if idx != -1:
                    text = text[:idx]
        return text

    def token_nlls(self, text: str, context: str = "") -> TokenNLLs:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always; arrives in Sprint 3.
        """
        raise NotImplementedError("DummyClient.token_nlls arrives in Sprint 3")


# --- HFClient ------------------------------------------------------------


class HFClient(ModelClient):
    """Wraps `transformers` with `revision=` for checkpoint selection.

    Continuations are tokenized in context (tokenize prompt+continuation, then
    subtract the prompt's token count) — never independently. Independent
    tokenization is the #1 source of cross-framework mismatch (e.g. leading-space
    merges at the prompt/continuation boundary).
    """

    def __init__(self, model_id: str, revision: str = "main"):
        """Load a `transformers` causal LM and tokenizer for `model_id`/`revision`.

        Args:
            model_id: Hugging Face Hub repo id to load.
            revision: Checkpoint/revision to load.

        Side Effects:
            Downloads/loads the tokenizer and model weights, and moves the
            model to the available device (CUDA if present, else CPU).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def _prompt_continuation_ids(self, prompt: str, continuation: str):
        """Tokenize prompt+continuation jointly; split by prompt token count.

        This is the in-context tokenization rule: tokenizing `continuation` on its
        own can merge/split differently at the boundary (e.g. a leading space
        attaching to the previous token), which silently shifts scored positions.

        Args:
            prompt: Context string.
            continuation: Continuation string to be scored in context.

        Returns:
            A tuple of (full token ids for prompt+continuation, prompt token
            count, continuation-only token ids).
        """
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
        n_prompt = len(prompt_ids)
        continuation_ids = full_ids[n_prompt:]
        return full_ids, n_prompt, continuation_ids

    def loglikelihood(self, prompt: str, continuations: list[str]) -> list[LoglikResult]:
        """See `ModelClient.loglikelihood`. Scores continuations via joint in-context tokenization."""
        import torch

        results = []
        for cont in continuations:
            full_ids, n_prompt, cont_ids = self._prompt_continuation_ids(prompt, cont)
            if len(cont_ids) == 0:
                results.append(LoglikResult(loglik=0.0, n_tokens=0))
                continue

            input_ids = torch.tensor([full_ids], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids).logits[0]  # (seq_len, vocab)

            log_probs = torch.log_softmax(logits.float(), dim=-1)
            # Position i's logits predict token i+1. Continuation tokens start at
            # index n_prompt, so their predicting positions start at n_prompt - 1.
            total_loglik = 0.0
            for i, token_id in enumerate(cont_ids):
                pred_pos = n_prompt - 1 + i
                total_loglik += log_probs[pred_pos, token_id].item()

            results.append(LoglikResult(loglik=total_loglik, n_tokens=len(cont_ids)))
        return results

    def generate(self, prompt: str, params: GenParams) -> str:
        """See `ModelClient.generate`. Uses greedy decoding unless `params.temperature > 0`."""
        import torch

        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.temperature > 0.0,
                temperature=params.temperature if params.temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if params.stop:
            for stop_seq in params.stop:
                idx = text.find(stop_seq)
                if idx != -1:
                    text = text[:idx]
        return text

    def token_nlls(self, text: str, context: str = "") -> TokenNLLs:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always; arrives in Sprint 3.
        """
        raise NotImplementedError("HFClient.token_nlls arrives in Sprint 3")

    def unload(self) -> None:
        """See `ModelClient.unload`. Drops the model reference and clears the CUDA cache if used.

        Side Effects:
            Deletes `self.model` and frees CUDA memory when a GPU is available.
        """
        del self.model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# model_id -> HF Hub repo id, for the fixed Pythia scaling ladder (project_summary.md).
_PYTHIA_HF_REPOS = {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
}


def _make_hf_factory(hf_repo: str) -> Callable[..., ModelClient]:
    """Build a registry factory that constructs an `HFClient` bound to `hf_repo`.

    Args:
        hf_repo: Hugging Face Hub repo id the returned factory will load.

    Returns:
        A factory callable compatible with `register`/`get_client`.
    """

    def factory(revision: str = "main", **kwargs) -> ModelClient:
        return HFClient(hf_repo, revision=revision, **kwargs)

    return factory


register("dummy", DummyClient)
for _model_id, _hf_repo in _PYTHIA_HF_REPOS.items():
    register(_model_id, _make_hf_factory(_hf_repo))
