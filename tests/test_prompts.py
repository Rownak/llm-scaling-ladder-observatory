"""PromptVariant loading + renderer golden test (architecture.md §5)."""

from ladder.datasets import get_loader
from ladder.prompts import PromptVariant, load_variant, render
from ladder.records import Example


def test_load_variant_mc_letter_v1():
    variant = load_variant("arc_easy/mc_letter_v1")
    assert variant.id == "arc_easy/mc_letter_v1"
    assert variant.task_family == "mc"
    assert variant.continuation_style == "letter"
    assert variant.num_fewshot == 0


def test_render_golden_first_fixture_example():
    loader = get_loader("arc_easy")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("arc_easy/mc_letter_v1")

    request = render(example, variant)

    expected_prompt = (
        "Question: Which statement best explains why photosynthesis is the "
        "foundation of most food webs?\n"
        "A. Sunlight is the source of energy for nearly all ecosystems.\n"
        "B. Most ecosystems are found on land instead of in water.\n"
        "C. Carbon dioxide is more available than other gases.\n"
        "D. The producers in all ecosystems are plants.\n"
        "Answer:\n"
    )
    assert request.prompt == expected_prompt
    assert request.example_id == example.example_id
    assert request.prompt_variant_id == "arc_easy/mc_letter_v1"
    assert request.kind == "loglik"
    assert request.continuations == [" A", " B", " C", " D"]
    assert request.gen_params is None


def test_render_option_text_continuation_style():
    example = Example(
        dataset="arc_easy",
        split="fixture",
        example_id="test-1",
        payload={"question": "Q?", "choices": ["foo", "bar"], "answer_index": 0},
    )
    variant = PromptVariant(
        id="test/mc_option_text_v1",
        task_family="mc",
        template="Question: {question}\n{lettered_choices}\nAnswer:\n",
        continuation_style="option_text",
        num_fewshot=0,
    )
    request = render(example, variant)
    assert request.continuations == [" foo", " bar"]


def test_load_variant_hellaswag_mc_context_v1():
    variant = load_variant("hellaswag/mc_context_v1")
    assert variant.id == "hellaswag/mc_context_v1"
    assert variant.task_family == "mc"
    assert variant.continuation_style == "option_text"
    assert variant.num_fewshot == 0


def test_render_golden_hellaswag_first_fixture_example():
    loader = get_loader("hellaswag")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("hellaswag/mc_context_v1")

    request = render(example, variant)

    assert request.prompt == "A man is sitting on a roof. he\n"
    assert request.example_id == example.example_id
    assert request.prompt_variant_id == "hellaswag/mc_context_v1"
    assert request.kind == "loglik"
    assert request.continuations == [
        " is using wrap to wrap a pair of skis.",
        " is ripping level tiles off.",
        " is holding a rubik's cube.",
        " starts pulling up roofing on a roof.",
    ]
    assert request.gen_params is None


def test_load_variant_mmlu_mc_letter_v1():
    variant = load_variant("mmlu/mc_letter_v1")
    assert variant.id == "mmlu/mc_letter_v1"
    assert variant.task_family == "mc"
    assert variant.continuation_style == "letter"
    assert variant.num_fewshot == 0


def test_render_golden_mmlu_first_fixture_example():
    loader = get_loader("mmlu")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("mmlu/mc_letter_v1")

    request = render(example, variant)

    expected_prompt = (
        'Question: What is true for a type-Ia ("type one-a") supernova?\n'
        "A. This type occurs in binary systems.\n"
        "B. This type occurs in young galaxies.\n"
        "C. This type produces gamma-ray bursts.\n"
        "D. This type produces high amounts of X-rays.\n"
        "Answer:\n"
    )
    assert request.prompt == expected_prompt
    assert request.example_id == example.example_id
    assert request.prompt_variant_id == "mmlu/mc_letter_v1"
    assert request.kind == "loglik"
    assert request.continuations == [" A", " B", " C", " D"]
    assert request.gen_params is None


def test_render_is_pure_no_shared_mutation():
    loader = get_loader("arc_easy")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("arc_easy/mc_letter_v1")

    original_choices = list(example.payload["choices"])

    request_a = render(example, variant)
    request_b = render(example, variant)

    assert request_a == request_b
    # Rendering must not mutate the input Example's payload.
    assert example.payload["choices"] == original_choices
