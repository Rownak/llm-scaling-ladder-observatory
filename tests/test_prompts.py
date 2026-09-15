"""PromptVariant loading + renderer golden test (architecture.md §5)."""

import random

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


def test_render_golden_arc_easy_option_text_v1():
    loader = get_loader("arc_easy")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("arc_easy/mc_option_text_v1")

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
    assert request.prompt_variant_id == "arc_easy/mc_option_text_v1"
    assert request.kind == "loglik"
    assert request.continuations == [
        " Sunlight is the source of energy for nearly all ecosystems.",
        " Most ecosystems are found on land instead of in water.",
        " Carbon dioxide is more available than other gases.",
        " The producers in all ecosystems are plants.",
    ]
    assert request.gen_params is None


def test_render_golden_arc_easy_letter_instr_v1():
    loader = get_loader("arc_easy")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("arc_easy/mc_letter_instr_v1")

    request = render(example, variant)

    expected_prompt = (
        "Answer the following multiple choice question by choosing the letter "
        "of the correct option.\n"
        "Question: Which statement best explains why photosynthesis is the "
        "foundation of most food webs?\n"
        "A. Sunlight is the source of energy for nearly all ecosystems.\n"
        "B. Most ecosystems are found on land instead of in water.\n"
        "C. Carbon dioxide is more available than other gases.\n"
        "D. The producers in all ecosystems are plants.\n"
        "Answer:\n"
    )
    assert request.prompt == expected_prompt
    assert request.prompt_variant_id == "arc_easy/mc_letter_instr_v1"
    assert request.kind == "loglik"
    assert request.continuations == [" A", " B", " C", " D"]
    assert request.gen_params is None


def test_render_golden_mmlu_option_text_v1():
    loader = get_loader("mmlu")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("mmlu/mc_option_text_v1")

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
    assert request.prompt_variant_id == "mmlu/mc_option_text_v1"
    assert request.kind == "loglik"
    assert request.continuations == [
        " This type occurs in binary systems.",
        " This type occurs in young galaxies.",
        " This type produces gamma-ray bursts.",
        " This type produces high amounts of X-rays.",
    ]
    assert request.gen_params is None


def test_load_variant_lambada_cloze_v1():
    variant = load_variant("lambada/cloze_v1")
    assert variant.id == "lambada/cloze_v1"
    assert variant.task_family == "cloze"
    assert variant.continuation_style is None
    assert variant.num_fewshot == 0


def test_render_golden_first_lambada_fixture_example():
    loader = get_loader("lambada")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("lambada/cloze_v1")

    request = render(example, variant)

    assert request.prompt == "She looked out the window and smiled at the falling"
    assert request.example_id == example.example_id
    assert request.prompt_variant_id == "lambada/cloze_v1"
    assert request.kind == "generate"
    assert request.continuations is None
    assert request.gen_params is None


def test_load_variant_gsm8k_gen_v1():
    variant = load_variant("gsm8k/gen_v1")
    assert variant.id == "gsm8k/gen_v1"
    assert variant.task_family == "generative"
    assert variant.continuation_style is None
    assert variant.num_fewshot == 0
    assert variant.max_new_tokens == 256
    assert variant.stop == ["Question:"]


def test_render_golden_first_gsm8k_fixture_example():
    loader = get_loader("gsm8k")
    example = next(loader.load("fixture", limit=1))
    variant = load_variant("gsm8k/gen_v1")

    request = render(example, variant)

    assert "Natalia sold clips" in request.prompt
    assert 'Final answer: <number>' in request.prompt
    assert request.example_id == example.example_id
    assert request.prompt_variant_id == "gsm8k/gen_v1"
    assert request.kind == "generate"
    assert request.continuations is None
    assert request.gen_params == {"max_new_tokens": 256, "stop": ["Question:"], "temperature": 0.0}


def test_load_variant_arc_easy_mc_letter_5shot_v1():
    variant = load_variant("arc_easy/mc_letter_5shot_v1")
    assert variant.id == "arc_easy/mc_letter_5shot_v1"
    assert variant.task_family == "mc"
    assert variant.continuation_style == "letter"
    assert variant.num_fewshot == 5
    assert variant.fewshot_split == "train"
    assert variant.fewshot_seed == 0


def _fewshot_variant_for_fixture_tier():
    """The real `arc_easy/mc_letter_5shot_v1` variant with `fewshot_split`
    overridden to the offline fixture tier ("fixture_train", see datasets.py)
    instead of the real HF "train" split, so tests run network-disabled."""
    variant = load_variant("arc_easy/mc_letter_5shot_v1")
    return variant.model_copy(update={"fewshot_split": "fixture_train"})


def test_render_golden_5shot_prompt():
    loader = get_loader("arc_easy")
    variant = _fewshot_variant_for_fixture_tier()

    pool = list(loader.load(variant.fewshot_split))
    rng = random.Random(variant.fewshot_seed)
    demos = rng.sample(pool, variant.num_fewshot)

    example = next(loader.load("fixture", limit=1))
    request = render(example, variant, demos=demos)

    expected_prompt = (
        "Question: A student wants to determine which type of soil holds the most water. "
        "What should the student measure?\n"
        "A. the color of each soil sample\n"
        "B. the mass of water absorbed by each soil sample\n"
        "C. the temperature of each soil sample\n"
        "D. the number of soil samples tested\n"
        "Answer:\n B\n\n"
        "Question: Which of the following is a renewable source of energy?\n"
        "A. coal\n"
        "B. natural gas\n"
        "C. solar power\n"
        "D. petroleum\n"
        "Answer:\n C\n\n"
        "Question: Which form of energy is produced when a rubber band is stretched?\n"
        "A. chemical energy\n"
        "B. electrical energy\n"
        "C. potential energy\n"
        "D. radiant energy\n"
        "Answer:\n C\n\n"
        "Question: Which of these is a physical change?\n"
        "A. burning wood\n"
        "B. rusting iron\n"
        "C. melting ice\n"
        "D. baking a cake\n"
        "Answer:\n C\n\n"
        "Question: Which body system is primarily responsible for removing carbon dioxide from the blood?\n"
        "A. digestive system\n"
        "B. respiratory system\n"
        "C. skeletal system\n"
        "D. muscular system\n"
        "Answer:\n B\n\n"
        "Question: Which statement best explains why photosynthesis is the foundation of most food webs?\n"
        "A. Sunlight is the source of energy for nearly all ecosystems.\n"
        "B. Most ecosystems are found on land instead of in water.\n"
        "C. Carbon dioxide is more available than other gases.\n"
        "D. The producers in all ecosystems are plants.\n"
        "Answer:\n"
    )
    assert request.prompt == expected_prompt
    assert request.example_id == example.example_id
    assert request.prompt_variant_id == "arc_easy/mc_letter_5shot_v1"
    assert request.kind == "loglik"
    assert request.continuations == [" A", " B", " C", " D"]
    assert request.gen_params is None


def test_render_5shot_is_deterministic_across_two_calls():
    loader = get_loader("arc_easy")
    variant = _fewshot_variant_for_fixture_tier()

    pool = list(loader.load(variant.fewshot_split))

    def draw_demos():
        rng = random.Random(variant.fewshot_seed)
        return rng.sample(pool, variant.num_fewshot)

    example = next(loader.load("fixture", limit=1))
    request_a = render(example, variant, demos=draw_demos())
    request_b = render(example, variant, demos=draw_demos())

    assert request_a == request_b


def test_fewshot_demo_selection_via_evaluators_is_deterministic():
    from ladder.evaluators import _select_fewshot_demos

    variant = _fewshot_variant_for_fixture_tier()

    demos_a = _select_fewshot_demos(variant)
    demos_b = _select_fewshot_demos(variant)

    assert demos_a is not None
    assert len(demos_a) == 5
    assert [d.example_id for d in demos_a] == [d.example_id for d in demos_b]


def test_select_fewshot_demos_returns_none_for_zero_shot_variant():
    from ladder.evaluators import _select_fewshot_demos

    variant = load_variant("arc_easy/mc_letter_v1")
    assert _select_fewshot_demos(variant) is None


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
