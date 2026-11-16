"""Benchmark adapters bundling the per-benchmark protocol pieces.

Each adapter provides:
- initial_prompt: callable(question) -> str
- parse_output:   callable(raw_output) -> parsed_answer (str or None)
- parse_gold:     callable(gold_field) -> normalized gold (str or None)
- answers_match:  callable(pred, gold) -> bool

Loaders live in data.py (GSM8K) and math_data.py (MATH); the adapter is purely
about prompts and answer comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .parse import (
    answers_match,
    math_answers_match,
    mcq_answers_match,
    musique_answers_match,
    parse_gold,
    parse_math_answer,
    parse_mcq_answer,
    parse_model_output,
    parse_qa_answer,
    scibench_answers_match,
)


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str
    initial_prompt: Callable[[str], str]
    parse_output: Callable[[str], str | None]
    parse_gold: Callable[[str], str | None]
    answers_match: Callable[[str | None, str | None], bool]
    # Boilerplate appended to debate/review prompts so the model knows what
    # form the final answer should take. Du's original wording assumed GSM8K
    # ("single numerical number"); per-adapter override fixes prompt-drift on
    # non-numeric benchmarks like GPQA (multiple-choice).
    final_answer_instruction: str = (
        "Your final answer should be a single numerical number, in the form "
        "\\boxed{answer}, at the end of your response."
    )


def _gsm8k_initial_prompt(question: str) -> str:
    return (
        "Can you solve the following math problem? {q} Explain your reasoning. "
        "Your final answer should be a single numerical number, in the form "
        "\\boxed{{answer}}, at the end of your response."
    ).format(q=question)


def _math_initial_prompt(question: str) -> str:
    return (
        "Can you solve the following math problem? {q} Explain your reasoning. "
        "Your final answer should be in the form \\boxed{{answer}}, at the end of "
        "your response."
    ).format(q=question)


def _aime_initial_prompt(question: str) -> str:
    return (
        "Solve the following AIME problem. The answer is an integer between 0 and "
        "999 inclusive. {q} Explain your reasoning. Your final answer must be in "
        "the form \\boxed{{N}} where N is an integer, at the end of your response."
    ).format(q=question)


def _gpqa_initial_prompt(question: str) -> str:
    return (
        "The following is a multiple-choice question. Select the single best "
        "answer from options A, B, C, or D.\n\n{q}\n\n"
        "Explain your reasoning. Your final answer must be in the form "
        "\\boxed{{X}} where X is a single letter (A, B, C, or D), at the end "
        "of your response."
    ).format(q=question)


GSM8K = BenchmarkAdapter(
    name="gsm8k",
    initial_prompt=_gsm8k_initial_prompt,
    parse_output=parse_model_output,
    parse_gold=parse_gold,
    answers_match=answers_match,
)


MATH = BenchmarkAdapter(
    name="math",
    initial_prompt=_math_initial_prompt,
    parse_output=parse_math_answer,
    parse_gold=lambda s: s if s else None,  # MATH-500 supplies the answer directly
    answers_match=math_answers_match,
)


AIME = BenchmarkAdapter(
    name="aime",
    initial_prompt=_aime_initial_prompt,
    parse_output=parse_math_answer,
    parse_gold=lambda s: s if s else None,  # AIME answers stored as integer strings
    answers_match=math_answers_match,
)


GPQA = BenchmarkAdapter(
    name="gpqa",
    initial_prompt=_gpqa_initial_prompt,
    parse_output=parse_mcq_answer,
    parse_gold=lambda s: s.strip().upper() if s else None,
    answers_match=mcq_answers_match,
    final_answer_instruction=(
        "Your final answer must be a single letter (A, B, C, or D) in the "
        "form \\boxed{X}, at the end of your response."
    ),
)


def _scibench_initial_prompt(question: str) -> str:
    return (
        "Solve the following college-level science problem. {q} Explain your "
        "reasoning. Report the final numeric answer (in the problem's expected "
        "units) in the form \\boxed{{answer}}, at the end of your response."
    ).format(q=question)


SCIBENCH = BenchmarkAdapter(
    name="scibench",
    initial_prompt=_scibench_initial_prompt,
    parse_output=parse_math_answer,
    parse_gold=lambda s: s if s else None,  # gold supplied as numeric string
    answers_match=scibench_answers_match,
    final_answer_instruction=(
        "Your final answer should be the numeric result in the form "
        "\\boxed{answer}, at the end of your response."
    ),
)


def _musique_initial_prompt(question: str) -> str:
    # `question` already contains the 20 paragraphs + the multi-hop question
    # (composed in musique_data.py). We only add task + answer-format framing.
    return (
        "{q}\n\nAnswer the question using only the paragraphs above; the answer is "
        "a short phrase found by reasoning across multiple paragraphs. Explain your "
        "reasoning, then give your final answer as a short phrase in the form "
        "\\boxed{{answer}}, at the end of your response."
    ).format(q=question)


MUSIQUE = BenchmarkAdapter(
    name="musique",
    initial_prompt=_musique_initial_prompt,
    parse_output=parse_qa_answer,
    parse_gold=lambda s: s if s else None,  # gold is "answer ||| alias1 ||| ..."
    answers_match=musique_answers_match,  # normalized EM over answer+aliases
    final_answer_instruction=(
        "Your final answer should be a short answer phrase, in the form "
        "\\boxed{answer}, at the end of your response."
    ),
)


ADAPTERS: dict[str, BenchmarkAdapter] = {
    "gsm8k": GSM8K,
    "math": MATH,
    "aime": AIME,
    "gpqa": GPQA,
    "scibench": SCIBENCH,
    "musique": MUSIQUE,
}


def get_adapter(name: str) -> BenchmarkAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"Unknown benchmark: {name!r}. Known: {sorted(ADAPTERS)}")
    return ADAPTERS[name]
