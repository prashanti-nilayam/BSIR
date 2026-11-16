"""Per-agent prompt-diversity variants for the heterogeneous debate experiment.

Each variant supplies a matched pair:
  - initial(question) -> str          : the round-0 user prompt
  - round_msg(others, question, prev_idx, adapter) -> {"role","content"}
                                        : the round-r>0 reconsideration prompt

All variants preserve the exact answer-format contract:
  - the initial prompt keeps the literal ``\\boxed{answer}`` instruction, and
  - the round message ends with ``adapter.final_answer_instruction`` ({fmt}),
which the output parser depends on. Only the *framing* wording differs across
variants, so answer extraction is unaffected.

VARIANT_SET is indexed by agent id (variant i -> agent i). Mirrors the round
structure of ``debate._debate_message`` exactly (same per-solution formatting
and prev_idx convention) so trajectories stay comparable to the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .benchmarks import BenchmarkAdapter


def _round_content(
    prefix: str,
    closing: str,
    other_agent_contexts: list,
    question: str,
    prev_idx: int,
    adapter: BenchmarkAdapter,
) -> dict[str, str]:
    """Build a round message with variant-specific prefix/closing but the
    baseline per-solution formatting and the mandatory {fmt} instruction."""
    text = prefix
    for ctx in other_agent_contexts:
        text += "\n\n One agent solution: ```{}```".format(ctx[prev_idx]["content"])
    text += closing.format(q=question, fmt=adapter.final_answer_instruction)
    return {"role": "user", "content": text}


@dataclass(frozen=True)
class PromptVariant:
    initial: Callable[[str], str]
    round_msg: Callable[[list, str, int, BenchmarkAdapter], dict[str, str]]


# --- Variant A (agent 0): careful / step-by-step -------------------------------
def _initial_a(question: str) -> str:
    return (
        "Solve this math problem step by step. {q} Work through your reasoning "
        "carefully before answering. Your final answer should be a single "
        "numerical number, in the form \\boxed{{answer}}, at the end of your "
        "response."
    ).format(q=question)


def _round_a(others: list, question: str, prev_idx: int, adapter: BenchmarkAdapter) -> dict[str, str]:
    return _round_content(
        "Other agents produced these solutions to the problem: ",
        "\n\n Review these other solutions carefully, then give your own answer "
        "to the problem. \n The original problem is {q}. {fmt}",
        others, question, prev_idx, adapter,
    )


# --- Variant B (agent 1): independent / critical -------------------------------
def _initial_b(question: str) -> str:
    return (
        "Here is a math problem for you to solve: {q} Reason through it "
        "independently and explain each step. Conclude with your final answer "
        "as a single numerical number in the form \\boxed{{answer}} at the end "
        "of your response."
    ).format(q=question)


def _round_b(others: list, question: str, prev_idx: int, adapter: BenchmarkAdapter) -> dict[str, str]:
    return _round_content(
        "Below are solutions from other agents: ",
        "\n\n Critically evaluate these solutions — do not simply agree — "
        "then provide your own best answer. \n The original problem is {q}. {fmt}",
        others, question, prev_idx, adapter,
    )


# --- Variant C (agent 2): concise / direct -------------------------------------
def _initial_c(question: str) -> str:
    return (
        "Please solve the following math problem: {q} Briefly justify your "
        "reasoning, then state your final answer. Your final answer should be a "
        "single numerical number, in the form \\boxed{{answer}}, at the end of "
        "your response."
    ).format(q=question)


def _round_c(others: list, question: str, prev_idx: int, adapter: BenchmarkAdapter) -> dict[str, str]:
    return _round_content(
        "Here are the answers other agents gave: ",
        "\n\n Taking these into account, decide on your final answer to the "
        "problem. \n The original problem is {q}. {fmt}",
        others, question, prev_idx, adapter,
    )


VARIANT_SET: list[PromptVariant] = [
    PromptVariant(initial=_initial_a, round_msg=_round_a),
    PromptVariant(initial=_initial_b, round_msg=_round_b),
    PromptVariant(initial=_initial_c, round_msg=_round_c),
]
