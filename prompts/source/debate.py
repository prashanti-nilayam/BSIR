"""Du et al. (2023) multi-agent debate runner.

Preserves the prompt templates from `vendor/llm_multiagent_debate/gsm/gen_gsm.py`
verbatim. Ports the OpenAI call to the modern SDK. Adds per-round per-agent
trajectory logging via the divergent_regime.logging schema.

Protocol:
- K agents see the same initial question and each independently produce a round-0 answer.
- At round r > 0, each agent is shown the other agents' round-(r-1) answers and asked to
  reconsider, producing a round-r answer.
- We run R total rounds (round indices 0, 1, ..., R-1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .benchmarks import GSM8K, BenchmarkAdapter
from .llm_client import chat_complete
from .logging import TrajectoryLogger, TrajectoryRecord
import hashlib
import random


class _Item(Protocol):
    item_id: str
    question: str
    gold_answer: str


def _initial_user_message(question: str, adapter: BenchmarkAdapter) -> dict[str, str]:
    return {"role": "user", "content": adapter.initial_prompt(question)}


def _debate_message(
    other_agent_contexts: list[list[dict[str, str]]],
    question: str,
    prev_idx: int,
    adapter: BenchmarkAdapter = GSM8K,
) -> dict[str, str]:
    """Construct the round-r user message containing other agents' round-(r-1) answers.

    `prev_idx` is the index into each agent's message history of the previous assistant
    answer. Matches Du et al.'s convention: history is [user_q, asst_r0, user_r1, asst_r1, ...],
    so the assistant turn from round r-1 is at index 2*r - 1.

    `adapter` supplies the benchmark-specific `final_answer_instruction` so the model
    is told the correct answer format (e.g. "single letter A-D" for GPQA instead of
    Du's GSM8K default "single numerical number").
    """
    prefix = "These are the solutions to the problem from other agents: "
    for ctx in other_agent_contexts:
        prefix += "\n\n One agent solution: ```{}```".format(ctx[prev_idx]["content"])
    prefix += (
        "\n\n Using the solutions from other agents as additional information, can you "
        "provide your answer to the problem? \n The original problem is {q}. {fmt}"
    ).format(q=question, fmt=adapter.final_answer_instruction)
    return {"role": "user", "content": prefix}


@dataclass
class DebateConfig:
    model: str | list[str]
    adapter: BenchmarkAdapter = GSM8K
    n_agents: int = 3
    n_rounds: int = 3
    temperature: float = 0.0
    # Optional per-agent temperatures. When set, entry i overrides `temperature`
    # for agent i (used to inject sampling diversity within an otherwise
    # homogeneous panel). Length must equal n_agents.
    temperatures: list[float] | None = None
    max_tokens: int = 1024
    # Optional per-agent system prompts. Each entry is a callable taking the
    # benchmark item and returning the system-prompt string (or None for no
    # prompt). Used to inject role-specific instructions for adversarial
    # agents that need item-specific context (e.g., gold answer).
    agent_system_prompts: list[Callable[[Any], str | None] | None] | None = None
    # Optional per-agent initial-user-message factories. When set for slot i,
    # the standard adapter.initial_prompt(question) message is REPLACED by
    # the output of the factory. Used by adversarial agents that need a
    # self-contained R0 prompt (avoids conflict between system and user
    # instructions on heavily-aligned models). Later-round messages still go
    # through _debate_message unchanged.
    agent_initial_user_factories: list[Callable[[Any], str | None] | None] | None = None
    # Optional per-agent round-message factories. When set for slot i, the
    # standard _debate_message(...) round prompt is REPLACED by the factory
    # output for that agent. Signature: (other_agent_contexts, question,
    # prev_idx, adapter) -> {"role": "user", "content": str}. Used to give each
    # agent slightly-reworded reconsideration instructions (prompt-diversity).
    agent_debate_message_factories: (
        list[Callable[[list, str, int, BenchmarkAdapter], dict[str, str]] | None] | None
    ) = None
    # Adversary compliance retry: when a weak adversary fails to commit to its
    # assigned wrong target (defects to gold or drifts), regenerate that slot's
    # response up to N times with an escalated commitment prompt. Escalates
    # COMMITMENT only (not argument quality) so attack strength is held constant.
    adversary_slot: int | None = None
    adversary_target_fn: Callable[[Any], str | None] | None = None
    adversary_compliance_retries: int = 0
    # Per-agent visibility quorum. None (default) = every agent sees every other
    # agent's previous response, i.e. the standard fully-connected protocol.
    # An int q means each agent sees only q of the n_agents-1 others, drawn
    # independently PER AGENT (so no two agents share a context) and PER ROUND.
    #
    # The draw is a deterministic function of (quorum_seed, item_id, round,
    # agent_id), so the same seed reproduces the same masks exactly and
    # replicates remain comparable. Applied uniformly to all slots including the
    # adversary: restricting what the adversary *sees* would be free defense in
    # this threat model and is not realistic for a compromised endpoint.
    quorum: int | None = None
    quorum_seed: int = 0

    def __post_init__(self) -> None:
        if self.quorum is not None:
            if not (1 <= self.quorum <= self.n_agents - 1):
                raise ValueError(
                    f"quorum {self.quorum} must be in [1, n_agents-1] "
                    f"= [1, {self.n_agents - 1}]"
                )
        if isinstance(self.model, list):
            if len(self.model) != self.n_agents:
                raise ValueError(
                    f"DebateConfig.model list length {len(self.model)} does not match n_agents {self.n_agents}"
                )
        if self.agent_system_prompts is not None:
            if len(self.agent_system_prompts) != self.n_agents:
                raise ValueError(
                    f"agent_system_prompts length {len(self.agent_system_prompts)} does not match n_agents {self.n_agents}"
                )
        if self.agent_initial_user_factories is not None:
            if len(self.agent_initial_user_factories) != self.n_agents:
                raise ValueError(
                    f"agent_initial_user_factories length {len(self.agent_initial_user_factories)} does not match n_agents {self.n_agents}"
                )
        if self.temperatures is not None:
            if len(self.temperatures) != self.n_agents:
                raise ValueError(
                    f"temperatures length {len(self.temperatures)} does not match n_agents {self.n_agents}"
                )
        if self.agent_debate_message_factories is not None:
            if len(self.agent_debate_message_factories) != self.n_agents:
                raise ValueError(
                    f"agent_debate_message_factories length {len(self.agent_debate_message_factories)} does not match n_agents {self.n_agents}"
                )

    def model_for_agent(self, agent_id: int) -> str:
        if isinstance(self.model, list):
            return self.model[agent_id]
        return self.model

    def visible_peers_for_agent(
        self, agent_id: int, round_idx: int, item_id: str
    ) -> list[int]:
        """Peer slots whose previous response agent_id sees this round, ascending.

        With quorum=None this is every other slot (the standard protocol). With
        quorum=q it is a q-subset of the others, drawn independently for each
        (agent, round, item) so that no two agents share a context and the
        assignment cannot be anticipated from another agent's view.

        Determinism: the draw is seeded by (quorum_seed, item_id, round_idx,
        agent_id) via blake2b, so it is stable across processes, Python versions
        and replicate order. Re-running a config reproduces every mask.
        """
        others = [j for j in range(self.n_agents) if j != agent_id]
        if self.quorum is None or self.quorum >= len(others):
            return others
        key = f"{self.quorum_seed}|{item_id}|{round_idx}|{agent_id}".encode()
        digest = hashlib.blake2b(key, digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        return sorted(rng.sample(others, self.quorum))

    def system_prompt_for_agent(self, agent_id: int, item: Any) -> str | None:
        if self.agent_system_prompts is None:
            return None
        fn = self.agent_system_prompts[agent_id]
        if fn is None:
            return None
        return fn(item)

    def initial_user_for_agent(self, agent_id: int, item: Any) -> str | None:
        if self.agent_initial_user_factories is None:
            return None
        fn = self.agent_initial_user_factories[agent_id]
        if fn is None:
            return None
        return fn(item)

    def temperature_for_agent(self, agent_id: int) -> float:
        if self.temperatures is not None:
            return self.temperatures[agent_id]
        return self.temperature

    def debate_message_for_agent(
        self,
        agent_id: int,
        other_agent_contexts: list,
        question: str,
        prev_idx: int,
        adapter: BenchmarkAdapter,
    ) -> dict[str, str]:
        """Round-r user message for agent_id: per-agent factory if set, else default."""
        if self.agent_debate_message_factories is not None:
            fn = self.agent_debate_message_factories[agent_id]
            if fn is not None:
                return fn(other_agent_contexts, question, prev_idx, adapter)
        return _debate_message(other_agent_contexts, question, prev_idx, adapter)


def run_debate_on_item(
    item: _Item,
    config: DebateConfig,
    logger: TrajectoryLogger | None = None,
) -> list[list[dict[str, str]]]:
    """Run Du-style debate on a single benchmark item.

    Returns the full agent_contexts list (K agents × full conversation history).
    Also emits one TrajectoryRecord per (item, round, agent) if a logger is provided.
    """
    adapter = config.adapter
    gold_parsed = adapter.parse_gold(item.gold_answer)

    agent_contexts: list[list[dict[str, str]]] = []
    for i in range(config.n_agents):
        custom_initial = config.initial_user_for_agent(i, item)
        if custom_initial is not None:
            agent_contexts.append([{"role": "user", "content": custom_initial}])
        else:
            agent_contexts.append([_initial_user_message(item.question, adapter)])
    # System prompts are not stored in agent_contexts (would shift prev_idx
    # used by _debate_message). They are prepended only at LLM-call time.
    agent_sys_prompts: list[str | None] = [
        config.system_prompt_for_agent(i, item) for i in range(config.n_agents)
    ]

    for r in range(config.n_rounds):
        for i in range(config.n_agents):
            if r > 0:
                others = agent_contexts[:i] + agent_contexts[i + 1 :]
                prev_idx = 2 * r - 1
                agent_contexts[i].append(
                    config.debate_message_for_agent(i, others, item.question, prev_idx, adapter)
                )

            agent_model = config.model_for_agent(i)
            sys_prompt = agent_sys_prompts[i]
            call_messages = (
                [{"role": "system", "content": sys_prompt}] + agent_contexts[i]
                if sys_prompt
                else agent_contexts[i]
            )
            result = chat_complete(
                messages=call_messages,
                model=agent_model,
                temperature=config.temperature_for_agent(i),
                max_tokens=config.max_tokens,
            )
            agent_contexts[i].append({"role": "assistant", "content": result.content})

            if logger is not None:
                parsed = adapter.parse_output(result.content)
                logger.write(
                    TrajectoryRecord(
                        item_id=item.item_id,
                        round=r,
                        agent_id=i,
                        raw_output=result.content,
                        parsed_answer=parsed,
                        gold_answer=gold_parsed or "",
                        correct=adapter.answers_match(parsed, gold_parsed),
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        model=agent_model,
                    )
                )

    return agent_contexts
