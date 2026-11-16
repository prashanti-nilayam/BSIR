"""Trace loading for the Borrowed Strength result scripts.

Records are (item_id, round, agent_id) triples. Resumed runs can write the same
triple more than once; the last record written wins, matching the indexing
policy used to produce the manuscript.

Answers are re-scored from `raw_output` with the benchmark-appropriate parser
rather than trusting any stored correctness flag.
"""
from __future__ import annotations

import json
from pathlib import Path

from artifact_paths import TRACES
import scoring

# benchmark -> (parse_output, parse_gold, answers_match).
# Mirrors the adapter definitions used to produce the manuscript: MATH and
# MuSiQue supply gold directly, so their parse_gold is a pass-through; only
# GSM8K stores gold inside a worked solution that must be parsed out.
_passthrough = lambda s: s if s else None


def _gsm8k_gold(s):
    """Traces store the extracted gold, but some carry the full '#### n'
    solution. Try the solution parser, fall back to the value as given."""
    if not s:
        return None
    return scoring.parse_gold(s) or s


ADAPTERS = {
    "math": (scoring.parse_math_answer, _passthrough,
             scoring.math_answers_match),
    "gsm8k": (scoring.parse_model_output, _gsm8k_gold,
              scoring.answers_match),
    "musique": (scoring.parse_qa_answer, _passthrough,
                scoring.musique_answers_match),
}


def find(name: str) -> Path:
    """Locate a trace by stem, searching both released groups."""
    for group in ("experiment", "control"):
        p = TRACES / group / f"{name}.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.jsonl not under {TRACES}")


def load(name: str, benchmark: str) -> dict:
    """Return {(item_id, round, agent_id): record} with answers re-scored.

    Each record gains `ok` (bool): the re-scored correctness of that response.
    """
    parse_out, parse_gold, match = ADAPTERS[benchmark]
    records: dict[tuple, dict] = {}
    with find(name).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["item_id"], int(r["round"]), int(r["agent_id"]))
            records[key] = r                      # last record wins

    for r in records.values():
        pred = parse_out(r.get("raw_output") or "")
        gold = parse_gold(r.get("gold_answer"))
        r["ok"] = bool(pred is not None and gold is not None
                       and match(pred, gold))
    return records


def by_item_round(records: dict) -> dict:
    """Regroup into {(item_id, round): {agent_id: record}}."""
    out: dict[tuple, dict] = {}
    for (item, rnd, agent), r in records.items():
        out.setdefault((item, rnd), {})[agent] = r
    return out


def complete_items(records: dict, n_agents: int, rounds: range) -> set:
    """Items with every agent present in every requested round."""
    grouped = by_item_round(records)
    items = {i for (i, _) in grouped}
    return {
        i for i in items
        if all(len(grouped.get((i, rd), {})) == n_agents for rd in rounds)
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name}  ({len(rows)} rows)")
