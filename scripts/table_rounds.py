#!/usr/bin/env python3
"""Round-by-round accuracy in the five full-reference settings.

Series per setting:
  peer_alone    -- peer model in its own homogeneous panel
  peer_in_panel -- peer in slot 2 of the honest heterogeneous panel
  defenders     -- defender slots 0,1 of that same panel
  panel_two     -- at least two of three agents correct
  oracle        -- any one agent correct

Output: results/rounds.csv
"""
from artifact_paths import RESULTS
import panels
import traces

ROUNDS = range(5)
N_AGENTS = 3
OUT = RESULTS / "rounds.csv"

# label -> direction key, in table order
SETTINGS = [
    ("GPT-OSS-120B -> Llama-70B",      "llama70b_gptoss_math"),
    ("GPT-OSS-120B -> GPT-4.1",        "gpt41_gptoss_math"),
    ("Gemini-2.5-Flash -> GPT-4.1",    "gpt41_gemini_math"),
    ("GPT-4.1 -> Gemini-2.5-Flash",    "gemini_gpt41_math"),
    ("Gemma-2B -> Llama-1B",           "llama1b_gemma2b_gsm8k"),
]


def series(direction):
    """Per-round rates plus the matched replicate-item count."""
    bench = direction["benchmark"]
    d_homo = [traces.load(r, bench) for r in direction["d_homo"]]
    mixed = [traces.load(r, bench) for r in direction["mixed"]]
    p_homo = [traces.load(r, bench) for r in direction["p_homo"]]

    # Full-trajectory support. Only the heterogeneous panel and the peer's own
    # reference enter here -- no series in this table comes from [D,D,D].
    support = []
    for i in range(len(mixed)):
        sup = traces.complete_items(mixed[i], N_AGENTS, ROUNDS)
        ph = p_homo[i] if i < len(p_homo) else p_homo[0]
        # A full-reference setting needs the peer's own trajectory too. The
        # small-model diagnostic's reference is a single-agent R0 capture, so
        # there it can only constrain by item presence.
        p_agents = {a for (_, _, a) in ph}
        p_rounds = {rd for (_, rd, _) in ph}
        if len(p_agents) == N_AGENTS and set(ROUNDS) <= p_rounds:
            sup &= traces.complete_items(ph, N_AGENTS, ROUNDS)
        else:
            sup &= {it for (it, _, _) in ph}
        support.append(sup)
    n_items = sum(len(s) for s in support)

    out = {k: {} for k in ("peer_alone", "peer_in_panel", "defenders",
                           "panel_two", "oracle")}
    for rd in ROUNDS:
        acc = {k: [0, 0] for k in out}
        for i, sup in enumerate(support):
            mx = mixed[i]
            ph = p_homo[i] if i < len(p_homo) else p_homo[0]
            p_slots = sorted({a for (_, _, a) in ph})
            for it in sup:
                for s in p_slots:                      # peer alone
                    r = ph.get((it, rd, s))
                    if r is not None:
                        acc["peer_alone"][0] += r["ok"]
                        acc["peer_alone"][1] += 1
                peer = mx.get((it, rd, panels.PEER_SLOT))
                if peer is not None:
                    acc["peer_in_panel"][0] += peer["ok"]
                    acc["peer_in_panel"][1] += 1
                n_ok = 0
                for s in panels.DEFENDER_SLOTS:
                    r = mx.get((it, rd, s))
                    if r is not None:
                        acc["defenders"][0] += r["ok"]
                        acc["defenders"][1] += 1
                        n_ok += r["ok"]
                total_ok = n_ok + (peer["ok"] if peer is not None else 0)
                acc["panel_two"][0] += int(total_ok >= 2)
                acc["panel_two"][1] += 1
                acc["oracle"][0] += int(total_ok >= 1)
                acc["oracle"][1] += 1
        for k, (h, t) in acc.items():
            out[k][rd] = (100.0 * h / t) if t else None
    return out, n_items


def main() -> None:
    rows = []
    for label, key in SETTINGS:
        d = next(x for x in panels.DIRECTIONS if x["key"] == key)
        vals, n_items = series(d)
        for name, per_round in vals.items():
            row = {"setting": label, "series": name, "replicate_items": n_items}
            for rd in ROUNDS:
                v = per_round[rd]
                row[f"R{rd}"] = "" if v is None else round(v, 1)
            rows.append(row)
            print(f"  {label:30s} {name:14s} "
                  + "  ".join(f"{row[f'R{rd}']:>5}" for rd in ROUNDS))

    traces.write_csv(OUT, ["setting", "series", "replicate_items"]
                     + [f"R{rd}" for rd in ROUNDS], rows)


if __name__ == "__main__":
    main()
