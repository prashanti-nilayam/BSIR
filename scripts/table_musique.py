#!/usr/bin/env python3
"""Open-answer control: 300 MuSiQue-Ans items, homogeneous Gemma-3-12B panel.

Alias-normalized exact match and token F1, recomputed from raw answers.
Output: results/musique.csv
"""
from artifact_paths import RESULTS
import scoring
import traces

RUN = "musique_gemma3_homo_5round_rep1"
ROUNDS = range(5)
N_AGENTS = 3
OUT = RESULTS / "musique.csv"


def main() -> None:
    records = traces.load(RUN, "musique")
    grouped = traces.by_item_round(records)
    items = sorted(traces.complete_items(records, N_AGENTS, ROUNDS))

    rows = []
    for rd in ROUNDS:
        em_hits = two = orc = 0
        f1_sum = 0.0
        n_resp = 0
        for item in items:
            agents = grouped[(item, rd)]
            n_correct = 0
            for a in agents.values():
                pred = scoring.parse_qa_answer(a.get("raw_output") or "")
                gold = a.get("gold_answer")
                n_correct += int(a["ok"])
                f1_sum += scoring.qa_token_f1(pred or "", gold or "")
                n_resp += 1
            em_hits += n_correct
            two += int(n_correct >= 2)
            orc += int(n_correct >= 1)
        rows.append({
            "round": f"R{rd}",
            "items": len(items),
            "pooled_em": round(100.0 * em_hits / n_resp, 2),
            "at_least_two": round(100.0 * two / len(items), 2),
            "oracle": round(100.0 * orc / len(items), 2),
            "f1": round(100.0 * f1_sum / n_resp, 2),
        })

    traces.write_csv(OUT, ["round", "items", "pooled_em", "at_least_two",
                           "oracle", "f1"], rows)
    for r in rows:
        print(f"  {r['round']}  EM {r['pooled_em']:6.2f}  >=2 {r['at_least_two']:6.2f}"
              f"  oracle {r['oracle']:6.2f}  F1 {r['f1']:6.2f}")


if __name__ == "__main__":
    main()
