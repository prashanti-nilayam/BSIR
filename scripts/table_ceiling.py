#!/usr/bin/env python3
"""Where the panel stops: best-round values in the five full-reference settings.

Each column is the maximum over rounds of the corresponding series in
results/rounds.csv. Shortfall is the >=2 panel minus the oracle.

Output: results/ceiling.csv
"""
from artifact_paths import RESULTS
import panels
import table_rounds
import traces

OUT = RESULTS / "ceiling.csv"


def main() -> None:
    rows = []
    for label, key in table_rounds.SETTINGS:
        d = next(x for x in panels.DIRECTIONS if x["key"] == key)
        vals, n_items = table_rounds.series(d)
        best = {k: max(v for v in per.values() if v is not None)
                for k, per in vals.items()}
        shortfall = best["panel_two"] - best["oracle"]
        rows.append({
            "setting": label,
            "n": n_items,
            "peer_alone": round(best["peer_alone"], 1),
            "peer_in_panel": round(best["peer_in_panel"], 1),
            "defenders": round(best["defenders"], 1),
            "panel_two": round(best["panel_two"], 1),
            "oracle": round(best["oracle"], 1),
            "shortfall": round(shortfall, 1),
        })
        print(f"  {label:30s} n={n_items:5d}  alone={best['peer_alone']:5.1f}  "
              f"in_panel={best['peer_in_panel']:5.1f}  def={best['defenders']:5.1f}  "
              f">=2={best['panel_two']:5.1f}  oracle={best['oracle']:5.1f}  "
              f"short={shortfall:+6.1f}")

    traces.write_csv(OUT, ["setting", "n", "peer_alone", "peer_in_panel",
                           "defenders", "panel_two", "oracle", "shortfall"], rows)


if __name__ == "__main__":
    main()
