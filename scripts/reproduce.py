#!/usr/bin/env python3
"""One-command, self-contained artifact reproduction workflow."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Keep the distributed tree free of machine-specific compiled-path metadata.
sys.dont_write_bytecode = True

from artifact_paths import REPO, TRACES

SCRIPTS = Path(__file__).resolve().parent
GENERATORS = [
    "table_directions.py", "table_rates.py", "table_reception.py",
    "table_wilson.py", "table_structural.py", "table_discordant_flows.py",
    "table_flow_specifications.py", "table_headroom.py",
    "table_rounds.py", "table_ceiling.py", "table_confirmatory.py",
    "table_composition_accounting.py",
    "table_musique.py", "table_musique_heterogeneous.py",
    "table_specifications.py", "table_logistic.py",
    "table_gate.py", "table_agent_diversity_ablation.py",
    "table_role_effects.py", "table_sensitivity.py",
    "prose_quantities.py", "design_quantities.py", "plot_data.py",
    "same_family.py", "token_cost.py", "release_quantities.py",
]


def run(command, env):
    print("+", " ".join(str(x) for x in command), flush=True)
    result = subprocess.run(command, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="regenerate and verify all bundled results from the traces")
    parser.add_argument("--output", type=Path, default=REPO / "reproduced")
    args = parser.parse_args()
    output = args.output.resolve()
    protected = (REPO / "traces", REPO / "results",
                 REPO / "scripts", REPO / "claims")
    if output == REPO or any(output == path or path in output.parents
                             for path in protected):
        parser.error("--output must not overwrite the released bundle")

    results, figures = output / "results", output / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["BORROWED_STRENGTH_TRACES"] = str(TRACES)
    env["BORROWED_STRENGTH_RESULTS"] = str(results)
    env["MPLCONFIGDIR"] = str(output / ".matplotlib")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SOURCE_DATE_EPOCH"] = "0"

    run([sys.executable, SCRIPTS / "trace_manifest.py"], env)
    for name in GENERATORS:
        run([sys.executable, SCRIPTS / name], env)
    run([sys.executable, SCRIPTS / "figures.py", figures], env)
    run([sys.executable, SCRIPTS / "verify_artifact.py", figures], env)
    print("REPRODUCTION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
