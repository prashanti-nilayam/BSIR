#!/usr/bin/env python3
"""Strict, self-contained verification of regenerated reviewer artifacts."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from artifact_paths import REPO, RESULTS

REFERENCE_RESULTS = REPO / "results"
FIGURES = {
    "fig1_gate.pdf", "fig2_reception.pdf",
    "fig3_surrender.pdf", "fig4_confirmatory.pdf",
}


def rows(name):
    return list(csv.DictReader((RESULTS / f"{name}.csv").open()))


def rounded_equal(a, b, digits):
    return round(float(a), digits) == round(float(b), digits)


def verify_plot_data():
    problems = []
    discovery = rows("plot_discovery")
    reception = rows("reception")
    directions = rows("directions")[:9]
    for full, rec, direction in zip(discovery, reception, directions):
        checks = [("gamma", "Gamma", 1), ("gain", "gain", 1),
                  ("adv_gain", "adv_gain", 2), ("rho", "rho", 1),
                  ("unaided", "unaided", 1), ("uptake", "uptake", 1)]
        for full_key, shown_key, digits in checks:
            target = rec if shown_key in rec else direction
            if not rounded_equal(full[full_key], target[shown_key], digits):
                problems.append(f"plot_discovery {full['key']} {full_key} disagrees with presentation CSV")

    specs = {r["predictor"]: r for r in rows("specifications")}
    for full in rows("plot_specifications"):
        shown = specs[full["predictor"]]
        for key, digits in (("slope", 3), ("intercept", 6), ("r_squared", 4)):
            if not rounded_equal(full[key], shown[key], digits):
                problems.append(f"plot specification {full['predictor']} {key} mismatch")

    for full, shown in zip(rows("plot_ceiling"), rows("ceiling")):
        for key in ("peer_in_panel", "panel_two", "oracle"):
            if not rounded_equal(full[key], shown[key], 1):
                problems.append(f"plot ceiling {full['setting']} {key} mismatch")

    confirm = {r["quantity"]: r for r in rows("confirmatory")}
    mapping = {"gamma": "Gamma (pp)", "r1_gain": "R1 gain (pp)",
               "r4_gain": "R4 gain (pp)", "rho": "rho (%)",
               "rho_lo": "rho CI lo", "rho_hi": "rho CI hi",
               "upsilon": "upsilon (%)", "upsilon_lo": "upsilon CI lo",
               "upsilon_hi": "upsilon CI hi"}
    for full in rows("plot_confirmatory"):
        peer = full["peer"]
        for key, quantity in mapping.items():
            if full[key] == "":
                continue
            if not rounded_equal(full[key], confirm[quantity][peer], 2):
                problems.append(f"plot confirmatory {peer} {key} mismatch")
    return problems


def verify_new_accounting():
    problems = []
    directions = {r["direction"]: r for r in rows("directions")}
    reception = {r["direction"]: r for r in rows("reception")}
    for row in rows("discordant_flows"):
        if not rounded_equal(float(row["S_mass"]) - float(row["T_mass"]),
                             row["mixed_gap"], 1):
            problems.append(f"discordant mixed-gap identity fails: {row['direction']}")
        if not rounded_equal(float(row["S_rho"]) - float(row["T_eta"]),
                             row["discordant_net"], 1):
            problems.append(f"discordant net-flow identity fails: {row['direction']}")
        if row["cohort"] == "discovery":
            direction = directions[row["direction"]]
            rec = reception[row["direction"]]
            for left, right, digits in ((row["Gamma"], direction["Gamma"], 1),
                                        (row["rho"], rec["rho"], 1),
                                        (row["controlled_gain"], rec["gain"], 1)):
                if not rounded_equal(left, right, digits):
                    problems.append(f"discordant presentation mismatch: {row['direction']}")

    round_rows = {(r["setting"], r["series"]): r for r in rows("rounds")}
    for row in rows("composition_accounting"):
        setting = row["setting"]
        if not rounded_equal(float(row["panel_only"]) - float(row["peer_only"]),
                             row["net_vs_peer"], 1):
            problems.append(f"composition win/loss identity fails: {setting}")
        if not rounded_equal(float(row["panel_accuracy"]) - float(row["peer_accuracy"]),
                             row["net_vs_peer"], 1):
            problems.append(f"composition accuracy identity fails: {setting}")
        if abs(float(row["panel_accuracy"])
               - float(round_rows[(setting, "panel_two")]["R4"])) > 0.051:
            problems.append(f"composition panel R4 mismatch: {setting}")
        if abs(float(row["peer_accuracy"])
               - float(round_rows[(setting, "peer_in_panel")]["R4"])) > 0.051:
            problems.append(f"composition peer R4 mismatch: {setting}")

    main_product = next(r for r in rows("specifications")
                        if r["predictor"] == "Gamma+ x rho")
    flow_product = next(r for r in rows("flow_specifications")
                        if r["predictor"] == "Gamma+ x rho")
    for key, digits in (("slope", 3), ("intercept", 4), ("r_squared", 4)):
        if not rounded_equal(main_product[key], flow_product[key], digits):
            problems.append(f"flow specification disagrees on {key}")
    return problems


def verify_design():
    expected = {
        "discovery_directions": 9, "confirmatory_directions": 2,
        "total_directions": 11, "primary_agents": 3, "primary_rounds": 5,
        "math_hard_items": 262, "gsm8k_items": 1319,
        "small_model_diagnostic_items": 100, "musique_items": 300,
        "musique_heterogeneous_panels": 6, "musique_matched_items": 300,
        "musique_agent_round_records": 27000,
        "musique_agent_round_records_per_panel": 4500,
        "musique_replicates_per_condition": 1,
        "discovery_matched_replicate_items": 12717,
        "discovery_defender_condition_observations": 76302,
        "confirmatory_matched_replicate_items": 1572,
        "confirmatory_defender_condition_observations": 9432,
        "per_protocol_matched_replicate_items": 10580,
        "bootstrap_seed": 260719826, "bootstrap_replicates": 4000,
        "math_completion_token_budget": 4096,
        "gsm8k_completion_token_budget": 1024,
        "primary_temperature": 0, "primary_replicates": 3,
    }
    have = {r["quantity"]: int(r["value"]) for r in rows("design_quantities")}
    return [f"design {key}: generated {have.get(key)} vs expected {value}"
            for key, value in expected.items() if have.get(key) != value]


def verify_result_files():
    problems = []
    expected = {path.name: path for path in REFERENCE_RESULTS.glob("*.csv")}
    generated = {path.name: path for path in RESULTS.glob("*.csv")}
    for name in sorted(set(expected) | set(generated)):
        if name not in expected:
            problems.append(f"unexpected generated result: {name}")
        elif name not in generated:
            problems.append(f"missing generated result: {name}")
        elif expected[name].read_bytes() != generated[name].read_bytes():
            problems.append(f"generated result differs: {name}")
    return problems


def verify_figures(directory):
    problems = []
    generated = {path.name: path for path in directory.glob("*.pdf")}
    for name in sorted(FIGURES | set(generated)):
        if name not in FIGURES:
            problems.append(f"unexpected generated figure: {name}")
        elif name not in generated:
            problems.append(f"missing generated figure: {name}")
        elif not generated[name].read_bytes().startswith(b"%PDF-"):
            problems.append(f"invalid generated PDF: {name}")
    return problems


def verify_camera_ready():
    """Check the added control and token accounting on their own populations."""
    problems = []
    ladder = rows("same_family_ladder")
    if len(ladder) != 3 or ladder[0]["direction"] != "Qwen-7B <- Qwen-72B":
        problems.append("same-family ladder population mismatch")
    else:
        same = ladder[0]
        if int(same["n_items"]) != 786 or int(same["rho_n"]) != 180:
            problems.append("same-family support/denominator mismatch")
        if same["uptake"] or same["adv_gain"]:
            problems.append("same-family control must not imply an adversarial arm")
    directions = {r["direction"]: float(r["Gamma"]) for r in rows("directions")}
    directions[ladder[0]["direction"]] = float(ladder[0]["Gamma"])
    accounting = rows("token_cost")
    if len(accounting) != 12:
        problems.append("token comparison must contain 12 directions")
    positive = [r for r in accounting if directions[r["direction"]] > 0]
    if len(positive) != 9:
        problems.append("token positive-gap population must contain 9 directions")
    for row in accounting:
        if int(row["debate_calls_per_item"]) != 15 or int(row["direct_calls_per_item"]) != 1:
            problems.append("incorrect token call accounting: " + row["direction"])
    for row in rows("release_quantities"):
        if row["agrees"] != "YES":
            problems.append("camera-ready claim mismatch: " + row["quantity"])
    return problems


def main():
    if len(sys.argv) != 2:
        print("usage: verify_artifact.py <generated_figures>")
        return 2
    figures = Path(sys.argv[1]).resolve()
    problems = (verify_result_files() + verify_plot_data()
                + verify_new_accounting() + verify_design()
                + verify_figures(figures) + verify_camera_ready())
    if problems:
        print(f"FAIL: {len(problems)} strict artifact mismatch(es)")
        for problem in problems:
            print("  ", problem)
        return 1
    n_results = len(list(REFERENCE_RESULTS.glob("*.csv")))
    print(f"PASS: {n_results} result CSVs match exactly; plot data and design "
          f"quantities are consistent; {len(FIGURES)} figures were generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
