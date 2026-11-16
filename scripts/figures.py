#!/usr/bin/env python3
"""Emit the manuscript figures from results/*.csv.

Every value plotted comes from the CSVs produced by the table scripts, so the
figures and the tables cannot disagree.

Usage:  python3 figures.py <output_dir>
"""
import csv
import os
import sys
from pathlib import Path

# Stabilize PDF metadata so pinned Matplotlib/font versions produce identical
# bytes across repeated reproduction runs.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from artifact_paths import RESULTS        # noqa: E402
INK, BLUE, RED, GOLD, GREY = "#1b1b1b", "#2b6cb0", "#c0392b", "#b8860b", "#8a8a8a"
plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.linewidth": .7,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": .02,
})

SHORT = {
    "Gemma-3-12B <- Qwen-7B": r"Gemma-3-12B $\leftarrow$ Qwen-7B",
    "Qwen-7B <- Llama-8B": r"Qwen-7B $\leftarrow$ Llama-8B",
    "Gemini-Flash <- GPT-4.1": r"Gemini-Flash $\leftarrow$ GPT-4.1",
    "GPT-4.1 <- Gemini-Flash": r"GPT-4.1 $\leftarrow$ Gemini-Flash",
    "GPT-4.1 <- OSS-120B": r"GPT-4.1 $\leftarrow$ OSS-120B",
    "Llama-8B <- Qwen-7B": r"Llama-8B $\leftarrow$ Qwen-7B",
    "Qwen-72B <- OSS-120B": r"Qwen-72B $\leftarrow$ OSS-120B",
    "Llama-1B <- Gemma-2B": r"Llama-1B $\leftarrow$ Gemma-2B",
    "Llama-70B <- OSS-120B": r"Llama-70B $\leftarrow$ OSS-120B",
}


def read(name):
    return list(csv.DictReader((RESULTS / name).open()))


def fig_gate(out):
    rec = read("plot_discovery.csv")
    gap = np.array([float(r["gamma"]) for r in rec])
    gain = np.array([float(r["gain"]) for r in rec])
    adv_cor = np.array([float(r["adv_excess_corruption"]) for r in rec])
    gated = gap > 0

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax, y, ylabel, colour, title in (
        (axes[0], gain, "Honest net accuracy change (pp)", BLUE,
         "(a) Benefit is gated"),
        (axes[1], adv_cor, "Adversarial excess corruption (pp)", RED,
         "(b) Exposure is gated"),
    ):
        lo, hi = min(-14, y.min() - 4), y.max() + 5
        ax.axvspan(gap.min() - 4, 0, color=GREY, alpha=.14, lw=0)
        ax.axhline(0, color=INK, lw=.6, zorder=1)
        ax.axvline(0, color=INK, lw=.6, ls=(0, (4, 2)), zorder=1)
        ax.scatter(gap[~gated], y[~gated], s=34, facecolor="white",
                   edgecolor=colour, lw=1.1, zorder=3)
        ax.scatter(gap[gated], y[gated], s=34, color=colour, zorder=3)
        slope, intercept = np.polyfit(np.maximum(gap, 0), y, 1)
        span = np.linspace(0, gap.max() + 2, 50)
        ax.plot(span, intercept + slope * span, color=colour, lw=1.0,
                alpha=.55, zorder=2)
        ax.plot([gap.min() - 4, 0], [intercept, intercept], color=colour,
                lw=1.0, alpha=.55, zorder=2)
        ax.set_xlim(gap.min() - 4, gap.max() + 4)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"Capability gap $\Gamma$ (pp)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(gap.min() - 2.5, min(-14, gain.min() - 4) + 1.6,
                 "ungated:\nnear-zero net", fontsize=7.2, color=GREY,
                 va="bottom")
    k = [r["direction"] for r in rec].index("Llama-8B <- Qwen-7B")
    axes[1].annotate(
        r"Llama-8B $\leftarrow$ Qwen-7B", xy=(gap[k], adv_cor[k]),
        xytext=(16, -4), textcoords="offset points", fontsize=6.8, color=GREY,
        arrowprops=dict(arrowstyle="-", lw=.5, color=GREY, shrinkA=0,
                        shrinkB=3))
    fig.savefig(out / "fig1_gate.pdf")
    plt.close(fig)


def fig_reception(out):
    rec = read("plot_discovery.csv")
    spec = {r["predictor"]: r for r in read("plot_specifications.csv")}
    gap = np.array([float(r["gamma"]) for r in rec])
    rho = np.array([float(r["rho"]) for r in rec])
    unaided = np.array([float(r["unaided"]) for r in rec])
    gain = np.array([float(r["gain"]) for r in rec])

    n = len(gap)
    mx, my = gap.mean(), rho.mean()
    r_gap_rho = (((gap - mx) * (rho - my)).sum()
                 / np.sqrt(((gap - mx) ** 2).sum() * ((rho - my) ** 2).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    ax.axvline(0, color=INK, lw=.6, ls=(0, (4, 2)))
    for x, hi, lo in zip(gap, rho, unaided):
        ax.plot([x, x], [lo, hi], color=GREY, lw=.7, zorder=1)
    ax.scatter(gap, rho, s=34, color=GOLD, zorder=3, label=r"reception $\rho$")
    ax.scatter(gap, unaided, s=16, color=GREY, marker="_", zorder=3,
               label="unaided control")
    ax.set_xlabel(r"Capability gap $\Gamma$ (pp)")
    ax.set_ylabel("Rate (\\%)")
    ax.set_ylim(-3, 100)
    ax.set_xlim(gap.min() - 4, gap.max() + 4)
    ax.set_title("(a) Reception varies across directions", loc="left",
                 fontsize=8.5)
    ax.text(gap.max() - 1, 4.0, f"$r={r_gap_rho:.2f}$", fontsize=7.4,
            ha="right", color=GOLD)
    ax.legend(loc="upper left", fontsize=7.0, handletextpad=.4,
              borderaxespad=.2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    hinge, product = spec["Gamma+"], spec["Gamma+ x rho"]
    x_hinge = np.maximum(gap, 0)
    pred_h = float(hinge["intercept"]) + float(hinge["slope"]) * x_hinge
    pred_p = (float(product["intercept"])
              + float(product["slope"]) * (x_hinge * rho / 100.0))
    limits = [-2, max(gain.max(), pred_h.max(), pred_p.max()) + 3]
    ax.plot(limits, limits, color=INK, lw=.6, ls=(0, (4, 2)), zorder=1)
    ax.scatter(pred_h, gain, s=30, facecolor="white", edgecolor=GREY, lw=1.0,
               zorder=3,
               label=r"$\Gamma^{+}$ only ($R^2=%.3f$)" % float(hinge["r_squared"]))
    ax.scatter(pred_p, gain, s=30, color=GOLD, zorder=4,
               label=r"$\Gamma^{+}\!\times\!\rho$ ($R^2=%.3f$)"
                     % float(product["r_squared"]))
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Predicted net accuracy change (pp)")
    ax.set_ylabel("Observed (pp)")
    ax.set_title("(b) Opportunity times uptake", loc="left", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7.0, handletextpad=.4,
              borderaxespad=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "fig2_reception.pdf")
    plt.close(fig)


def fig_surrender(out):
    rec = read("plot_discovery.csv")
    ceil = read("plot_ceiling.csv")
    order = [r["direction"] for r in rec]
    gain = np.array([float(r["gain"]) for r in rec])
    adv_gain = np.array([float(r["adv_gain"]) for r in rec])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    idx = np.arange(len(order))
    w = .38
    ax.axhline(0, color=INK, lw=.6)
    ax.bar(idx - w / 2, gain, w, color=BLUE, label="honest peer")
    ax.bar(idx + w / 2, adv_gain, w, color=RED,
           label="same peer, adversarial role")
    ax.set_xticks(idx)
    ax.set_xticklabels([SHORT.get(c, c) for c in order], rotation=48,
                       ha="right", rotation_mode="anchor", fontsize=6.4)
    ax.set_ylabel("Net accuracy change vs.\nhomogeneous debate (pp)")
    ax.set_title("(a) The gain is surrenderable", loc="left", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7.0, handletextpad=.4,
              borderaxespad=.2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    names = [r"OSS $\to$ L-70B", r"OSS $\to$ GPT-4.1", r"Gem $\to$ GPT-4.1",
             r"GPT-4.1 $\to$ Gem", r"G-2B $\to$ L-1B"]
    peer = [float(r["peer_in_panel"]) for r in ceil]
    two = [float(r["panel_two"]) for r in ceil]
    orc = [float(r["oracle"]) for r in ceil]
    pos = np.arange(len(ceil))
    w = .26
    ax.bar(pos - w, peer, w, color=GOLD, label="influential peer")
    ax.bar(pos, two, w, color=BLUE, label=r"panel ($\geq$2 correct)")
    ax.bar(pos + w, orc, w, color=GREY, label="oracle (any correct)")
    ax.set_xticks(pos)
    ax.set_xticklabels(names, rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=6.6)
    ax.set_ylabel("Best-round accuracy (\\%)")
    ax.set_ylim(0, 108)
    ax.set_title("(b) The panel approaches the peer", loc="left",
                 fontsize=8.5)
    ax.legend(loc="upper left", fontsize=6.8, handletextpad=.4,
              borderaxespad=.2, ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "fig3_surrender.pdf")
    plt.close(fig)


def fig_confirmatory(out):
    rows = read("plot_confirmatory.csv")
    gaps = np.array([float(r["gamma"]) for r in rows])
    r1 = np.array([float(r["r1_gain"]) for r in rows])
    r4 = np.array([float(r["r4_gain"]) for r in rows])
    peers = ["Gemma-3-12B", "OSS-120B"]
    rho = np.array([float(r["rho"]) for r in rows[1:]])
    rho_lo = np.array([float(r["rho_lo"]) for r in rows[1:]])
    rho_hi = np.array([float(r["rho_hi"]) for r in rows[1:]])
    ups = np.array([float(r["upsilon"]) for r in rows[1:]])
    ups_lo = np.array([float(r["upsilon_lo"]) for r in rows[1:]])
    ups_hi = np.array([float(r["upsilon_hi"]) for r in rows[1:]])

    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.25),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    ax.plot(gaps, r1, "o-", color=BLUE, lw=1.1, ms=4, label="R1")
    ax.plot(gaps, r4, "s--", color=GOLD, lw=1.1, ms=4, label="R4")
    ax.set_xlabel(r"Peer gap $\Gamma_d$ (pp)")
    ax.set_ylabel("Controlled gain (pp)")
    ax.set_title("(a) Benefit scales with supply", fontsize=8.3)
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    for ax, v, lo, hi, colour, ylab, title, top in (
        (axes[1], rho, rho_lo, rho_hi, BLUE, r"Reception $\rho_d$ (\%)",
         "(b) Receiver stays stable", 90),
        (axes[2], ups, ups_lo, ups_hi, RED,
         r"Adversarial uptake $\upsilon_d$ (\%)",
         "(c) Exposure also rises", 10),
    ):
        x = np.arange(2)
        ax.bar(x, v, .55, color=colour)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK,
                    capsize=2, lw=.8)
        ax.set_xticks(x)
        ax.set_xticklabels(peers, rotation=22, fontsize=7, ha="right")
        ax.set_ylim(0, top)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "fig4_confirmatory.pdf")
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_gate, fig_reception, fig_surrender, fig_confirmatory):
        fn(out)
        print(f"wrote {fn.__name__}")


if __name__ == "__main__":
    main()
