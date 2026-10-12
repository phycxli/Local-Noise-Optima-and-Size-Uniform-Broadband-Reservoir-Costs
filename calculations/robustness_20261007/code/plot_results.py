"""White-background scientific plots from accepted research outputs."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from physics import ROOT


def records(filename):
    with (ROOT / "results" / filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    summary = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
    adaptive, trace = records("physical_adaptive_optimality_scan.csv"), records("matched_trace_budget.csv")
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "mathtext.fontset": "stix",
                         "axes.linewidth": .8, "axes.facecolor": "white", "figure.facecolor": "white",
                         "xtick.direction": "in", "ytick.direction": "in", "savefig.facecolor": "white",
                         "axes.labelsize": 10, "legend.fontsize": 8, "pdf.fonttype": 42})
    colors = {"uniform": "#3068ab", "dimerized": "#b44b59"}
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 5.9))
    rng = np.random.default_rng(20261007)
    ax = axs[0, 0]
    for name in colors:
        group = [r for r in adaptive if r["model"] == name]
        x = np.array([1000 * float(r["halfwidth"]) for r in group])
        y = np.array([float(r["optimality_gap_lower_float"]) for r in group])
        ax.scatter(x + rng.uniform(-.45, .45, len(x)), y, s=10, alpha=.25, linewidths=0, color=colors[name])
        minima = [min(float(r["optimality_gap_lower_float"]) for r in group if float(r["halfwidth"]) == width)
                  for width in (.045, .05, .055)]
        ax.plot([45, 50, 55], minima, "o-", ms=4, lw=1.3, color=colors[name], label=name.capitalize())
    ax.set(xlabel="Band halfwidth (kHz)", ylabel=r"$\overline{\mathcal{N}}^{*}_{r=1}-\overline{\mathcal{N}}_{r=2}$", xticks=[45, 50, 55], ylim=(0, None))
    ax.legend(frameon=False, loc="upper right")
    ax = axs[0, 1]
    for name in colors:
        group = [r for r in trace if r["model"] == name and float(r["linewidth"]) == 80.]
        ax.plot([float(r["trace_cap"]) for r in group], [float(r["gap_lower_float"]) for r in group],
                "s-", ms=4, lw=1.3, color=colors[name], label=name.capitalize())
    ax.set(xlabel=r"Complete trace cap $\tau$ (MHz)", ylabel="Noise gap lower bound (quanta)")
    ax.legend(frameon=False)
    ax = axs[1, 0]
    box = summary["continuous_box_certificate"]
    errors = np.linspace(0, .004, 100)
    ax.plot(1000 * errors, box["gap_lower"] - .1 - box["profile_energy_upper"] * errors,
            color="#21866e", lw=1.7, label=r"Continuous physical box, $N=64$")
    for r in summary["markov_all_length_error_certificates"]:
        ax.plot(1000 * errors, r["optimality_gap_lower"] - .1 - r["all_length_profile_energy_upper"] * errors,
                color=colors[r["model"]], ls="--", lw=1.2, label=r["model"].capitalize() + r", Markov $N\geq64$")
    ax.axhline(0, color="black", lw=.7)
    ax.plot([1], [box["remaining_measured_gap_lower"]], "o", color="#21866e", ms=4)
    ax.set(xlabel="Combined force-noise error (kHz)", ylabel="Resolvable gap lower bound (quanta)", ylim=(-.6, .8))
    ax.legend(frameon=False, loc="lower left", fontsize=7.3)
    ax = axs[1, 1]
    resource = summary["baseline_hardware_resources"]
    x = np.arange(2)
    ax.bar(x - .16, [r["complete_loss_max_eigenvalue_MHz"] for r in resource], width=.28, color="#3068ab", label="Complete loss rate")
    ax.bar(x + .16, [r["maximum_coupling_per_addressed_mode_MHz"] for r in resource], width=.28, color="#b44b59", label="Largest addressed coupling")
    ax.axhline(8, lw=.8, ls="--", color="black")
    ax.set(xticks=x, xticklabels=[r"$r=1$", r"$r=2$"], ylabel=r"Angular rate divided by $2\pi$ (MHz)", ylim=(0, 12.5))
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    for ax, label in zip(axs.flat, "abcd"):
        ax.tick_params(top=True, right=True, length=3.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", top=True, right=True, length=2)
        ax.text(.5, -.28, f"({label})", transform=ax.transAxes, ha="center", va="top", fontsize=10)
    fig.subplots_adjust(left=.095, right=.98, top=.98, bottom=.16, hspace=.62, wspace=.33)
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "robustness_and_resources.pdf")
    fig.savefig(directory / "robustness_and_resources.png", dpi=220)
    print(str(directory / "robustness_and_resources.png"), flush=True)


if __name__ == "__main__":
    main()
