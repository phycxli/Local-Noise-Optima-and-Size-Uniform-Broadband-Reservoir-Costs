"""Redraw the supplementary figures from saved results, without optimization."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np

from plot_current_results import result_figure_audit, result_style
from figure_palette import BLUE, INK, MODEL_COLORS


ROOT = Path(__file__).resolve().parents[1]
CALC = ROOT / "calculations/robustness_20261007"
PROVENANCE = ROOT / "results/reports/current_supplement_figures_audit.json"
S1_OLD = "figures/finite_frequency_local_optimality.pdf"
S1_NEW = "figures/figS1_finite_frequency_validation.pdf"
S2 = "figures/figS2_robustness_and_resources.pdf"
COLORS = {**MODEL_COLORS, "box": BLUE}
RESOURCE_COLORS = {"rate": "#535f68", "coupling": "#c4cbd0"}
INPUTS = (
    "results/finite_frequency_local_optimality_scalar.csv",
    "results/finite_frequency_local_optimality_multiband.csv",
    "calculations/robustness_20261007/results/physical_adaptive_optimality_scan.csv",
    "calculations/robustness_20261007/results/matched_trace_budget.csv",
    "calculations/robustness_20261007/results/summary.json",
    "scripts/run_finite_frequency_local_optimality.py",
    "calculations/robustness_20261007/code/plot_results.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def data_hash(x, y):
    return hashlib.sha256(np.column_stack((x, y)).astype("<f8").tobytes()).hexdigest()


def saved_series():
    scalar, multiband, adaptive, trace = [records(name) for name in INPUTS[:4]]
    summary = json.loads((CALC / "results/summary.json").read_text(encoding="utf-8"))
    assert len(trace) == summary["matched_trace_budget_audit"]["rows"] == 20
    assert {(row["model"], float(row["linewidth"]), float(row["trace_cap"]))
            for row in trace} == {
                (model, linewidth, cap)
                for model in ("uniform", "dimerized")
                for linewidth in (80., 100.)
                for cap in (175., 190., 210., 250., 325.)}
    series = {}
    for omega in sorted({float(row["omega"]) for row in scalar}):
        group = sorted((row for row in scalar if float(row["omega"]) == omega),
                       key=lambda row: float(row["N"]))
        series[f"S1_a_omega_{omega:g}"] = (
            np.array([float(row["N"]) for row in group]),
            np.array([float(row["sdp_R1"]) for row in group]))
    indices = np.arange(1, len(scalar) + 1)
    series["S1_b_scalar_sdp"] = (indices, np.array([
        abs(float(row["sdp_R1"]) - float(row["sdp_Rinf"])) for row in scalar]))
    indices = np.arange(1, len(multiband) + 1)
    series["S1_c_two_band_sdp"] = (indices, np.array([
        abs(float(row["sdp_R1"]) - float(row["sdp_Rinf"])) for row in multiband]))
    series["S1_c_explicit_kernel"] = (indices, np.array([
        abs(float(row["explicit_error"])) for row in multiband]))
    rng = np.random.default_rng(20261007)
    for model in ("uniform", "dimerized"):
        group = [row for row in adaptive if row["model"] == model]
        widths = np.array([1000 * float(row["halfwidth"]) for row in group])
        gaps = np.array([float(row["optimality_gap_lower_float"]) for row in group])
        series[f"S2_a_{model}_cases"] = (widths + rng.uniform(-.45, .45, len(widths)), gaps)
        series[f"S2_a_{model}_minima"] = (np.array([45, 50, 55]), np.array([
            min(float(row["optimality_gap_lower_float"]) for row in group
                if float(row["halfwidth"]) == width) for width in (.045, .05, .055)]))
        for linewidth in (80., 100.):
            group = sorted((row for row in trace if row["model"] == model
                            and float(row["linewidth"]) == linewidth),
                           key=lambda row: float(row["trace_cap"]))
            series[f"S2_b_{model}_kappa_{linewidth:g}"] = (
                np.array([float(row["trace_cap"]) for row in group]),
                np.array([float(row["gap_lower_float"]) for row in group]))
    errors = np.linspace(0, .004, 100)
    box = summary["continuous_box_certificate"]
    assert box["configuration"]["model"] == "uniform"
    assert box["configuration"]["N"] == 64
    for key, region in (("box", box), ("high_gain", box["high_gain_subregion"])):
        residual = (region["gap_lower"] - 2 * box["measurement_error_each"]
                    - region["profile_energy_upper"] * box["force_noise_norm_error_sum"])
        assert residual == region["remaining_measured_gap_lower"]
        series[f"S2_c_{key}"] = (
            1000 * errors, region["gap_lower"] - 2 * box["measurement_error_each"]
            - region["profile_energy_upper"] * errors)
        series[f"S2_c_{key}_example"] = (
            np.array([1000 * box["force_noise_norm_error_sum"]]),
            np.array([region["remaining_measured_gap_lower"]]))
    for row in summary["markov_all_length_error_certificates"]:
        assert row["N_minimum"] == 64
        series[f"S2_c_{row['model']}"] = (
            1000 * errors, row["optimality_gap_lower"] - 2 * row["measurement_error_each"]
            - row["all_length_profile_energy_upper"] * errors)
    resources = summary["baseline_hardware_resources"]
    assert [row["radius"] for row in resources] == [1, 2]
    series["S2_d_loss_rate"] = (np.array([0., 1.]), np.array([
        row["complete_loss_max_eigenvalue_MHz"] for row in resources]))
    series["S2_d_coupling"] = (np.array([0., 1.]), np.array([
        row["maximum_coupling_per_addressed_mode_MHz"] for row in resources]))
    return series


def finish(fig, filename, expected, shown):
    audit = result_figure_audit(fig)
    assert set(expected) == set(shown)
    for key, values in expected.items():
        assert np.array_equal(np.column_stack(values), shown[key]), key
    audit["displayed_scientific_series_sha256"] = {key: data_hash(*values) for key, values in expected.items()}
    audit["all_plotted_values_match_saved_results"] = True
    pdf = ROOT / filename
    fig.savefig(pdf, metadata={"Title": pdf.stem, "CreationDate": None, "ModDate": None})
    fig.savefig(pdf.with_suffix(".png"), dpi=600)
    audit["outputs_sha256"] = {path.relative_to(ROOT).as_posix(): sha(path)
                                for path in (pdf, pdf.with_suffix(".png"))}
    plt.close(fig)
    return audit


def draw_line(ax, series, shown, key, **kwargs):
    line, = ax.plot(*series[key], gid=key, **kwargs)
    shown[key] = np.column_stack((line.get_xdata(), line.get_ydata()))
    return line


def panel_labels(axes, offset):
    for ax, label in zip(axes, "abcd"):
        result_style(ax)
        ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.5, width=.5)
        ax.text(.5, offset, f"({label})", transform=ax.transAxes, ha="center", va="top", fontsize=10.5)


def figure_s1(series):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    fig.subplots_adjust(left=.068, right=.988, top=.955, bottom=.305, wspace=.50)
    shown = {}
    ax = axes[0]
    negative, positive = "S1_a_omega_-0.03", "S1_a_omega_0.03"
    assert all(np.array_equal(a, b) for a, b in zip(series[negative], series[positive]))
    draw_line(ax, series, shown, negative, color=COLORS["uniform"], lw=1.2,
              marker="o", ms=4.5, mfc="white", mew=1.0, label=r"$\omega=\pm0.03$")
    # Both opposite-detuning records occupy the same curve, with no display offset.
    shown[positive] = shown[negative].copy()
    draw_line(ax, series, shown, "S1_a_omega_0", color=COLORS["dimerized"], lw=1.2,
              marker="D", ms=3.9, label=r"$\omega=0$")
    ax.set(xlabel=r"Chain length $N$", ylabel=r"$\mathcal{N}^{\mathrm{exc}}_{r=1}$ (quanta)",
           xticks=[27, 30, 33], yticks=[.4, .5, .6, .7, .8], ylim=(.32, .84), xlim=(26.5, 33.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, loc="upper left", handlelength=1.7, borderpad=.1, labelspacing=.25)
    ax = axes[1]
    draw_line(ax, series, shown, "S1_b_scalar_sdp", color=COLORS["uniform"],
              ls="none", marker="o", ms=4.1)
    ax.set(yscale="log", xlabel="Sample index", ylabel=r"$|\mathcal{N}^{\mathrm{exc}}_{r=1}-\mathcal{N}^{\mathrm{exc}}_{\mathrm{all}}|$ (quanta)",
           ylim=(1.4e-5, 4.5e-4), yticks=[2e-5, 1e-4, 3e-4])
    ax.set_yticklabels([r"$2\times10^{-5}$", r"$10^{-4}$", r"$3\times10^{-4}$"])
    ax = axes[2]
    draw_line(ax, series, shown, "S1_c_two_band_sdp", color=COLORS["uniform"], lw=1.0,
              marker="o", ms=3.6, label="SDP difference")
    draw_line(ax, series, shown, "S1_c_explicit_kernel", color=COLORS["dimerized"], lw=1.0,
              ls="--", marker="s", mfc="white", mew=.9, ms=3.6, label="Kernel error")
    ax.set(yscale="log", xlabel="Sample index", ylabel="Absolute error (quanta)", ylim=(1e-13, 2e-4))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, loc="center", handlelength=1.7, borderpad=.1, labelspacing=.3)
    for ax in axes[1:]:
        ax.set(xticks=[1, 3, 5, 7, 9], xlim=(.5, 9.5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    panel_labels(axes, -.34)
    return finish(fig, S1_NEW, {k: v for k, v in series.items() if k.startswith("S1_")}, shown)


def figure_s2(series):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.9))
    fig.subplots_adjust(left=.096, right=.983, top=.975, bottom=.135, hspace=.57, wspace=.35)
    shown = {}
    ax = axes[0, 0]
    for model, marker in (("uniform", "o"), ("dimerized", "D")):
        key = f"S2_a_{model}_cases"
        points = ax.scatter(*series[key], s=8, alpha=.20, linewidths=0, color=COLORS[model], gid=key)
        shown[key] = np.asarray(points.get_offsets())
        draw_line(ax, series, shown, f"S2_a_{model}_minima", marker=marker,
                  color=COLORS[model], ms=4.5, lw=1.4, label=model.capitalize())
    ax.set(xlabel="Band halfwidth (kHz)", ylabel="Noise-gap lower bound (quanta)",
           xticks=[45, 50, 55], xlim=(44, 56), ylim=(0, 1.16), yticks=[0, .2, .4, .6, .8, 1.0])
    ax.legend(frameon=False, loc="lower left", handlelength=1.9, borderpad=.2, labelspacing=.3)
    ax = axes[0, 1]
    for model, marker in (("uniform", "o"), ("dimerized", "D")):
        # Nested markers show almost coincident linewidth results without offsets.
        for linewidth in (100, 80):
            draw_line(ax, series, shown, f"S2_b_{model}_kappa_{linewidth}",
                      marker=marker, color=COLORS[model],
                      ms=5.8 if linewidth == 100 else 3.2,
                      mfc="white" if linewidth == 100 else COLORS[model],
                      mew=.9, lw=1.3 if linewidth == 100 else .85,
                      ls="--" if linewidth == 100 else "-",
                      label=f"{model.capitalize()}, {linewidth} MHz")
    ax.set(xlabel=r"Complete trace cap $J_0\tau/(2\pi)$ (MHz)", ylabel="Noise-gap lower bound (quanta)",
           xticks=[175, 190, 210, 250, 325], xlim=(165, 335), ylim=(0, 1.12), yticks=[0, .2, .4, .6, .8, 1.0])
    ax.legend(frameon=False, loc="lower right", handlelength=2.0,
              borderpad=.2, labelspacing=.3, fontsize=9.)
    ax = axes[1, 0]
    draw_line(ax, series, shown, "S2_c_box", color=COLORS["box"], lw=1.5,
              ls=(0, (1.2, 1.5)), label=r"$N=64$, broad region")
    draw_line(ax, series, shown, "S2_c_high_gain", color=COLORS["box"], lw=1.7,
              label=r"$N=64$, high-gain region")
    for model in ("uniform", "dimerized"):
        draw_line(ax, series, shown, f"S2_c_{model}", color=COLORS[model], lw=1.3, ls="--",
                  label=model.capitalize() + r", $N\geq64$")
    for key, marker in (("box", "o"), ("high_gain", "s")):
        draw_line(ax, series, shown, f"S2_c_{key}_example", color=COLORS["box"],
                  marker=marker, ls="none", ms=4.7, mfc="white", mew=1.)
    ax.axhline(0, color=INK, lw=.7)
    ax.set(xlabel="Combined force-noise error (kHz)", ylabel="Resolvable gap bound (quanta)",
           xticks=[0, 1, 2, 3, 4], xlim=(-.08, 4.08), ylim=(-.60, .8),
           yticks=[-.6, -.4, -.2, 0, .2, .4, .6, .8])
    ax.legend(frameon=False, loc="lower left", handlelength=2.0,
              borderpad=.2, labelspacing=.35, fontsize=9.)
    ax = axes[1, 1]
    handles = []
    for key, offset, color, hatch, label in (
        ("S2_d_loss_rate", -.17, RESOURCE_COLORS["rate"], "", "Complete loss rate"),
        ("S2_d_coupling", .17, RESOURCE_COLORS["coupling"], "///", "Largest addressed coupling")):
        x, y = series[key]
        bars = ax.bar(x + offset, y, width=.27, color=color, edgecolor=INK,
                      linewidth=.6, hatch=hatch, label=label)
        assert np.allclose([bar.get_x() + bar.get_width() / 2 for bar in bars], x + offset, atol=1e-15)
        shown[key] = np.column_stack((x, [bar.get_height() for bar in bars]))
        handles.append(bars)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .16,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    for x in series["S2_d_loss_rate"][0] - .17:
        ax.plot([x - .18, x + .18], [8, 8], color=INK, ls="--", lw=.9)
    handles.append(Line2D([0], [0], color=INK, ls="--", lw=.9, label="Complete-rate cap"))
    ax.legend(handles=handles, frameon=False, loc="upper left", handlelength=1.6, borderpad=.2, labelspacing=.3)
    ax.set(xticks=[0, 1], xticklabels=[r"$r=1$", r"$r=2$"], xlim=(-.55, 1.55),
           ylabel=r"Rate or coupling / $2\pi$ (MHz)", ylim=(0, 14.6), yticks=[0, 2, 4, 6, 8, 10, 12, 14])
    for ax in axes.flat:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        if ax is not axes[1, 1]:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    panel_labels(axes.flat, -.28)
    return finish(fig, S2, {k: v for k, v in series.items() if k.startswith("S2_")}, shown)


def main():
    input_hashes = {name: sha(ROOT / name) for name in INPUTS}
    plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "mathtext.fontset": "stix",
                         "axes.labelsize": 10, "axes.labelpad": 3.5, "legend.fontsize": 9.5,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.facecolor": "white",
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    series = saved_series()
    summary = json.loads((CALC / "results/summary.json").read_text(encoding="utf-8"))
    box = summary["continuous_box_certificate"]
    tolerance_parameters = {}
    for key, region in (("broad_region", box), ("high_gain_region", box["high_gain_subregion"])):
        tolerance_parameters[key] = {
            "gap_lower_quanta": region["gap_lower"],
            "response_energy_upper": region["profile_energy_upper"],
            "residual_at_1_kHz_quanta": region["remaining_measured_gap_lower"],
            "bound_zero_kHz": 1000 * (region["gap_lower"] - 2 * box["measurement_error_each"])
                              / region["profile_energy_upper"]}
    for row in summary["markov_all_length_error_certificates"]:
        tolerance_parameters[row["model"] + "_nominal_all_N"] = {
            "gap_lower_quanta": row["optimality_gap_lower"],
            "response_energy_upper": row["all_length_profile_energy_upper"],
            "residual_at_1_kHz_quanta": row["optimality_gap_lower"]
                - 2 * row["measurement_error_each"] - .001 * row["all_length_profile_energy_upper"],
            "bound_zero_kHz": 1000 * row["maximum_combined_force_error_after_measurement"]}
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "generator": Path(__file__).relative_to(ROOT).as_posix(), "input_files_sha256": input_hashes,
              "figures": {"S1": figure_s1(series), "S2": figure_s2(series)},
              "presentation_changes": ["White backgrounds, thin frames and no grids",
                                       "Bottom-centered panel letters without panel headings",
                                       "Larger Arial/STIX labels and distinct markers",
                                       "Exact coincident opposite-detuning curves use one legend entry",
                                       "Rate-cap segments apply to complete-rate bars only",
                                       "All 20 matched-trace cases shown with nested linewidth markers and no data offsets",
                                       "Broad and high-gain finite-N tolerance bounds both retained",
                                       "Independent neutral resource fills and hatching avoid model-color ambiguity"],
              "S1_samples": {"scalar_N": [27, 30, 33], "scalar_detunings": [-.03, 0, .03],
                             "two_band_N": [8, 9, 10], "two_band_detunings": [-.15, 0, .15]},
              "S2_scatter_jitter": "Original deterministic x-only display jitter, seed 20261007, +/-0.45 kHz",
              "S2_trace_budget": {
                  "cases": 20, "models": ["uniform", "dimerized"],
                  "linewidths_MHz": [80, 100], "trace_caps_MHz": [175, 190, 210, 250, 325],
                  "solid_filled_markers": "80 MHz", "dashed_open_markers": "100 MHz",
                  "display_offsets": "None; near-coincident values are shown at their saved coordinates"},
              "S2_continuous_regions": {
                  "scope": "Uniform chain, N=64, two specified feasible finite-auxiliary devices",
                  "broad_region": "Original continuous box, relative drift halfspan 0.005",
                  "high_gain_region": "Nested high-gain region, relative drift halfspan 0.00125",
                  "both_regions_shown": True,
                  "not_claimed": "All-N or continuous nearest-neighbor optimality separation"},
              "S2_nominal_markov": {
                  "scope": "Uniform and dimerized nominal chains for every N>=64",
                  "measurement_error_each_quanta": .05,
                  "negative_bound_meaning": "This conservative witness no longer certifies a positive resolvable difference; no sign reversal is implied"},
              "S2_tolerance_formula": "Delta_res = Delta_0 - 2*u - C_Q*delta; x=1000*delta kHz; u=0.05 quanta per device",
              "S2_tolerance_parameters": tolerance_parameters,
              "S2_resource_palette": RESOURCE_COLORS,
              "S2_resource_hatching": {"complete_loss_rate": "solid", "largest_addressed_coupling": "diagonal"},
              "S2_font_size_at_148_8_mm_width_pt": 9. * (148.8 / 25.4) / 7.5,
              "original_figure_S1_sha256": sha(ROOT / S1_OLD),
              "original_figure_S2_sha256": sha(CALC / "figures/robustness_and_resources.pdf")}
    assert input_hashes == {name: sha(ROOT / name) for name in INPUTS}
    PROVENANCE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_plotted_values_match_saved_results": True,
                      "provenance": PROVENANCE.relative_to(ROOT).as_posix(),
                      "figure_outputs": [S1_NEW, S2]}, indent=2))


if __name__ == "__main__":
    main()
