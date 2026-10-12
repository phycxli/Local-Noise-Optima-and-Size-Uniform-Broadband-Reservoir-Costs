"""Plot the current pointwise, shared-band and resource-design results."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from figure_palette import BLUE, INK, MODEL_COLORS, MUTED, RANGE_COLORS, RED
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import MODELS, normalized_profile, source_matrix
COLORS = RANGE_COLORS


def read_json(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))

def result_style(ax):
    ax.set_facecolor("white")
    ax.grid(False, which="both", axis="both")
    ax.tick_params(direction="in", length=2.6, width=.6, color=INK,
                   top=True, right=True, pad=3.0)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(.6)
    ax.set_axisbelow(True)

def result_figure_audit(fig, *, allow_grid=False):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    lines = {}
    for ax in fig.axes:
        artists += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title,
                                    ax._left_title, ax._right_title]
        if ax.get_legend() is not None:
            artists += list(ax.get_legend().get_texts())
        for line in ax.lines:
            if line.get_gid():
                data = np.column_stack((line.get_xdata(), line.get_ydata())).astype("<f8")
                lines[line.get_gid()] = hashlib.sha256(data.tobytes()).hexdigest()
    artists = [artist for artist in artists if artist.get_visible() and artist.get_text()]
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    collisions = [(artists[i].get_text(), artists[j].get_text())
                  for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                  if boxes[i].overlaps(boxes[j])]
    assert not collisions, collisions
    assert all(fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
               for box in boxes), "A result label is outside the figure."
    assert fig.get_facecolor() == (1., 1., 1., 1.)
    assert all(ax.get_facecolor() == (1., 1., 1., 1.) for ax in fig.axes)
    grid_lines = sum(line.get_visible() for ax in fig.axes
                     for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert allow_grid or grid_lines == 0
    panel_labels = []
    for ax in fig.axes:
        for artist in ax.texts:
            if artist.get_text() in ("(a)", "(b)", "(c)", "(d)"):
                box, axes_box = artist.get_window_extent(renderer), ax.get_window_extent(renderer)
                assert abs((box.x0 + box.x1 - axes_box.x0 - axes_box.x1) / 2) < 1
                assert box.y1 < axes_box.y0
                assert not any((ax.get_title(), ax.get_title(loc="left"), ax.get_title(loc="right")))
                panel_labels.append(artist.get_text())
    return {"figure_size_inches": fig.get_size_inches().tolist(),
            "text_labels": len(boxes), "label_collisions": collisions,
            "all_labels_within_figure": True,
            "all_backgrounds_pure_white": True, "visible_grid_lines": grid_lines,
            "bottom_center_panel_labels": panel_labels,
            "minimum_font_pt": min(artist.get_fontsize() for artist in artists),
            "displayed_line_data_sha256": lines}

def save(fig, name, dpi=240):
    directory = ROOT / "figures"
    directory.mkdir(exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{extension}", dpi=dpi)
    plt.close(fig)

def group_for(resources, model, width=.05):
    return next(g for g in resources["groups"] if g["model"] == model and g["halfwidth"] == width)

def nearest(group):
    return next(b for b in group["locality_bounds"] if b["radius"] == 1)

def pointwise_figure(rows):
    metadata = read_json("balanced_pointwise/audit.json")
    for name, digest in metadata["output_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    with (ROOT / metadata["pointwise_csv"]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (ROOT / metadata["frequency_csv"]).open(encoding="utf-8") as handle:
        spectrum = list(csv.DictReader(handle))
    fig = plt.figure(figsize=(3.38, 3.90))
    fig.set_facecolor("white")
    ax = fig.add_axes([.17, .615, .78, .355])
    rows = sorted(rows, key=lambda row: float(row["N"]))
    sizes = [float(row["N"]) for row in rows]
    ax.semilogy(sizes, [float(row["current_noise"]) for row in rows], color=INK, lw=1.3,
                label="original bonds", gid="pointwise_original")
    ax.semilogy(sizes, [float(row["fixed_drift_value"]) for row in rows], color=BLUE,
                lw=1.4, label="fixed-drift floor", gid="pointwise_fixed_drift")
    valid = [row for row in rows if float(row["construction_valid"]) > .5]
    ax.semilogy([float(row["N"]) for row in valid], [float(row["noise"]) for row in valid],
                "o", ms=3.9, markevery=[i for i, row in enumerate(valid) if i % 3 == 0 or float(row["N"]) == 60], mfc="white", mew=.9, color=RED,
                label="local construction", gid="pointwise_local")
    ax.semilogy(sizes, [float(row["caves_floor"]) for row in rows], "--", color=MUTED, lw=.95,
                dashes=(3, 2), label="Caves floor", gid="pointwise_caves")
    ax.set(xlabel=r"chain length $N$", ylabel="input-referred noise", ylim=(.3, 500))
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.015, .41), fontsize=7.2,
              handlelength=1.65, handletextpad=.45, labelspacing=.55)
    reference = next(row for row in rows if float(row["N"]) == 60)
    ratio = float(reference["current_noise"]) / float(reference["noise"])
    ax.text(.76, .74, rf"${ratio:.1f}\times$" + "\n" + r"at $N=60$",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.1, color=RED)
    ax.text(.5, -.30, "(a)", transform=ax.transAxes, ha="center", va="center", fontsize=8)
    result_style(ax)

    zoom = fig.add_axes([.275, .745, .30, .125])
    detail = [row for row in rows if 35 <= float(row["N"]) <= 60]
    detail_sizes = [float(row["N"]) for row in detail]
    zoom.plot(detail_sizes, [float(row["fixed_drift_value"]) for row in detail],
              color=BLUE, lw=1.1, gid="zoom_fixed_drift")
    zoom.plot(detail_sizes, [float(row["noise"]) for row in detail], "o", ms=3.1,
              mfc="white", mew=.8, color=RED, gid="zoom_local")
    zoom.plot(detail_sizes, [float(row["caves_floor"]) for row in detail], "--", lw=1,
              color=MUTED, dashes=(3, 2), gid="zoom_caves")
    zoom.set(xlim=(33, 62), ylim=(.42, 1.04), xticks=[40, 60], yticks=[.5, 1.0])
    zoom.set_title("Noise floors", loc="left", fontsize=7.2, pad=4, color=INK)
    result_style(zoom)
    zoom.set_facecolor("white")
    zoom.tick_params(labelsize=7.2, length=2, pad=1.8)

    band = fig.add_axes([.17, .14, .78, .32])
    omega = [float(row["omega"]) for row in spectrum]
    band.semilogy(omega, [float(row["original_noise"]) for row in spectrum],
                  color=INK, lw=1.3, gid="frequency_original")
    band.semilogy(omega, [float(row["fixed_local_noise"]) for row in spectrum],
                  color=RED, lw=1.4, gid="frequency_fixed_local")
    band.semilogy(omega, [float(row["pointwise_floor"]) for row in spectrum],
                  color=BLUE, lw=1.2, gid="frequency_pointwise_floor")
    center = spectrum[len(spectrum) // 2]
    band.plot([0], [float(center["fixed_local_noise"])], "o", mfc="white", mec=RED, ms=3.9, mew=.9)
    band.set(xlabel=r"detuning $\omega$", ylabel="input-referred noise",
             xlim=(-.05, .05), ylim=(.7, 500), xticks=[-.05, 0, .05])
    band.text(.5, .88, r"$N=60,\ \omega_0=0$", transform=band.transAxes,
              ha="center", va="center", fontsize=8)
    band.text(.5, -.32, "(b)", transform=band.transAxes, ha="center", va="center", fontsize=8)
    result_style(band)
    result = result_figure_audit(fig)
    renderer = fig.canvas.get_renderer()
    for axis in fig.axes:
        for ticks, limits, coordinate in ((axis.get_xticklabels(), axis.get_xlim(), 0),
                                           (axis.get_yticklabels(), axis.get_ylim(), 1)):
            for tick in ticks:
                if not tick.get_visible() or not limits[0] <= tick.get_position()[coordinate] <= limits[1]:
                    continue
                box = tick.get_window_extent(renderer)
                assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), tick.get_text()
    result["tick_labels_within_figure"] = True
    result["frequency_scan_metadata"] = "results/balanced_pointwise/audit.json"
    result["fixed_pointwise_reservoir_across_frequency"] = True
    save(fig, "fig2_pointwise_noise", dpi=600)
    return result

def shared_figure(rows, certificate, resources):
    model, n, width, cap = MODELS["uniform"], 64, .05, 8
    omega = np.linspace(-width, width, 1201)
    profiles = np.array([normalized_profile(model, n, w) for w in omega])
    source = np.asarray(source_matrix(model, n), dtype=float)
    floor = np.maximum(0, np.einsum("fi,ij,fj->f", profiles.conj(), source, profiles).real)
    fig = plt.figure(figsize=(7.05, 2.90))
    fig.set_facecolor("white")
    axes = [fig.add_axes(position) for position in
            ([.073, .24, .267, .58], [.414, .24, .222, .58], [.737, .24, .252, .58])]
    ax, plotted = axes[0], []
    for radius in ("all", "2", "1"):
        record = next(r for r in certificate["rows"] if r["N"] == n and
                      r["rate_cap"] == cap and r["radius"] == radius)
        path = ROOT / record["matrix_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["matrix_sha256"]:
            raise ValueError("The spectral reservoir changed after certification.")
        with np.load(path, allow_pickle=False) as saved:
            gain = saved["loss"] + source
            noise = np.einsum("fi,ij,fj->f", profiles.conj(), gain, profiles).real
        mean = float(simpson(noise, x=omega) / (2 * width))
        if not record["reported_lower"] - 1e-9 <= mean <= record["reported_upper"] + 1e-9:
            raise ValueError("Displayed spectrum disagrees with its certified band mean.")
        ax.plot(omega, noise, color=RANGE_COLORS[radius],
                lw=1.35, linestyle={"all": "-", "2": "--", "1": "-."}[radius],
                marker={"all": "o", "2": "s", "1": "^"}[radius],
                markevery=[300, 600, 900], ms=3.8, mew=.8, mfc="white",
                gid="shared_spectrum_" + radius,
                label="shared, all" if radius == "all" else rf"shared, $r={radius}$")
        plotted.append({"radius": radius, "matrix_file": record["matrix_file"],
                        "matrix_sha256": record["matrix_sha256"], "spectral_mean": mean,
                        "certified_interval": [record["reported_lower"], record["reported_upper"]],
                        "sample_frequencies": omega[::300].tolist(),
                        "sample_noise": noise[::300].tolist()})
    ax.plot(omega, floor, "--", color=MUTED, lw=.9, dashes=(3, 2),
            label="adaptive floor", gid="shared_adaptive_floor")
    ax.set(xlabel=r"frequency $\omega$", ylabel="input-referred noise", ylim=(.65, 4.45))
    ax.set_xticks([-.05, 0, .05])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2,
              fontsize=7.2, columnspacing=.7, handlelength=1.6, handletextpad=.3, borderaxespad=0)

    ax = axes[1]
    selected = {radius: next(r for r in rows if int(r["N"]) == n and
                float(r["rate_cap"]) == cap and r["radius"] == radius)
                for radius in ("all", "2", "1")}
    adaptive = float(selected["all"]["point_floor_average"])
    ax.scatter([0], [adaptive], color=MUTED, marker="x", s=22, linewidths=1.1, zorder=3)
    ax.text(0, adaptive + .17, f"{adaptive:.3f}", ha="center", fontsize=7.5, color=MUTED)
    for x, radius in enumerate(("all", "2", "1"), start=1):
        low, high = float(selected[radius]["lower"]), float(selected[radius]["upper"])
        mean = (low + high) / 2
        ax.vlines(x, adaptive, mean,
                  color=RANGE_COLORS[radius],
                  linewidth=.75, zorder=2)
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]],
                    color=RANGE_COLORS[radius],
                    fmt={"all": "o", "2": "s", "1": "^"}[radius], ms=4.6,
                    mfc="white", mew=.95, capsize=2.5, lw=.85, zorder=3)
        ax.text(x, mean + .17, f"{mean:.3f}", ha="center", fontsize=7.5,
                color=RANGE_COLORS[radius])
    ax.axhline(adaptive, color=MUTED, ls="--", lw=.85)
    ax.annotate("", xy=(1, 3.90), xytext=(3, 3.90),
                arrowprops={"arrowstyle": "|-|", "color": INK, "lw": .7,
                            "mutation_scale": 3})
    ax.text(2, 4.00, r"$\Delta_{\rm loc}\simeq1.065$", ha="center", va="bottom",
            fontsize=7.3, color=INK)
    ax.text(.98, 1.09, r"$N=64,\ R=8$", transform=ax.transAxes, ha="right", fontsize=7.5,
            color=INK)
    ax.set(xticks=range(4), xticklabels=["adaptive", "all", "2", "1"],
           xlabel="reservoir design", ylabel="band-average noise",
           xlim=(-.55, 3.55), ylim=(.65, 4.45))
    ax.tick_params(axis="x", labelsize=7.2)

    ax = axes[2]
    for y, name in ((1, "uniform"), (0, "dimerized")):
        group = group_for(resources, name)
        bound = nearest(group)
        upper, lower = group["nonlocal_construction_upper"], bound["all_N_local_noise_lower_at_R8"]
        ax.plot([upper, lower], [y, y], color=INK, lw=.8, solid_capstyle="butt", zorder=2)
        ax.scatter([upper], [y], facecolors="white", marker="v", s=35,
                   edgecolors=INK, linewidths=.9, zorder=3,
                   label="reference upper" if y else None)
        ax.scatter([lower], [y], facecolors="white", marker="^", s=35,
                   edgecolors=RED, linewidths=.9, zorder=3,
                   label="nearest-neighbor lower" if y else None)
        ax.text((upper + lower) / 2, y + .20,
                f"gap > {np.floor(bound['all_N_locality_gap_lower_at_R8'] * 1e5) / 1e5:.5f}",
                ha="center", fontsize=7.7, color=INK)
    ax.set(yticks=[0, 1], yticklabels=["dimerized", "uniform"], xlabel="band-average noise bound",
           xlim=(2.25, 3.65), ylim=(-.75, 1.55))
    ax.set_xticks([2.4, 2.8, 3.2, 3.6])
    ax.text(.98, 1.09, r"$N\geq64,\ R=8$", transform=ax.transAxes, ha="right", fontsize=7.5,
            color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=7.2, handletextpad=.3,
              borderaxespad=.1, labelspacing=.4)
    for ax, letter in zip(axes, ("(a)", "(b)", "(c)")):
        result_style(ax)
        ax.annotate(letter, xy=(.5, 0), xycoords="axes fraction", xytext=(0, -33),
                    textcoords="offset points", ha="center", va="top", fontsize=8.4,
                    annotation_clip=False)
    shared_figure.presentation_audit = result_figure_audit(fig)
    save(fig, "fig3_shared_band_cost", dpi=600)
    return {"frequency_grid_points": len(omega),
            "pointwise_floor_mean": float(simpson(floor, x=omega) / (2 * width)),
            "shared_spectra": plotted, "spectra_not_claimed_pointwise_ordered": True,
            "all_length_panel_shows_bounds_not_exact_optima": True}

def design_figure(resources, audit, frontier):
    fig, axes = plt.subplots(3, 1, figsize=(3.38, 4.60), layout="constrained",
                             gridspec_kw={"height_ratios": [1.12, 1.03, .87]})
    group, caps, ax = group_for(resources, "uniform"), np.linspace(8, 16, 201), axes[0]
    for bound in sorted(group["locality_bounds"], key=lambda b: b["radius"]):
        radius = str(bound["radius"])
        ax.plot(caps, np.maximum(0, bound["alpha"] - caps * bound["beta_with_boundary"]),
                color=COLORS[radius], lw=1.2, label=rf"$r={radius}$ lower", gid="design_rate_" + radius)
    upper = group["nonlocal_construction_upper"]
    target = upper + .1
    local = nearest(group)
    ax.fill_between(caps, target,
                    np.maximum(target, local["alpha"] - caps * local["beta_with_boundary"]),
                    color=COLORS["1"], alpha=.09, linewidth=0)
    ax.axhline(upper, color=COLORS["all"], lw=1, label="fixed reference upper", gid="design_reference")
    ax.axhline(target, color=MUTED, ls="--", lw=.8, gid="design_target")
    threshold = nearest(group)["necessary_R_for_reference_plus_0p1"]
    ax.axvline(threshold, color=MUTED, ls=":", lw=.8, gid="design_necessary_rate")
    ax.text(threshold + .18, 2.92, "13.0093", fontsize=7.4, color=MUTED)
    ax.text(8.15, 2.68, "ref. + 0.1", color=MUTED, fontsize=7.2)
    ax.set(xlabel=r"complete rate cap $R$", ylabel="noise bound", xlim=(8, 16), ylim=(1.85, 3.9))
    ax.set_yticks([2.0, 2.5, 3.0, 3.5])
    ax.text(8.15, 3.67, r"$r=1$ lower", color=COLORS["1"], fontsize=7.2)
    ax.text(8.15, 1.98, r"$r=2$ lower", color=COLORS["2"], fontsize=7.2)
    ax.text(.98, .93, r"$\Omega=0.05$", transform=ax.transAxes,
            fontsize=7.2, ha="right", va="top", color=INK)

    ax = axes[1]
    data_hashes = {}
    for name, color in MODEL_COLORS.items():
        certified = next(g for g in frontier["groups"] if g["model"] == name)
        edges = [c["halfwidth_interval"][0] for c in certified["cells"]]
        edges.append(certified["cells"][-1]["halfwidth_interval"][1])
        gaps = [c["gap_lower"] for c in certified["cells"]]
        lowers = [c["local_lower"] for c in certified["cells"]]
        uppers = [c["reference_upper"] for c in certified["cells"]]
        ax.stairs(lowers, edges, color=color, lw=1.1, label=name + " lower", baseline=None)
        ax.stairs(uppers, edges, color=color, lw=1.1, ls="--", label=name + " upper", baseline=None)
        data_hashes[name] = hashlib.sha256(np.column_stack(
            (edges[:-1], edges[1:], lowers, uppers, gaps)).astype("<f8").tobytes()).hexdigest()
    ax.set(xlabel=r"band halfwidth $\Omega$", ylabel="band-average\nnoise bound",
           xlim=(.029, .071), ylim=(1.8, 4.45))
    ax.set_xticks([.03, .05, .07])
    ax.legend(frameon=False, loc="upper left", fontsize=7.2, labelspacing=.2,
              ncols=2, columnspacing=.6, handlelength=1.3, handletextpad=.35,
              borderaxespad=.15, borderpad=.2)
    ax.text(.97, .06, r"$N\geq64,\ R=8$", transform=ax.transAxes, ha="right",
            fontsize=7.2, color=INK)

    ax = axes[2]
    intervals = {}
    target_bound_hashes = {}
    for y, name, target in ((1, "uniform", 2.6), (0, "dimerized", 2.9)):
        lower = nearest(group_for(resources, name))["all_N_local_noise_lower_at_R8"]
        upper = next(c for c in audit["range_two_constructions"] if c["model"] == name)["all_N_noise_upper"]
        if not upper < target < lower:
            raise ValueError("Minimum-range target is not separated by the certified bounds.")
        certified = next(g for g in frontier["groups"] if g["model"] == name)
        intervals[name] = certified["minimum_range_two_intervals"]
        color = MODEL_COLORS[name]
        edges = [cell["halfwidth_interval"][0] for cell in certified["cells"]]
        edges.append(certified["cells"][-1]["halfwidth_interval"][1])
        lowers = [cell["local_lower"] - target for cell in certified["cells"]]
        range2_uppers = [cell["range2_upper"] - target for cell in certified["cells"]]
        ax.stairs(lowers, edges, color=color, lw=1.1, baseline=None,
                  label=r"$r=1$ lower" if y else None)
        ax.stairs(range2_uppers, edges, color=color, lw=1.1, ls="--", baseline=None,
                  label=r"$r=2$ upper" if y else None)
        target_bound_hashes[name] = hashlib.sha256(np.column_stack(
            (edges[:-1], edges[1:], lowers, range2_uppers)).astype("<f8").tobytes()).hexdigest()
        for a, b in intervals[name]:
            cell = next(cell for cell in certified["cells"] if cell["halfwidth_interval"][1] == b)
            assert cell["minimum_range_two_certified"]
            ordinate = cell["range2_upper"] - target
            ax.plot([b], [ordinate], "o", color=color, ms=3.5, mfc="white", mew=.9,
                    gid="design_range_endpoint_" + name)
            ax.annotate(f"{b:.4f}", xy=(b, ordinate),
                        xytext=(.042 if y else .0615, -.18 if y else -.32),
                        fontsize=7.2, ha="center", color=color,
                        arrowprops={"arrowstyle": "-", "color": color, "lw": .6})
    ax.axhline(0, color=MUTED, ls=":", lw=.8, gid="design_target_zero")
    ax.axvline(.05, color=MUTED, ls=":", lw=.65)
    ax.set(xlim=(.029, .071), ylim=(-.45, 1.18), xlabel=r"band halfwidth $\Omega$",
           ylabel=r"noise bound $-T$")
    ax.set_yticks([-.4, 0, .4, .8])
    ax.set_xticks([.03, .05, .07])
    ax.legend(frameon=False, loc="upper left", fontsize=7.2, ncols=2,
              handlelength=1.4, handletextpad=.35, columnspacing=.8,
              borderaxespad=.15, borderpad=.2)
    for ax, letter in zip(axes, ("(a)", "(b)", "(c)")):
        result_style(ax)
        ax.annotate(letter, xy=(.5, 0), xycoords="axes fraction", xytext=(0, -33),
                    textcoords="offset points", ha="center", va="top", fontsize=8.4,
                    annotation_clip=False)
    design_figure.presentation_audit = result_figure_audit(fig)
    save(fig, "fig4_design_targets", dpi=600)
    return {"continuous_cell_data_sha256": data_hashes,
            "same_reference_device_across_bandwidths": True,
            "minimum_range_two_intervals": intervals,
            "target_relative_bounds_sha256": target_bound_hashes,
            "range_two_panel_shows_certified_bounds_relative_to_target": True,
            "presentation": design_figure.presentation_audit}

def main():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.1,
                         "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    with (ROOT / "results/balanced_pointwise/pointwise.csv").open(encoding="utf-8") as handle:
        points = list(csv.DictReader(handle))
    with (ROOT / "results/rate_capped_band_optimization.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    certificate = read_json("rate_capped_band_optimization_certificate.json")
    resources = read_json("size_uniform_resources_certificate.json")
    audit = read_json("size_uniform_resources_audit.json")
    frontier = read_json("fig4_shared_device_certificate.json")
    assert certificate["all_checks_passed"] and resources["all_checks_passed"] and frontier["all_checks_passed"]
    result = {"pointwise": pointwise_figure(points),
              "shared": shared_figure(rows, certificate, resources),
              "design": design_figure(resources, audit, frontier)}
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
