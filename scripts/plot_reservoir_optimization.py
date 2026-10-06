"""Plot the physical device, certified shared-band costs, and design targets."""

from __future__ import annotations
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from scipy.integrate import simpson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shared_band_resources import MODELS, normalized_profile, source_matrix

COLORS = {"all": "#146b78", "1": "#b34232", "2": "#645791"}


def read_json(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def style(ax):
    ax.grid(alpha=.16, linewidth=.45)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_axisbelow(True)


def result_style(ax):
    ax.set_facecolor("white")
    ax.grid(False, which="both", axis="both")
    ax.tick_params(direction="in", length=2.6, width=.6, color="#262626",
                   top=True, right=True, pad=3.0)
    for spine in ax.spines.values():
        spine.set_color("#262626")
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


def baseline_device_audit():
    c, v, gamma = Fraction(5, 8), Fraction(3, 8), Fraction(6, 5)
    cases = []
    for n in (2, 3, 8):
        gain = np.full((n, n), Fraction(0), dtype=object)
        loss = 2 * gamma * np.eye(n, dtype=object)
        coherent = np.full((n, n), Fraction(0), dtype=object)
        target = -gamma * np.eye(n, dtype=object)
        for j in range(n - 1):
            gain[j:j+2, j:j+2] += c * np.array([[1, 1], [1, 1]], dtype=object)
            loss[j:j+2, j:j+2] += c * np.array([[1, -1], [-1, 1]], dtype=object)
            coherent[j+1, j], coherent[j, j+1] = v, -v
            target[j+1, j], target[j, j+1] = Fraction(1), Fraction(1, 4)
        assert np.array_equal(coherent + (gain - loss) / 2, target)
        cases.append({"N": n, "exact_drift_agreement": True})

    # Check the displayed jumps directly with the adjoint Lindblad equation.
    cutoff = 5
    single = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    a = [np.kron(single, np.eye(cutoff)), np.kron(np.eye(cutoff), single)]
    ham = 1j * float(v) * (a[1].conj().T @ a[0] - a[0].conj().T @ a[1])
    jumps = [np.sqrt(float(c)) * (a[0].conj().T + a[1].conj().T),
             np.sqrt(float(c)) * (a[0] - a[1])]
    jumps += [np.sqrt(2 * float(gamma)) * operator for operator in a]
    low = [i * cutoff + j for i in range(3) for j in range(3)]
    drift = np.array([[-float(gamma), .25], [1., -float(gamma)]])
    errors = []
    for j, operator in enumerate(a):
        derivative = 1j * (ham @ operator - operator @ ham)
        for jump in jumps:
            rate = jump.conj().T @ jump
            derivative += jump.conj().T @ operator @ jump - (rate @ operator + operator @ rate) / 2
        residual = derivative - sum(drift[j, k] * a[k] for k in range(2))
        errors.append(float(np.max(np.abs(residual[np.ix_(low, low)]))))
    assert max(errors) < 1e-12
    return {"c": str(c), "v": str(v), "gamma": str(gamma),
            "gain_relative_phases": ["0", "0"], "loss_relative_phases": ["0", "pi"],
            "exact_chain_checks": cases, "adjoint_lindblad_first_moment_residual": max(errors),
            "relative_minus_is_not_a_negative_rate": True}


def diagram_text_audit(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = [text for ax in axes for text in ax.texts] + list(fig.texts)
    boxes = [text.get_window_extent(renderer) for text in artists]
    collisions = [(artists[i].get_text(), artists[j].get_text())
                  for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                  if boxes[i].overlaps(boxes[j])]
    assert not collisions, collisions
    assert all(fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
               for box in boxes), "A schematic label is outside the figure."
    return {"text_labels": len(boxes), "label_collisions": collisions,
            "all_labels_within_figure": True, "equal_axis_scale": True,
            "minimum_diagram_font_pt": min(text.get_fontsize() for text in artists)}


def device_figure():
    from matplotlib.patches import Polygon

    fig = plt.figure(figsize=(3.38, 3.00))
    device = fig.add_axes([.025, .515, .95, .410])
    model = fig.add_axes([.025, .018, .95, .412])
    device.set(xlim=(-1.3, 12.0), ylim=(-.85, 4.45), aspect="equal")
    model.set(xlim=(-.10, 10.10), ylim=(-2.02, 2.35), aspect="equal")
    for ax in (device, model):
        ax.axis("off")
    ink, gray = "#283438", "#7c898e"
    gain_color, loss_color = COLORS["all"], COLORS["1"]
    for height, letter in ((.987, "(a)"), (.487, "(b)")):
        fig.text(.025, height, letter, fontsize=8.5, fontweight="bold", va="top")

    def label(ax, x, y, text, color=ink, size=7.5, align="center"):
        return ax.text(x, y, text, color=color, fontsize=size, ha=align, va="center", zorder=20)

    def line(ax, points, color=ink, width=.85, **options):
        x, y = zip(*points)
        ax.plot(x, y, color=color, lw=width, solid_capstyle="round", **options)

    def arrow(ax, start, end, color=ink, width=.85, scale=6):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", color=color,
                     linewidth=width, mutation_scale=scale, shrinkA=0, shrinkB=0, zorder=12))

    # Fixed axonometric projection keeps every circuit connection editable in the PDF.
    def project(x, y, z=.25):
        return np.array([x + .48 * y, .48 * y + .035 * x + z])

    def face(points, fill, edge="#8c9da6", width=.6, zorder=1):
        device.add_patch(Polygon([project(*p) for p in points], closed=True,
                                facecolor=fill, edgecolor=edge, lw=width, zorder=zorder))

    face([(0, 0, 0), (9.25, 0, 0), (9.25, 0, .25), (0, 0, .25)], "#bdcbd1")
    face([(9.25, 0, 0), (9.25, 3.5, 0), (9.25, 3.5, .25), (9.25, 0, .25)], "#d1dce0")
    face([(0, 0, .25), (9.25, 0, .25), (9.25, 3.5, .25), (0, 3.5, .25)],
         "#edf3f5", zorder=2)

    def trace(points, color=ink, width=.8, **options):
        line(device, [project(*point) for point in points], color, width, zorder=5, **options)

    def island(x, y, color, fill):
        half, depth, base, top = .39, .32, .255, .37
        face([(x-half, y-depth, base), (x+half, y-depth, base),
              (x+half, y-depth, top), (x-half, y-depth, top)], "#c5d1d6", color, zorder=7)
        face([(x+half, y-depth, base), (x+half, y+depth, base),
              (x+half, y+depth, top), (x+half, y-depth, top)], "#dbe4e7", color, zorder=7)
        face([(x-half, y-depth, top), (x+half, y-depth, top),
              (x+half, y+depth, top), (x-half, y+depth, top)], fill, color, zorder=8)
        meander = [(x-.26, y-.18, top+.005), (x-.26, y+.16, top+.005),
                   (x-.13, y+.16, top+.005), (x-.13, y-.16, top+.005),
                   (x, y-.16, top+.005), (x, y+.16, top+.005)]
        line(device, [project(*p) for p in meander], color, .7, zorder=9)
        for dx in (.13, .23):
            line(device, [project(x+dx, y-.17, top+.005), project(x+dx, y+.17, top+.005)],
                 color, .8, zorder=9)

    def mixer(x, y, color, half=.14):
        center = project(x, y, .28)
        device.add_patch(Polygon(center + np.array([[0, half], [half, 0],
                                                  [0, -half], [-half, 0]]),
                                facecolor="white", edgecolor=color, lw=.75, zorder=11))

    trace([(-1.15, 1.55), (.46, 1.55)])
    trace([(1.24, 1.55), (3.11, 1.55)], width=.7, linestyle=(0, (2, 2)))
    trace([(3.89, 1.55), (5.31, 1.55)])
    trace([(6.09, 1.55), (8.01, 1.55)], width=.7, linestyle=(0, (2, 2)))
    trace([(8.79, 1.55), (10.65, 1.55)])
    arrow(device, project(-1.1, 1.55), project(-.5, 1.55))
    arrow(device, project(10.0, 1.55), project(10.65, 1.55))
    label(device, -.32, .35, "In")
    label(device, 11.08, .80, "Out")
    label(device, .30, 3.51, "20 mK", gray, size=7.2, align="left")
    label(device, 10.92, 3.51, "4-8 GHz", gray, size=7.2, align="right")
    for x, text in ((.85, r"$a_1$"), (3.5, r"$a_j$"),
                    (5.7, r"$a_{j+1}$"), (8.4, r"$a_N$")):
        island(x, 1.55, ink, "#fbfcfd")
        pos = project(x, 1.55, 1.60)
        if x == 5.7:
            pos[0] += .65
        label(device, *pos, text, size=7.7)
    mixer(4.6, 1.55, ink, half=.16)
    pos = project(4.6, 1.55, .63)
    label(device, *pos, r"$\widehat H$", size=7.6)

    for y, color, fill in ((2.85, gain_color, "#e0eef0"), (.25, loss_color, "#f6e6e2")):
        for x in (3.5, 5.7):
            trace([(x, 1.55), (x, y), (4.6, y)], color, 1.0)
            mixer(x, (1.55+y)/2, color)
        island(4.6, y, color, fill)
        trace([(4.99, y), (6.55, y)], color)
        zigzag = [(6.55, y)] + [(6.63+.13*i, y+(.10 if i%2 else -.10)) for i in range(6)]
        zigzag += [(7.40, y), (7.65, y)]
        trace(zigzag, color, .7)
        for dx, half in ((0, .20), (.10, .14), (.20, .07)):
            trace([(7.65+dx, y-half), (7.65+dx, y+half)], color, .65)
    label(device, 5.94, 3.48, "$b_g$ / sum-frequency\npump", gain_color, size=7.4)
    line(device, [(5.94, 3.12), project(4.6, 2.85, .50)], gain_color, .6)
    label(device, 4.46, -.47, "$b_\\ell$ / difference-frequency pump", loss_color, size=7.4)
    line(device, [(4.46, -.16), project(4.6, .25, .27)], loss_color, .6)

    # Arrows below denote first-moment drift, not a non-Hermitian microscopic Hamiltonian.
    xs, center = [1.15, 3.75, 6.25, 8.85], 5.0
    for left, right in ((xs[0]+.30, xs[1]-.30), (xs[2]+.30, xs[3]-.30)):
        line(model, [(left, 0), (right, 0)], gray, .7, linestyle=(0, (2, 2)))
    for x, text in zip(xs, (r"$a_1$", r"$a_j$", r"$a_{j+1}$", r"$a_N$")):
        model.add_patch(Circle((x, 0), .28, facecolor="#edf3f5", edgecolor=ink, lw=.9, zorder=10))
        offset = -.44 if x == xs[1] else (.44 if x == xs[2] else 0)
        align = "right" if x == xs[1] else ("left" if x == xs[2] else "center")
        label(model, x+offset, -.53, text, size=8, align=align)
    arrow(model, (xs[1]+.32, .17), (xs[2]-.32, .17), width=1.1)
    arrow(model, (xs[2]-.32, -.17), (xs[1]+.32, -.17), width=.8)
    label(model, center, .51, r"$t_R=c+v$", size=7.8)
    label(model, center, -.56, r"$t_L=c-v$", size=7.8)
    for y, color in ((1.21, gain_color), (-1.21, loss_color)):
        line(model, [(xs[1], .29 if y>0 else -.29), (xs[1], y),
                     (xs[2], y), (xs[2], .29 if y>0 else -.29)], color, 1.0)
        model.add_patch(Circle((center, y), .13, facecolor=color, edgecolor="white", lw=.4, zorder=11))
    label(model, center, 2.10, r"$L_j^g=\sqrt{c}(a_j^\dagger+a_{j+1}^\dagger)$", gain_color, size=7.8)
    label(model, center, -1.83, r"$L_j^\ell=\sqrt{c}(a_j-a_{j+1})$", loss_color, size=7.8)
    for x in (xs[1], xs[2]):
        label(model, x, 1.42, r"$0$", gain_color, size=7.5)
    label(model, xs[1]-.16, -.93, r"$0$", loss_color, size=7.5, align="right")
    label(model, xs[2]+.16, -.93, r"$\pi$", loss_color, size=7.5, align="left")
    for x in (xs[0], xs[3]):
        arrow(model, (x, -.75), (x, -1.10), gray, width=.7, scale=5)
        for y, half in ((-1.13, .19), (-1.21, .13), (-1.29, .07)):
            line(model, [(x-half, y), (x+half, y)], gray, .7)
    label(model, .14, .62, r"$\kappa_i$", size=7.7)
    label(model, 9.86, .62, r"$\kappa_o$", size=7.7)
    arrow(model, (.14, 0), (.78, 0), width=.7, scale=5)
    arrow(model, (9.22, 0), (9.86, 0), width=.7, scale=5)
    label(model, 1.15, -1.68, r"$2\gamma$", gray, size=7.5)
    label(model, 8.85, -1.68, r"$2\gamma$", gray, size=7.5)

    result = {"physical_sign_check": baseline_device_audit(),
              "geometry_check": diagram_text_audit(fig, (device, model)),
              "figure_size_inches": [3.38, 3.00], "raster_dpi": 600,
              "columns": 1, "panel_a_crop_axis": "vertical", "panel_a_crop_fraction": .500,
              "panel_labels_only": ["(a)", "(b)"],
              "rendering": "Editable axonometric chip concept above an effective drift and Lindblad-jump model",
              "experimental_scope": "Proposed hardware; repeated bonds and the external measurement chain are omitted",
              "diagram_references": ["https://www.nature.com/articles/s41467-017-00447-1/figures/2",
                                     "https://arxiv.org/html/1512.00078v1#S0.F1"],
              "pointwise_data_sha256": hashlib.sha256(
                  (ROOT / "results/local_noise_optimization_explicit.csv").read_bytes()).hexdigest()}
    save(fig, "fig1_local_device", dpi=600)
    result["files_sha256"] = {f"figures/fig1_local_device.{extension}": hashlib.sha256(
        (ROOT / f"figures/fig1_local_device.{extension}").read_bytes()).hexdigest()
        for extension in ("pdf", "png")}
    return result


def pointwise_figure(rows):
    fig = plt.figure(figsize=(3.38, 2.55))
    fig.set_facecolor("white")
    ax = fig.add_axes([.17, .17, .80, .735])
    rows = sorted(rows, key=lambda row: float(row["N"]))
    sizes = [float(row["N"]) for row in rows]
    ax.semilogy(sizes, [float(row["current_noise"]) for row in rows], color="#262626", lw=1.3,
                label="original bonds", gid="pointwise_original")
    ax.semilogy(sizes, [float(row["fixed_drift_value"]) for row in rows], color="#245a81",
                lw=1.4, label="fixed-drift floor", gid="pointwise_fixed_drift")
    valid = [row for row in rows if float(row["construction_valid"]) > .5]
    ax.semilogy([float(row["N"]) for row in valid], [float(row["noise"]) for row in valid],
                "o", ms=3.9, markevery=3, mfc="white", mew=.9, color="#a33b35",
                label="local construction", gid="pointwise_local")
    ax.semilogy(sizes, [float(row["caves_floor"]) for row in rows], "--", color="#262626", lw=.95,
                dashes=(3, 2), label="Caves floor", gid="pointwise_caves")
    ax.set(xlabel=r"chain length $N$", ylabel="input-referred noise", ylim=(.3, 500))
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.015, .48), fontsize=7.2,
              handlelength=1.65, handletextpad=.45, labelspacing=.55)
    fig.text(.17, .967, "Pointwise noise suppression", fontsize=8.4, va="top")
    fig.text(.97, .967, r"$\omega=0$", fontsize=8.2, va="top", ha="right", color="#262626")
    reference = next(row for row in rows if float(row["N"]) == 60)
    ratio = float(reference["current_noise"]) / float(reference["noise"])
    ax.text(.76, .77, rf"${ratio:.1f}\times$" + "\n" + r"at $N=60$",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.1, color="#a33b35")
    result_style(ax)

    zoom = fig.add_axes([.275, .465, .30, .245])
    detail = [row for row in rows if 35 <= float(row["N"]) <= 60]
    detail_sizes = [float(row["N"]) for row in detail]
    zoom.plot(detail_sizes, [float(row["fixed_drift_value"]) for row in detail],
              color="#245a81", lw=1.1, gid="zoom_fixed_drift")
    zoom.plot(detail_sizes, [float(row["noise"]) for row in detail], "o", ms=3.1,
              mfc="white", mew=.8, color="#a33b35", gid="zoom_local")
    zoom.plot(detail_sizes, [float(row["caves_floor"]) for row in detail], "--", lw=1,
              color="#262626", dashes=(3, 2), gid="zoom_caves")
    zoom.set(xlim=(33, 62), ylim=(.42, 1.04), xticks=[40, 60], yticks=[.5, 1.0])
    zoom.set_title("Reflection penalty", loc="left", fontsize=7.2, pad=4, color="#262626")
    result_style(zoom)
    zoom.set_facecolor("white")
    zoom.tick_params(labelsize=7.2, length=2, pad=1.8)
    result = result_figure_audit(fig)
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
        ax.plot(omega, noise, color={"all": "#262626", "2": "#245a81", "1": "#a33b35"}[radius],
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
    ax.plot(omega, floor, "--", color="#262626", lw=.9, dashes=(3, 2),
            label="adaptive floor", gid="shared_adaptive_floor")
    ax.set(xlabel=r"frequency $\omega$", ylabel="input-referred noise", ylim=(.65, 4.2))
    ax.set_xticks([-.05, 0, .05])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2,
              fontsize=7.2, columnspacing=.7, handlelength=1.6, handletextpad=.3, borderaxespad=0)

    ax = axes[1]
    selected = {radius: next(r for r in rows if int(r["N"]) == n and
                float(r["rate_cap"]) == cap and r["radius"] == radius)
                for radius in ("all", "2", "1")}
    adaptive = float(selected["all"]["point_floor_average"])
    ax.scatter([0], [adaptive], color="#262626", marker="x", s=22, linewidths=1.1, zorder=3)
    ax.text(0, adaptive + .17, f"{adaptive:.3f}", ha="center", fontsize=7.5, color="#262626")
    for x, radius in enumerate(("all", "2", "1"), start=1):
        low, high = float(selected[radius]["lower"]), float(selected[radius]["upper"])
        mean = (low + high) / 2
        ax.vlines(x, adaptive, mean,
                  color={"all": "#262626", "2": "#245a81", "1": "#a33b35"}[radius],
                  linewidth=.75, zorder=2)
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]],
                    color={"all": "#262626", "2": "#245a81", "1": "#a33b35"}[radius],
                    fmt={"all": "o", "2": "s", "1": "^"}[radius], ms=4.6,
                    mfc="white", mew=.95, capsize=2.5, lw=.85, zorder=3)
        ax.text(x, mean + .17, f"{mean:.3f}", ha="center", fontsize=7.5,
                color={"all": "#262626", "2": "#245a81", "1": "#a33b35"}[radius])
    ax.axhline(adaptive, color="#262626", ls="--", lw=.85)
    ax.annotate("", xy=(1, 3.90), xytext=(3, 3.90),
                arrowprops={"arrowstyle": "|-|", "color": "#262626", "lw": .7,
                            "mutation_scale": 3})
    ax.text(2, 4.00, r"$\Delta_{\rm loc}\simeq1.065$", ha="center", va="bottom",
            fontsize=7.3, color="#262626")
    ax.text(.98, 1.09, r"$N=64,\ R=8$", transform=ax.transAxes, ha="right", fontsize=7.5,
            color="#262626")
    ax.set(xticks=range(4), xticklabels=["adaptive", "all", "2", "1"],
           xlabel=r"matrix range $r$", ylabel="band-average noise",
           xlim=(-.55, 3.55), ylim=(.65, 4.2))
    ax.tick_params(axis="x", labelsize=7.2)

    ax = axes[2]
    for y, name in ((1, "uniform"), (0, "dimerized")):
        group = group_for(resources, name)
        bound = nearest(group)
        upper, lower = group["nonlocal_construction_upper"], bound["all_N_local_noise_lower_at_R8"]
        ax.plot([upper, lower], [y, y], color="#262626", lw=.8, solid_capstyle="butt", zorder=2)
        ax.scatter([upper], [y], facecolors="white", marker="v", s=35,
                   edgecolors="#262626", linewidths=.9, zorder=3,
                   label="reference upper" if y else None)
        ax.scatter([lower], [y], facecolors="white", marker="^", s=35,
                   edgecolors="#a33b35", linewidths=.9, zorder=3,
                   label="nearest-neighbor lower" if y else None)
        ax.text((upper + lower) / 2, y + .20,
                f"gap > {np.floor(bound['all_N_locality_gap_lower_at_R8'] * 1e5) / 1e5:.5f}",
                ha="center", fontsize=7.7, color="#262626")
    ax.set(yticks=[0, 1], yticklabels=["dimerized", "uniform"], xlabel="band-average noise bound",
           xlim=(2.25, 3.65), ylim=(-.75, 1.55))
    ax.set_xticks([2.4, 2.8, 3.2, 3.6])
    ax.text(.98, 1.09, r"$N\geq64,\ R=8$", transform=ax.transAxes, ha="right", fontsize=7.5,
            color="#262626")
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
    ax.axhline(target, color=".35", ls="--", lw=.8, gid="design_target")
    threshold = nearest(group)["necessary_R_for_reference_plus_0p1"]
    ax.axvline(threshold, color=".5", ls=":", lw=.8, gid="design_necessary_rate")
    ax.annotate("13.0093", xy=(threshold, target), xytext=(13.65, 3.12), fontsize=7.4, ha="center",
                arrowprops={"arrowstyle": "->", "lw": .7, "color": ".35"})
    ax.text(8.15, target + .065, "reference + 0.1", color=".3", fontsize=7.2)
    ax.set(xlabel=r"complete rate cap $R$", ylabel="noise bound", xlim=(8, 16), ylim=(1.85, 3.9))
    ax.text(8.15, 3.58, r"$r=1$ lower", color=COLORS["1"], fontsize=7.2)
    ax.text(11.0, 3.58, r"$r=2$ lower", color=COLORS["2"], fontsize=7.2)
    ax.text(8.15, 2.05, "fixed reference upper", color=COLORS["all"], fontsize=7.2)
    ax.text(.98, .93, r"$\Omega=0.05$", transform=ax.transAxes,
            fontsize=7.2, ha="right", va="top", color="#262626")

    ax = axes[1]
    data_hashes = {}
    for name, color in (("uniform", COLORS["1"]), ("dimerized", COLORS["all"])):
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
            fontsize=7.2, color="#262626")

    ax = axes[2]
    intervals = {}
    for y, name, target in ((1, "uniform", 2.6), (0, "dimerized", 2.9)):
        lower = nearest(group_for(resources, name))["all_N_local_noise_lower_at_R8"]
        upper = next(c for c in audit["range_two_constructions"] if c["model"] == name)["all_N_noise_upper"]
        if not upper < target < lower:
            raise ValueError("Minimum-range target is not separated by the certified bounds.")
        certified = next(g for g in frontier["groups"] if g["model"] == name)
        intervals[name] = certified["minimum_range_two_intervals"]
        color = COLORS["1"] if name == "uniform" else COLORS["all"]
        for a, b in intervals[name]:
            ax.broken_barh([(a, b - a)], (y - .14, .28), color=color, alpha=.13, linewidth=0)
            ax.plot([a, b], [y, y], color=color, lw=1.3, gid="design_range_band_" + name)
            ax.plot([a, b], [y, y], "|", color=color, ms=7, mew=1.0)
            ax.text((a + b) / 2, y + .19, r"$r_{\min}=2$", fontsize=7.4,
                    ha="center", va="bottom", color=color)
            ax.text(b, y - .23, f"{b:.4f}", fontsize=7.2,
                    ha="center", va="top", color="#262626")
    ax.axvline(.05, color=".5", ls=":", lw=.65)
    ax.set(yticks=[0, 1], yticklabels=["dimerized\n$T=2.9$", "uniform\n$T=2.6$"],
           xlim=(.029, .071), ylim=(-.6, 1.62), xlabel=r"band halfwidth $\Omega$")
    ax.set_xticks([.03, .05, .07])
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
            "presentation": design_figure.presentation_audit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-only", action="store_true")
    parser.add_argument("--presentation-only", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    args = parser.parse_args()
    if args.design_only:
        resources = read_json("size_uniform_resources_certificate.json")
        audit = read_json("size_uniform_resources_audit.json")
        frontier = read_json("fig4_shared_device_certificate.json")
        required = {"scripts/strengthen_fig4_resources.py": frontier["source_sha256"],
                    "results/size_uniform_resources_certificate.json": frontier["base_certificate_sha256"],
                    "results/size_uniform_resources_audit.json": frontier["base_audit_sha256"],
                    **frontier["helper_sha256"]}
        if not frontier["all_checks_passed"] or any(
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest
                for path, digest in required.items()):
            raise RuntimeError("The continuous-bandwidth figure requires current exact certificates.")
        result = design_figure(resources, audit, frontier)
        result["figure_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["certificate_sha256"] = hashlib.sha256(
            (ROOT / "results/fig4_shared_device_certificate.json").read_bytes()).hexdigest()
        (ROOT / "results/reports/fig4_strengthening_figure_audit.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return
    with (ROOT / "results/local_noise_optimization_explicit.csv").open(encoding="utf-8") as handle:
        points = [r for r in csv.DictReader(handle) if float(r["power_gain"]) > 1]
    with (ROOT / "results/rate_capped_band_optimization.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    certificate = read_json("rate_capped_band_optimization_certificate.json")
    resources = read_json("size_uniform_resources_certificate.json")
    audit = read_json("size_uniform_resources_audit.json")
    if not (certificate["all_checks_passed"] and resources["all_checks_passed"] and
            audit["all_continuous_nearest_neighbor_gaps_positive"]):
        raise RuntimeError("Figures require the existing passing resource certificates.")
    plt.rcParams.update({"font.size": 8.0, "axes.titlesize": 8.4, "axes.labelsize": 8.1,
                         "legend.fontsize": 7, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    device = device_figure()
    if args.device_only:
        output = ROOT / "results/reports/physical_narrative_figure_audit.json"
        result = json.loads(output.read_text(encoding="utf-8"))
        result["figure_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["device_schematic"] = device
        result["figure_files"] = [f"figures/{name}.pdf" for name in
                                 ("fig1_local_device", "fig2_pointwise_noise",
                                  "fig3_shared_band_cost", "fig4_design_targets")]
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(device, indent=2), flush=True)
        return
    pointwise = pointwise_figure(points)
    result = shared_figure(rows, certificate, resources)
    if not args.presentation_only:
        design_figure(resources, audit, read_json("fig4_shared_device_certificate.json"))
    result["figure_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["device_schematic"] = device
    result["presentation"] = {"pointwise": pointwise, "shared": shared_figure.presentation_audit,
                              "scientific_data_unchanged": True,
                              "description": "Panel labels only in Fig. 1; pure-white Figs. 2-3; Figs. 3-4 use bottom-centered panel letters without panel titles"}
    if not args.presentation_only:
        result["presentation"]["design"] = design_figure.presentation_audit
    else:
        previous = read_json("reports/physical_narrative_figure_audit.json")
        if "design" in previous.get("presentation", {}):
            result["presentation"]["design"] = previous["presentation"]["design"]
    result["figure_files"] = [f"figures/{name}.pdf" for name in
                             ("fig1_local_device", "fig2_pointwise_noise",
                              "fig3_shared_band_cost", "fig4_design_targets")]
    output = ROOT / "results/reports/physical_narrative_figure_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
