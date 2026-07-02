from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bgs.sampling import sample_on_vgss
from .random_walk import get_system_bounds, sample_point_on_maximal_cube


# ==========================================
# SHARED VISUALIZATION HELPERS
# ==========================================


def _run_visual_walk(conductors, master_vgss, rng, max_hops=100):
    """Run one small FRW example used only by the visualizations."""

    sys_center, sys_size = get_system_bounds(conductors)
    visual_kill_radius = sys_size * 2.0

    path_points = []
    maximal_cubes = []
    terminal = "Maximum hops reached"

    current_p = sample_on_vgss(master_vgss.sampling_context, rng)
    path_points.append(current_p.copy())

    print("\n[VISUALIZATION WALK LOG]")

    for hop in range(max_hops):
        if np.max(np.abs(current_p - sys_center)) > visual_kill_radius:
            terminal = "Infinity (Ground)"
            print("TERMINATED: Escaped to Infinity (Ground)")
            break

        distances = [c.distance_to_point(current_p) for c in conductors]
        nearest_idx = int(np.argmin(distances))
        min_dist = float(distances[nearest_idx])
        maximal_cubes.append((current_p.copy(), min_dist))

        print(
            f"Hop {hop + 1}: Center = "
            f"[{current_p[0]:.4f}, {current_p[1]:.4f}, {current_p[2]:.4f}], "
            f"a = {2 * min_dist:.4f}"
        )

        if min_dist < 1e-4:
            terminal = conductors[nearest_idx].name
            print(f"TERMINATED ON: {terminal}")
            break

        current_p = sample_point_on_maximal_cube(current_p, min_dist, rng)
        path_points.append(current_p.copy())

    return np.asarray(path_points, dtype=float), maximal_cubes, terminal


def _normalize_vgss_collection(master_vgss, all_net_vgss=None):
    collection = list(all_net_vgss) if all_net_vgss is not None else [master_vgss]
    master_name = master_vgss.net.name

    if not any(net_vgss.net.name == master_name for net_vgss in collection):
        collection.insert(0, master_vgss)

    return collection


def _net_colors(net_vgss_collection):
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0"]
    return {
        net_vgss.net.name: colors[index % len(colors)]
        for index, net_vgss in enumerate(net_vgss_collection)
    }


def _box_faces(min_bounds, max_bounds):
    x0, y0, z0 = np.asarray(min_bounds, dtype=float)
    x1, y1, z1 = np.asarray(max_bounds, dtype=float)

    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )

    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
    ]


def draw_cuboid(
    ax,
    min_bounds,
    max_bounds,
    color,
    alpha=0.5,
    wireframe=False,
    linewidth=1.0,
    linestyle="-",
    face_alpha=None,
    edge_alpha=None,
):
    """Draw a cuboid and return its Poly3DCollection."""

    faces = _box_faces(min_bounds, max_bounds)

    if wireframe:
        facecolor = to_rgba(color, 0.02 if face_alpha is None else face_alpha)
        edgecolor = to_rgba(color, 0.95 if edge_alpha is None else edge_alpha)
    else:
        facecolor = to_rgba(color, alpha if face_alpha is None else face_alpha)
        edgecolor = to_rgba("black", 0.75 if edge_alpha is None else edge_alpha)

    collection = Poly3DCollection(
        faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        linestyles=linestyle,
    )
    ax.add_collection3d(collection)
    return collection


def _set_equal_3d_axes(ax, bound_pairs, path_points):
    points = []
    for min_bounds, max_bounds in bound_pairs:
        points.extend([np.asarray(min_bounds, dtype=float), np.asarray(max_bounds, dtype=float)])

    if len(path_points):
        points.extend(path_points)

    point_array = np.asarray(points, dtype=float)
    minimum = np.min(point_array, axis=0)
    maximum = np.max(point_array, axis=0)
    middle = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.5) * 1.08

    ax.set_xlim(middle[0] - radius, middle[0] + radius)
    ax.set_ylim(middle[1] - radius, middle[1] + radius)
    ax.set_zlim(middle[2] - radius, middle[2] + radius)

    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _set_2d_limits(ax, first_axis, second_axis, bound_pairs, path_points):
    projected = []
    for min_bounds, max_bounds in bound_pairs:
        projected.extend(
            [
                [min_bounds[first_axis], min_bounds[second_axis]],
                [max_bounds[first_axis], max_bounds[second_axis]],
            ]
        )

    if len(path_points):
        projected.extend(path_points[:, [first_axis, second_axis]])

    projected = np.asarray(projected, dtype=float)
    minimum = np.min(projected, axis=0)
    maximum = np.max(projected, axis=0)
    span = maximum - minimum
    margin = np.maximum(span * 0.08, 0.25)

    ax.set_xlim(minimum[0] - margin[0], maximum[0] + margin[0])
    ax.set_ylim(minimum[1] - margin[1], maximum[1] + margin[1])
    ax.set_aspect("equal", adjustable="box")


def _save_or_show(figure, save_path=None, show=True):
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)


# ==========================================
# 2D VISUALIZATION
# ==========================================


def visualize_2d_walk(
    conductors,
    master_vgss,
    rng,
    all_net_vgss=None,
    *,
    max_hops=100,
    save_path=None,
    show=True,
):
    """Plot one visual FRW walk in the XY, XZ, and YZ projections."""

    path_points, maximal_cubes, terminal = _run_visual_walk(
        conductors, master_vgss, rng, max_hops=max_hops
    )
    net_vgss_collection = _normalize_vgss_collection(master_vgss, all_net_vgss)
    net_colors = _net_colors(net_vgss_collection)
    master_net_name = master_vgss.net.name
    master_conductors = set(master_vgss.net.conductor_names)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    projections = [
        (axes[0], 0, 1, "Top View (XY Plane)", "X", "Y"),
        (axes[1], 0, 2, "Front View (XZ Plane)", "X", "Z"),
        (axes[2], 1, 2, "Side View (YZ Plane)", "Y", "Z"),
    ]

    master_surface_patches = []
    other_surface_patches = []
    cube_patches = []
    path_artists = []
    hop_labels = []
    bound_pairs = [(c.min_bounds, c.max_bounds) for c in conductors]

    for ax, first, second, title, xlabel, ylabel in projections:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.45)

        for conductor in conductors:
            is_master = conductor.name in master_conductors
            color = "tab:red" if is_master else "0.70"
            lower = conductor.min_bounds[[first, second]]
            size = conductor.max_bounds[[first, second]] - lower
            rect = patches.Rectangle(
                lower,
                size[0],
                size[1],
                edgecolor="0.15",
                facecolor=color,
                linewidth=1.4 if is_master else 0.8,
                alpha=0.55 if is_master else 0.34,
                zorder=2,
            )
            ax.add_patch(rect)

        for net_vgss in net_vgss_collection:
            is_master_net = net_vgss.net.name == master_net_name
            color = net_colors[net_vgss.net.name]
            for surface in net_vgss.surfaces:
                lower = surface.min_bounds[[first, second]]
                size = surface.max_bounds[[first, second]] - lower
                rect = patches.Rectangle(
                    lower,
                    size[0],
                    size[1],
                    edgecolor=color,
                    facecolor="none",
                    linestyle="--",
                    linewidth=2.0 if is_master_net else 1.1,
                    alpha=0.95 if is_master_net else 0.45,
                    zorder=3,
                )
                ax.add_patch(rect)
                if is_master_net:
                    master_surface_patches.append(rect)
                else:
                    other_surface_patches.append(rect)
                bound_pairs.append((surface.min_bounds, surface.max_bounds))

        for center, radius in maximal_cubes:
            lower = center[[first, second]] - radius
            rect = patches.Rectangle(
                lower,
                2 * radius,
                2 * radius,
                edgecolor="tab:purple",
                facecolor=to_rgba("tab:purple", 0.06),
                linewidth=1.0,
                zorder=3,
                visible=False,
            )
            ax.add_patch(rect)
            cube_patches.append(rect)

        line = ax.plot(
            path_points[:, first],
            path_points[:, second],
            color="0.08",
            marker="o",
            markersize=4,
            linewidth=1.7,
            zorder=5,
        )[0]
        start = ax.scatter(
            path_points[0, first],
            path_points[0, second],
            marker="*",
            s=150,
            color="tab:green",
            edgecolor="black",
            linewidth=0.7,
            zorder=7,
        )
        end = ax.scatter(
            path_points[-1, first],
            path_points[-1, second],
            marker="X",
            s=100,
            color="crimson",
            edgecolor="black",
            linewidth=0.6,
            zorder=7,
        )
        path_artists.extend([line, start, end])

        for hop_index, point in enumerate(path_points):
            label = ax.annotate(
                str(hop_index),
                (point[first], point[second]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="0.2",
                visible=False,
                zorder=8,
            )
            hop_labels.append(label)

        _set_2d_limits(ax, first, second, bound_pairs, path_points)

    figure.suptitle(
        f"FRW example from {master_net_name} — terminal: {terminal}",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(bottom=0.22, top=0.86, wspace=0.26)

    checkbox_ax = figure.add_axes([0.36, 0.035, 0.28, 0.12])
    checkbox = CheckButtons(
        checkbox_ax,
        ["Master VGSS", "Other VGSS", "Maximal Squares", "Walk Path", "Hop Numbers"],
        [True, True, False, True, False],
    )

    artist_groups = {
        "Master VGSS": master_surface_patches,
        "Other VGSS": other_surface_patches,
        "Maximal Squares": cube_patches,
        "Walk Path": path_artists,
        "Hop Numbers": hop_labels,
    }

    def toggle(label):
        artists = artist_groups[label]
        new_visibility = not artists[0].get_visible() if artists else False
        for artist in artists:
            artist.set_visible(new_visibility)
        figure.canvas.draw_idle()

    checkbox.on_clicked(toggle)
    figure._frw_checkbuttons = checkbox

    _save_or_show(figure, save_path=save_path, show=show)
    return figure


# ==========================================
# 3D VISUALIZATION
# ==========================================


def visualize_3d_walk(
    conductors,
    master_vgss,
    rng,
    all_net_vgss=None,
    *,
    max_hops=100,
    save_path=None,
    show=True,
):
    """
    Plot one representative FRW path together with every generated VGSS.

    The selected master net is emphasized. Other nets remain visible with
    lighter Gaussian surfaces so the walk can be understood in full geometry.
    """

    path_points, maximal_cubes, terminal = _run_visual_walk(
        conductors, master_vgss, rng, max_hops=max_hops
    )
    net_vgss_collection = _normalize_vgss_collection(master_vgss, all_net_vgss)
    net_colors = _net_colors(net_vgss_collection)
    master_net_name = master_vgss.net.name
    master_conductors = set(master_vgss.net.conductor_names)

    figure = plt.figure(figsize=(12, 9))
    ax = figure.add_subplot(111, projection="3d")

    conductor_collections = []
    conductor_labels = []
    master_surface_collections = []
    other_surface_collections = []
    cube_collections = []
    hop_labels = []
    bound_pairs = []

    for conductor in conductors:
        is_master = conductor.name in master_conductors
        collection = draw_cuboid(
            ax,
            conductor.min_bounds,
            conductor.max_bounds,
            color="tab:red" if is_master else "0.68",
            alpha=0.48 if is_master else 0.24,
            linewidth=1.5 if is_master else 0.8,
        )
        conductor_collections.append(collection)
        bound_pairs.append((conductor.min_bounds, conductor.max_bounds))

        center = (conductor.min_bounds + conductor.max_bounds) / 2.0
        label = ax.text(
            *center,
            conductor.name,
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold" if is_master else "normal",
        )
        conductor_labels.append(label)

    legend_handles = [
        patches.Patch(
            facecolor=to_rgba("tab:red", 0.48),
            edgecolor="black",
            label=f"Master conductors: {', '.join(master_vgss.net.conductor_names)}",
        ),
        patches.Patch(
            facecolor=to_rgba("0.68", 0.24),
            edgecolor="black",
            label="Other conductors",
        ),
    ]

    for net_vgss in net_vgss_collection:
        is_master_net = net_vgss.net.name == master_net_name
        color = net_colors[net_vgss.net.name]

        for surface in net_vgss.surfaces:
            collection = draw_cuboid(
                ax,
                surface.min_bounds,
                surface.max_bounds,
                color=color,
                wireframe=True,
                linewidth=2.2 if is_master_net else 1.1,
                linestyle="--",
                face_alpha=0.025 if is_master_net else 0.008,
                edge_alpha=0.95 if is_master_net else 0.38,
            )
            if is_master_net:
                master_surface_collections.append(collection)
            else:
                other_surface_collections.append(collection)
            bound_pairs.append((surface.min_bounds, surface.max_bounds))

        conductor_names = ", ".join(net_vgss.net.conductor_names)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                linewidth=2.2 if is_master_net else 1.2,
                alpha=1.0 if is_master_net else 0.55,
                label=(
                    f"{net_vgss.net.name}: {conductor_names}"
                    + (" (master)" if is_master_net else "")
                ),
            )
        )

    for center, radius in maximal_cubes:
        cube_min = center - radius
        cube_max = center + radius
        collection = draw_cuboid(
            ax,
            cube_min,
            cube_max,
            color="tab:purple",
            wireframe=True,
            linewidth=0.9,
            face_alpha=0.0,
            edge_alpha=0.55,
        )
        collection.set_visible(False)
        cube_collections.append(collection)

    path_line = ax.plot(
        path_points[:, 0],
        path_points[:, 1],
        path_points[:, 2],
        color="0.08",
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="Walk path",
        zorder=8,
    )[0]
    start_marker = ax.scatter(
        path_points[0, 0],
        path_points[0, 1],
        path_points[0, 2],
        color="tab:green",
        edgecolor="black",
        linewidth=0.7,
        s=170,
        marker="*",
        depthshade=False,
        label="VGSS start",
        zorder=10,
    )
    terminal_marker = ax.scatter(
        path_points[-1, 0],
        path_points[-1, 1],
        path_points[-1, 2],
        color="crimson",
        edgecolor="black",
        linewidth=0.7,
        s=115,
        marker="X",
        depthshade=False,
        label=f"Terminal: {terminal}",
        zorder=10,
    )
    path_artists = [path_line, start_marker, terminal_marker]

    for hop_index, point in enumerate(path_points):
        label = ax.text(
            point[0],
            point[1],
            point[2],
            f" {hop_index}",
            fontsize=7,
            color="0.15",
            visible=False,
            zorder=11,
        )
        hop_labels.append(label)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"3D Floating Random Walk — {master_net_name}\n"
        f"{len(maximal_cubes)} hops, terminal: {terminal}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.35)
    _set_equal_3d_axes(ax, bound_pairs, path_points)

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="0.08",
                marker="o",
                markersize=4,
                linewidth=2,
                label="Walk path",
            ),
            Line2D(
                [0],
                [0],
                color="tab:green",
                marker="*",
                markeredgecolor="black",
                linestyle="None",
                markersize=12,
                label="VGSS start",
            ),
            Line2D(
                [0],
                [0],
                color="crimson",
                marker="X",
                markeredgecolor="black",
                linestyle="None",
                markersize=9,
                label=f"Terminal: {terminal}",
            ),
        ]
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    info_text = (
        f"Master net: {master_net_name}\n"
        f"Start: [{path_points[0, 0]:.3f}, {path_points[0, 1]:.3f}, {path_points[0, 2]:.3f}]\n"
        f"Hops: {len(maximal_cubes)}\n"
        f"Result: {terminal}"
    )
    ax.text2D(
        0.73,
        0.03,
        info_text,
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.82},
    )

    figure.subplots_adjust(left=0.05, right=0.83, bottom=0.08, top=0.92)
    checkbox_ax = figure.add_axes([0.84, 0.30, 0.15, 0.22])
    checkbox = CheckButtons(
        checkbox_ax,
        [
            "Master VGSS",
            "Other VGSS",
            "Maximal Cubes",
            "Walk Path",
            "Hop Numbers",
            "Labels",
        ],
        [True, True, False, True, False, True],
    )

    artist_groups = {
        "Master VGSS": master_surface_collections,
        "Other VGSS": other_surface_collections,
        "Maximal Cubes": cube_collections,
        "Walk Path": path_artists,
        "Hop Numbers": hop_labels,
        "Labels": conductor_labels,
    }

    def toggle(label):
        artists = artist_groups[label]
        new_visibility = not artists[0].get_visible() if artists else False
        for artist in artists:
            artist.set_visible(new_visibility)
        figure.canvas.draw_idle()

    checkbox.on_clicked(toggle)
    figure._frw_checkbuttons = checkbox

    _save_or_show(figure, save_path=save_path, show=show)
    return figure
