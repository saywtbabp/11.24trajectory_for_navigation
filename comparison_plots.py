"""Render paired MuJoCo comparisons from archived run data."""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from door_geometry import Leaf
from flight_corridors import _clip_polygon, _obstacles
from joint_minimum_snap import DEFAULT_DEVIATION_WEIGHT, solve_joint_waypoint_minimum_snap
from minimum_snap import MinimumSnapTrajectory
from scene import WALL_HALF, make_scene


COLORS = {"baseline": "#d76d19", "joint": "#136aac"}


def _read(folder, name):
    return json.loads((folder/(name+".json")).read_text(encoding="utf-8"))


def _wall_map(ax, arena):
    for x0, y0, x1, y1 in arena.walls:
        x, y = min(x0, x1)-WALL_HALF, min(y0, y1)-WALL_HALF
        ax.add_patch(Rectangle((x, y), abs(x1-x0)+2*WALL_HALF,
                               abs(y1-y0)+2*WALL_HALF, color="#444b55", alpha=.7))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=.2)


def _series(frames):
    time = np.asarray([row["t"] for row in frames])
    actual = np.asarray([row["actual"] for row in frames])
    reference = np.array([row.get("reference", [[np.nan, np.nan]]*4) for row in frames])
    return time, actual, reference


def _overview(folder, arena, records):
    fig, ax = plt.subplots(figsize=(16, 4.2), layout="constrained")
    _wall_map(ax, arena)
    initial = np.array(records["baseline"]["astar"][0]["points"])
    ax.plot(*initial.T, color="#888888", linestyle="--", marker=".", markersize=2,
            lw=1., label="Initial A* geometry")
    for solver, record in records.items():
        _, actual, reference = _series(record["frames"])
        ax.plot(*actual.T, color=COLORS[solver], alpha=.45, lw=1.2, label=solver+" truth")
        ax.plot(*reference[:, 0].T, color=COLORS[solver], alpha=.9, lw=.8, ls=":", label=solver+" reference")
    for gate in arena.gates:
        ax.text(gate.x, arena.bounds[3]+.35, gate.name, ha="center")
    ax.set_xlim(arena.bounds[0], arena.bounds[2])
    ax.set_ylim(arena.bounds[1]-.2, arena.bounds[3]+.95)
    ax.legend(loc="upper center", ncols=5, fontsize=8)
    fig.savefig(folder/"mission_overlay.png", dpi=170)
    plt.close(fig)


def _time_series(folder, records):
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True, layout="constrained")
    names = ("Speed (m/s)", "Acceleration (m/s²)", "Jerk (m/s³)", "Curvature (1/m)")
    for solver, record in records.items():
        time, _, reference = _series(record["frames"])
        _, velocity, accel, jerk = reference.transpose(1, 0, 2)
        speed = np.linalg.norm(velocity, axis=1)
        cross = abs(velocity[:, 0]*accel[:, 1]-velocity[:, 1]*accel[:, 0])
        curvature = np.where(speed > .30, cross/np.maximum(speed, .30)**3, np.nan)
        values = (speed, np.linalg.norm(accel, axis=1), np.linalg.norm(jerk, axis=1), curvature)
        for ax, name, value in zip(axes, names, values):
            ax.plot(time, value, color=COLORS[solver], lw=.65, alpha=.85, label=solver)
            ax.set_ylabel(name)
            ax.grid(alpha=.25)
    axes[0].legend(ncols=2)
    axes[-1].set_xlabel("Simulation time (s) · curvature shown for speed > 0.30 m/s")
    fig.savefig(folder/"reference_dynamics.png", dpi=150)
    plt.close(fig)


def _preview(plan):
    curve = MinimumSnapTrajectory(np.array(plan["controls"]), np.array(plan["durations"]))
    return np.array([curve.sample(t)[0] for t in np.linspace(0., curve.duration, 150)])


def _movement(plan):
    return float(np.linalg.norm(np.array(plan["optimized"])-np.array(plan["original"]), axis=1).max())


def _local_geometry(folder, arena, records):
    plans = [row for row in records["joint"]["plans"] if not row["braking"]]
    by_action = {}
    for plan in plans:
        key = (plan["door"], plan["phase"])
        if key not in by_action or _movement(plan) > _movement(by_action[key]):
            by_action[key] = plan
    priority = sorted(by_action.values(), key=_movement, reverse=True)
    chosen = []
    for phase in ("bypass", "approach", "release", "cross"):
        plan = next((item for item in priority if item["phase"] == phase), None)
        if plan is not None:
            chosen.append(plan)
    chosen += [p for p in priority if all(p is not item for item in chosen)][:max(0, 4-len(chosen))]
    if not chosen:
        return

    xmin, ymin, xmax, ymax = arena.bounds
    bounds = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for ax, joint in zip(axes.flat, chosen):
        _wall_map(ax, arena)
        for index, (a, b) in enumerate(joint["corridors"]):
            vertices = bounds.copy()
            for normal, offset in zip(a, b):
                vertices = _clip_polygon(vertices, normal, offset)
                if len(vertices) == 0:
                    break
            if len(vertices) >= 3:
                ax.add_patch(Polygon(vertices, facecolor="#69b588", edgecolor="#429466",
                                     alpha=.23, label="Joint corridor" if index == 0 else None))
        gates = {gate.name: gate for gate in arena.gates}
        leaves = tuple(Leaf(gates[row["gate"]], row["side"], row["angle"]) for row in joint["leaves"])
        if leaves:
            geometry = _obstacles(arena, leaves)
            corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
            for i in range(len(geometry.centers)-len(leaves), len(geometry.centers)):
                polygon = geometry.centers[i] + (corners*geometry.half_extents[i]) @ geometry.axes[i]
                ax.add_patch(Polygon(polygon, facecolor="#715771", alpha=.6))
        original, updated = np.array(joint["original"]), np.array(joint["optimized"])
        raw = next((entry for entry in reversed(records["joint"]["astar"]) if entry["t"] <= joint["t"]), None)
        if raw is not None:
            ax.plot(*np.array(raw["points"]).T, ls="--", marker=".", color="#888888",
                    lw=1., label="A* waypoints at last search")
        base = [p for p in records["baseline"]["plans"]
                if p["door"] == joint["door"] and p["phase"] == joint["phase"] and not p["braking"]]
        if base:
            best = min(base, key=lambda p: np.linalg.norm(np.array(p["original"])[[0, -1]]
                                                         -original[[0, -1]]))
            ax.plot(*_preview(best).T, color=COLORS["baseline"], lw=1.5, label="Baseline fitted")
        ax.plot(*original.T, marker="o", ls=":", lw=1.2, color=COLORS["baseline"], label="Heuristic knots")
        ax.plot(*_preview(joint).T, color=COLORS["joint"], lw=1.4, label="Joint-mode fitted")
        ax.scatter(*updated.T, color=COLORS["joint"], s=16, label="Optimized knots")
        for start, end in zip(original, updated):
            if np.linalg.norm(start-end) > 1e-5:
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", lw=1.8, color=COLORS["joint"]))
        lower, upper = original.min(axis=0)-.65, original.max(axis=0)+.65
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_title(f'{joint["door"] or ""} {joint["phase"]} · {joint["solver"]} · max move {_movement(joint):.3f} m')
        ax.legend(fontsize=7, loc="best")
    for ax in axes.flat[len(chosen):]:
        ax.set_visible(False)
    fig.savefig(folder/"local_corridors_and_knots.png", dpi=160)
    plt.close(fig)


def _fixed_time_audit(records):
    """Compare the inner objective on identical QPs for accepted joint windows."""
    results = []
    for plan in records["joint"]["plans"]:
        indices = plan["free_indices"]
        if not indices:
            continue
        points = np.asarray(plan["original"])
        durations = np.asarray(plan["durations"])
        accepted = MinimumSnapTrajectory(np.asarray(plan["controls"]), durations)
        _, velocity, acceleration, _ = accepted.sample(0.)
        free = np.zeros(len(points), dtype=bool)
        free[indices] = True
        kwargs = dict(points=points, durations=durations, start_velocity=velocity,
                      start_acceleration=acceleration, corridors=plan["corridors"])
        fixed = solve_joint_waypoint_minimum_snap(**kwargs)
        joint = solve_joint_waypoint_minimum_snap(**kwargs, free_mask=free)
        row = dict(t=plan["t"], door=plan["door"], phase=plan["phase"],
                   fixed_status=fixed.status, joint_status=joint.status)
        if fixed.trajectory is not None and joint.trajectory is not None:
            row.update(fixed_snap=fixed.snap, joint_snap=joint.snap, joint_deviation=joint.deviation,
                       joint_objective=joint.snap+DEFAULT_DEVIATION_WEIGHT*joint.deviation)
        results.append(row)
    return results


def _metric_row(scenario, solver, result):
    qp = result["trajectory"]["qp"]
    quality = result["trajectory_quality"]
    offline = result["comparison"]
    return dict(scenario=scenario, solver=solver, status=result["reason"], seconds=result["seconds"],
                distance_m=result["distance_m"], snap_m2_s7=offline["executed_snap_m2_s7"],
                jerk_rms=offline.get("jerk_rms"), jerk_p95=offline.get("jerk_p95"),
                jerk_peak=offline.get("jerk_peak"), curvature_peak=offline.get("curvature_peak"),
                tracking_p95_m=quality.get("truth_tracking_p95_m"),
                clearance_min_m=result["free_flight_min_clearance_m"],
                collision_episodes=offline["collision_episodes"],
                wall_contact_steps=result["wall_contact_steps"],
                passage_door_contact_steps=result["passage_door_contact_steps"],
                qp_p50_ms=qp.get("p50_ms"), qp_p95_ms=qp.get("p95_ms"), qp_max_ms=qp.get("max_ms"),
                qp_attempts=qp["attempts"], qp_success_rate=qp["successes"]/qp["attempts"] if qp["attempts"] else None,
                fallbacks=qp["fallbacks"], solve_failures=result["trajectory"]["solve_failures"],
                scan_p95_ms=quality["planning_scan_p95_ms"], scan_over_period=quality["planning_scans_over_period"],
                moved_knots=offline["moved_knots"], max_waypoint_move_m=offline["max_waypoint_move_m"],
                mean_waypoint_move_m=offline["mean_waypoint_move_m"])


def _csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_comparison(output, scenarios):
    arena, _, _ = make_scene()
    metrics, local = [], []
    for scenario in scenarios:
        folder = output/scenario
        records = {}
        for solver in COLORS:
            run = folder/solver
            if not (run/"result.json").is_file():
                continue
            records[solver] = {name: _read(run, name) for name in ("result", "frames", "plans", "astar")}
            result = records[solver]["result"]
            metrics.append(_metric_row(scenario, solver, result))
            for gate, values in result["trajectory_quality"]["bypass"].items():
                local.append(dict(scenario=scenario, solver=solver, gate=gate, **values))
        if len(records) == 2:
            _overview(folder, arena, records)
            _time_series(folder, records)
            _local_geometry(folder, arena, records)
            (folder/"fixed_time_audit.json").write_text(
                json.dumps(_fixed_time_audit(records), indent=2), encoding="utf-8")
    _csv(output/"metrics.csv", metrics)
    _csv(output/"bypass_metrics.csv", local)
