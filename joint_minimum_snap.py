"""Joint planar knot/derivative QP for fixed durations and convex corridors."""
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np

from minimum_snap import (
    MinimumSnapTrajectory, _POSITION_TOL, _PROGRESS_BASIS, _PROGRESS_SAMPLES,
    _PROGRESS_WEIGHT, _SNAP_FACTOR, _VELOCITY_BASIS,
)

SNAP_DIFFERENCE_FACTOR = 7 * 6 * 5 * 4
DEFAULT_DEVIATION_WEIGHT = 100.  # s^-7
DEFAULT_TRUST_M = .18


@dataclass(frozen=True)
class JointResult:
    status: str
    solve_time_s: float
    trajectory: Optional[MinimumSnapTrajectory] = None
    waypoints: Optional[np.ndarray] = None
    offsets: Optional[np.ndarray] = None
    snap: Optional[float] = None
    deviation: Optional[float] = None
    max_residual: Optional[float] = None
    reason: str = ""
    free_indices: tuple = ()
    solver_status: str = ""


def physical_snap(trajectory):
    """Exact squared snap integral in m²/s⁷ over the supplied full trajectory."""
    fourth = np.diff(trajectory.control_points, n=4, axis=1)
    weighted = SNAP_DIFFERENCE_FACTOR * (_SNAP_FACTOR @ fourth)
    return float(np.sum(np.sum(weighted**2, axis=(1, 2)) / trajectory.durations**7))


def _control_map(points, times, velocity, acceleration, free, start_jerk):
    """Return controls C + M Z. Rows of Z each contain both planar components.

    Free positions are absolute in a translated local frame. Derivative
    rows represent physical derivatives multiplied by median(T)**order.
    """
    count = len(times)
    scale = float(np.median(times))
    knots = points.copy()
    knots[free] = 0.
    q_map = np.zeros((count+1, len(free) + 3*(count-1) + int(start_jerk is None)))
    q_map[free, np.arange(len(free))] = 1.
    controls = np.empty((count, 8, 2))
    controls[:, :4] = knots[:-1, None]
    controls[:, 4:] = knots[1:, None]
    mapping = np.empty((count, 8, q_map.shape[1]))
    mapping[:, :4] = q_map[:-1, None]
    mapping[:, 4:] = q_map[1:, None]
    first = times[0]*velocity/7.
    second = times[0]**2*acceleration/42.
    controls[0, 1:4] += (first, 2.*first+second, 3.*first+3.*second)
    if start_jerk is None:
        mapping[0, 3, -1] += (times[0]/scale)**3/210.
    else:
        controls[0, 3] += times[0]**3*start_jerk/210.
    for knot in range(1, count):
        col = len(free) + 3*(knot-1)
        for seg, rows, sign in ((knot, [1, 2, 3], 1.), (knot-1, [6, 5, 4], -1.)):
            ratio = times[seg]/scale
            v, a, j = sign*ratio/7., ratio**2/42., sign*ratio**3/210.
            mapping[seg, rows, col:col+3] += ((v, 0., 0.), (2.*v, a, 0.), (3.*v, 3.*a, j))
    return controls, mapping, knots, q_map, scale


def _least_squares(controls, mapping, knots, q_map, points, free, times, scale,
                   deviation_weight, regularize_progress, wall_velocity_weight):
    """Residuals use legacy normalized Snap units for optional penalties."""
    size = mapping.shape[-1]
    weight = (scale/times)**3.5
    common = [((weight[:, None, None] * (_SNAP_FACTOR @ np.diff(mapping, n=4, axis=1)))
               .reshape(-1, size),
               (weight[:, None, None] * (_SNAP_FACTOR @ np.diff(controls, n=4, axis=1)))
               .reshape(-1, 2))]
    if len(free) and deviation_weight:
        weight = np.sqrt(deviation_weight)*scale**3.5/SNAP_DIFFERENCE_FACTOR
        common.append((weight*q_map[free], -weight*points[free]))
    if regularize_progress:
        expected_map = q_map[:-1, None] + _PROGRESS_SAMPLES*np.diff(q_map, axis=0)[:, None]
        expected = knots[:-1, None] + _PROGRESS_SAMPLES*np.diff(knots, axis=0)[:, None]
        common.append((_PROGRESS_WEIGHT*(_PROGRESS_BASIS @ mapping-expected_map).reshape(-1, size),
                       _PROGRESS_WEIGHT*(_PROGRESS_BASIS @ controls-expected).reshape(-1, 2)))
    dx, dy = np.diff(points, axis=0).T
    along = (abs(dy) >= 2.5) & (abs(dx) <= .30*abs(dy))
    systems = []
    for axis in range(2):
        matrices = [matrix for matrix, _ in common]
        offsets = [offset[:, axis] for _, offset in common]
        if axis == 0 and wall_velocity_weight and np.any(along):
            weight = wall_velocity_weight*np.sqrt(scale/times[along])[:, None]
            mean_map = np.diff(q_map, axis=0)[along, None]
            mean = np.diff(knots[:, 0])[along, None]
            matrices.append((weight[..., None]*(_VELOCITY_BASIS @ np.diff(mapping[along], axis=1)
                                                  - mean_map)).reshape(-1, size))
            offsets.append((weight*((_VELOCITY_BASIS @ np.diff(controls[along, :, :1], axis=1))[..., 0]
                                    - mean)).ravel())
        systems.append((np.vstack(matrices), np.concatenate(offsets)))
    return systems


def _osqp_backend():
    import osqp
    from scipy.sparse import csc_matrix, eye

    return osqp, csc_matrix, eye


def warm_joint_backend():
    """Load native libraries before the first online planning deadline."""
    try:
        _osqp_backend()
    except ImportError:
        return False
    return True


def solve_joint_waypoint_minimum_snap(points, durations, start_velocity, start_acceleration,
                                      corridors, *, free_mask=None, trust_radius=DEFAULT_TRUST_M,
                                      waypoint_regions=None, deviation_weight=DEFAULT_DEVIATION_WEIGHT,
                                      start_jerk=None, regularize_progress=False,
                                      wall_velocity_weight=0., max_iter=4000):
    """Solve for permitted internal positions and shared derivatives up to jerk.

    Optional regions map free knot indices to world-frame (A, b) half-spaces.
    The start and finish are always fixed. A zero trust radius fixes the free
    knots through equality bounds. Only independently checked solutions return
    a trajectory. The caller certifies the physical scene separately.
    """
    started = perf_counter()

    def finish(status, **kwargs):
        return JointResult(status, perf_counter()-started, **kwargs)

    points, times = np.asarray(points, dtype=float), np.asarray(durations, dtype=float)
    v0, a0 = np.asarray(start_velocity, dtype=float), np.asarray(start_acceleration, dtype=float)
    j0 = None if start_jerk is None else np.asarray(start_jerk, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or times.shape != (len(points)-1,) or len(corridors) != len(times)
            or v0.shape != (2,) or a0.shape != (2,) or (j0 is not None and j0.shape != (2,))):
        raise ValueError("expected planar knots, boundary derivatives and one corridor/time per segment")
    mask = np.zeros(len(points), dtype=bool) if free_mask is None else np.array(free_mask, dtype=bool)
    if mask.shape != (len(points),) or mask[0] or mask[-1]:
        raise ValueError("free_mask must exclude window endpoints")
    free = np.flatnonzero(mask)
    region_specs = {} if waypoint_regions is None else waypoint_regions
    if set(region_specs)-set(free):
        raise ValueError("custom regions require a free knot")
    radii = None if trust_radius is None else np.broadcast_to(np.asarray(trust_radius, dtype=float), (2,))
    values = (points, times, v0, a0, [deviation_weight, wall_velocity_weight])
    if (any(not np.isfinite(value).all() for value in values)
            or (j0 is not None and not np.isfinite(j0).all())
            or (radii is not None and (not np.isfinite(radii).all() or np.any(radii < 0)))
            or np.any(times <= 0.) or deviation_weight < 0. or wall_velocity_weight < 0. or max_iter < 1):
        return finish("invalid_input", reason="nonfinite input or invalid solver settings")
    # Refuse ranges where time powers and endpoint subtraction are unreliable.
    if np.min(times) < 1e-4 or np.max(times) > 1e4:
        return finish("ill_conditioned", reason="segment duration outside supported numerical range")

    def normalized(halfspace):
        a, b = map(lambda x: np.asarray(x, dtype=float), halfspace)
        if a.ndim != 2 or a.shape[1] != 2 or b.shape != (len(a),):
            raise ValueError("half-spaces must be A[m, 2], b[m]")
        lengths = np.linalg.norm(a, axis=1)
        if np.any(lengths <= 0.) or not all(np.isfinite(x).all() for x in (lengths, a, b)):
            raise ValueError("half-spaces require finite, nonzero normals")
        return a/lengths[:, None], b/lengths

    corridors = [normalized(c) for c in corridors]
    regions = {i: [normalized(region_specs[i])] if i in region_specs else [] for i in free}
    for i in free:
        if radii is not None:
            axes = np.vstack((np.eye(2), -np.eye(2)))
            regions[i].append((axes, axes @ points[i] + np.tile(radii, 2)))

    origin = points[0]
    controls, mapping, knots, q_map, scale = _control_map(points-origin, times, v0, a0, free, j0)
    n = mapping.shape[-1]
    info = dict(free_indices=tuple(map(int, free)))
    linear, bounds = [], []
    for control, matrix, (a, b) in zip(controls, mapping, corridors):
        linear.append(np.einsum("kn,md->kmnd", matrix, a).reshape(8*len(a), 2*n))
        bounds.append((b-a @ origin-control @ a.T).ravel())
    for i, constraints in regions.items():
        for a, b in constraints:
            linear.append(np.einsum("n,md->mnd", q_map[i], a).reshape(len(a), 2*n))
            bounds.append(b-a @ origin)
    linear, bounds = np.concatenate(linear), np.concatenate(bounds)

    # Center and whiten the least-squares objective. For each axis,
    # Z = center + lift y, yielding P=I and q=0 in OSQP's 1/2 convention.
    # Both axes still share the full set of oblique half-space rows.
    center = np.zeros((n, 2))
    lift = np.zeros((2*n, 2*n))
    try:
        if n:
            systems = _least_squares(controls, mapping, knots, q_map, points-origin, free, times,
                                     scale, deviation_weight, regularize_progress, wall_velocity_weight)
            for axis, (system, known) in enumerate(systems):
                col = np.linalg.norm(system, axis=0)
                if not np.isfinite(col).all() or np.any(col <= 0.):
                    return finish("ill_conditioned", reason="invalid objective column", **info)
                u, singular, vh = np.linalg.svd(system/col, full_matrices=False)
                if singular[-1] <= 1e-12*singular[0]:
                    return finish("ill_conditioned", reason="near-singular scaled objective", **info)
                axis_lift = (vh.T/singular)/col[:, None]
                lift[axis::2, axis::2] = axis_lift
                center[:, axis] = -axis_lift @ (u.T @ known)
    except np.linalg.LinAlgError as error:
        return finish("ill_conditioned", reason=str(error), **info)

    lengths = np.linalg.norm(linear, axis=1)
    variable = lengths > 0.
    fixed_residual = float(np.max(-bounds[~variable], initial=0.))
    if fixed_residual > _POSITION_TOL:
        return finish("infeasible", reason="fixed control point outside corridor",
                      max_residual=fixed_residual, **info)
    transformed = linear[variable] @ lift
    upper = bounds[variable]-linear[variable] @ center.ravel()
    solver_status = "unconstrained optimum"
    variables = center
    if not np.isfinite(upper).all() or not np.isfinite(transformed).all():
        return finish("ill_conditioned", reason="nonfinite transformed constraints", **info)
    if np.any(upper < -1e-11):
        lengths = np.linalg.norm(transformed, axis=1)
        if np.any(lengths <= 1e-15):
            return finish("ill_conditioned", reason="near-zero transformed constraint", **info)
        try:
            osqp, csc_matrix, eye = _osqp_backend()
        except ImportError as error:
            return finish("dependency_unavailable", reason=str(error), **info)
        solver = osqp.OSQP()
        try:
            solver.setup(P=eye(2*n, format="csc"), q=np.zeros(2*n),
                         A=csc_matrix(transformed/lengths[:, None]),
                         l=np.full(len(upper), -np.inf), u=upper/lengths,
                         verbose=False, polishing=True, eps_abs=1e-9, eps_rel=1e-9,
                         max_iter=max_iter, check_termination=min(max_iter, 25))
            solution = solver.solve(raise_error=False)
        except (ValueError, RuntimeError, osqp.OSQPException) as error:
            return finish("solver_error", reason=str(error), **info)
        solver_status = solution.info.status
        code = solution.info.status_val
        if code != 1 or solution.x is None or not np.isfinite(solution.x).all():
            status = "infeasible" if code in (3, 4) else "iteration_limit" if code == 7 else "solver_failed"
            return finish(status, reason=solver_status, solver_status=solver_status, **info)
        variables = center + (lift @ solution.x).reshape(n, 2)

    candidate = controls + mapping @ variables + origin
    for index in np.flatnonzero(~mask):
        if index < len(times):
            candidate[index, 0] = points[index]
        if index > 0:
            candidate[index-1, -1] = points[index]
    candidate[-1, 4:] = points[-1]
    optimized = np.vstack((candidate[0, 0], candidate[:, -1]))
    residual = max(
        [float(np.max(c @ a.T-b, initial=0.)) for c, (a, b) in zip(candidate, corridors)]
        + [float(np.max(a @ optimized[i]-b, initial=0.)) for i, constraints in regions.items()
           for a, b in constraints],
    )
    if not np.isfinite(candidate).all() or not np.isfinite(residual) or residual > _POSITION_TOL:
        return finish("residual_failed", reason="independent control or waypoint check",
                      max_residual=residual if np.isfinite(residual) else None,
                      solver_status=solver_status, **info)
    trajectory = MinimumSnapTrajectory(candidate, times.copy())
    # Decode physical endpoint derivatives from independently differenced
    # control polygons and compare against the variables that built them.
    expected = np.zeros((len(points), 4, 2))
    expected[:, 0] = points
    expected[free, 0] = origin + variables[:len(free)]
    expected[0, 1], expected[0, 2] = v0, a0
    expected[0, 3] = variables[-1]/scale**3 if j0 is None else j0
    for order in (1, 2, 3):
        expected[1:-1, order] = variables[len(free)+order-1:len(free)+3*(len(times)-1):3]/scale**order
    for order, polygon in enumerate(trajectory._derivatives):
        tolerance = 1e-8 + (128.*np.finfo(float).eps*np.max(abs(candidate), initial=1.)
                            * 7.**order / min(times)**order)
        if not (np.allclose(polygon[:, 0], expected[:-1, order], atol=tolerance, rtol=1e-8)
                and np.allclose(polygon[:, -1], expected[1:, order], atol=tolerance, rtol=1e-8)):
            return finish("residual_failed", reason="decoded endpoint derivative mismatch",
                          max_residual=residual, solver_status=solver_status, **info)
    offsets = optimized[free]-points[free]
    snap = physical_snap(trajectory)
    deviation = float(np.sum(offsets**2))
    if not np.isfinite(snap) or not np.isfinite(deviation):
        return finish("ill_conditioned", reason="nonfinite physical objective", **info)
    return finish("solved", trajectory=trajectory, waypoints=optimized, offsets=offsets, snap=snap,
                  deviation=deviation, max_residual=residual, solver_status=solver_status, **info)
