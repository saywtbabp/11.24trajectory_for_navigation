"""Rolling, corridor-certified references for the free-flight actions."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from flight_corridors import FLIGHT_MARGIN, build_corridors, certify_controls, sampled_clearance
from joint_minimum_snap import (
    DEFAULT_DEVIATION_WEIGHT, DEFAULT_TRUST_M, solve_joint_waypoint_minimum_snap, warm_joint_backend,
)
from minimum_snap import MinimumSnapTrajectory, solve_minimum_snap
from simulator import MIXER

FLIGHT_PHASES = frozenset(("fly", "approach", "bypass", "cross", "release", "retreat"))
ROUTE_LATERAL_M = .35
HANDOFF_LATERAL_M = .55
ROUTE_END_MARGIN_M = .40
WINDOW_M = 10.
REFRESH_S = .85
RETRY_S = .25
TRACK_ERROR_M = .17
HANDOFF_VELOCITY_ERROR_MPS = .25
MAX_ACCEL = 1.85
MAX_JERK = 6.
SPEEDS = {"fly": 1.50, "approach": .78, "bypass": 1.30, "cross": 1.20,
          "release": 1.02, "retreat": .68}
BYPASS_OUTLET_OFFSETS = ((.15, 0.), (.30, 0.), (.15, -.08))
MAX_HANDOFF_CANDIDATES = 6
HANDOFF_COMPARISON_BUDGET_S = .035
WALL_VELOCITY_WEIGHT = .25


def _right_controls(control, u):
    """Control polygon for the suffix of a Bézier segment after parameter u."""
    level = np.asarray(control)
    right = [level[-1]]
    while len(level) > 1:
        level = (1.-u)*level[:-1] + u*level[1:]
        right.append(level[-1])
    return np.asarray(right[::-1])


@dataclass(frozen=True)
class PlannedTrajectory(MinimumSnapTrajectory):
    reference_points: np.ndarray
    corridors: tuple
    solver: str
    free_indices: tuple
    leaves: tuple


def _seamless(curve):
    return tuple(zip(curve.control_points[:-1, -1], np.cumsum(curve.durations)[:-1]))


@dataclass
class ScheduledFlight:
    curve: object
    start: float
    revision: int
    seamless: tuple
    extend: bool
    braking: bool = False
    route_margin: float = ROUTE_LATERAL_M

    def reference(self, t):
        return self.curve.sample(t-self.start)

    def remaining(self, t):
        elapsed = max(0., t-self.start)
        controls = self.curve.control_points
        for i, duration in enumerate(self.curve.durations):
            if elapsed < duration:
                return (_right_controls(controls[i], elapsed/duration), *controls[i+1:])
            elapsed -= duration
        return (controls[-1][-1:],)


def _window(position, waypoints, *, include_bypass_exit=False, roles=None):
    points = [np.asarray(position)]
    indices = [None]
    remaining = WINDOW_M
    extend = False
    for index, waypoint in enumerate(waypoints):
        waypoint = np.asarray(waypoint)
        length = float(np.linalg.norm(waypoint-points[-1]))
        if length < .025 and len(points) > 1:
            continue
        if length < 1e-6:
            continue
        if length > remaining:
            points.append(points[-1] + (waypoint-points[-1])*remaining/length)
            indices.append(None)
            extend = True
            break
        points.append(waypoint)
        indices.append(index)
        remaining -= length
        if remaining < .10 or len(points) == (5 if include_bypass_exit else 4):
            extend = float(np.linalg.norm(points[-1]-waypoints[-1])) > .025
            break
    free = np.zeros(len(points), dtype=bool)
    if roles is not None:
        for local, route in enumerate(indices[1:-1], start=1):
            free[local] = roles[route] == "transition"
    return np.asarray(points), extend, free


class RollingFlight:
    def __init__(self, arena, mass, *, wall_smoothing=False, opt_mode="baseline",
                 deviation_weight=DEFAULT_DEVIATION_WEIGHT, trust_radius=DEFAULT_TRUST_M):
        if opt_mode not in ("baseline", "joint"):
            raise ValueError("opt_mode must be baseline or joint")
        self.opt_mode = opt_mode
        self.qp_backend_ready = warm_joint_backend() if opt_mode == "joint" else None
        self.deviation_weight = deviation_weight
        self.trust_radius = trust_radius
        self.qp_attempts = []
        self.joint_fallbacks = 0
        self.arena = arena
        self.mass = mass
        self.wall_smoothing = wall_smoothing
        self.active = None
        self.leaves_key = None
        self.next_attempt = 0.
        self.mode = "inactive"
        self.failure_cause = ""
        self.hold = None
        self.plans = 0
        self.extensions = 0
        self.rechecks = 0
        self.invalidations = 0
        self.failures = 0
        self.brakes = 0
        self.handoff_rejections = 0
        self.outlet_adjustments = 0
        self.handoff_adjustments = {}
        self.handoff_comparisons = {}
        self.handoff_budget_limits = 0
        self.outlet_checked_revision = None
        self.solve_ms = []
        self.preview = ()

    def _nominal_motor_peak(self, acceleration, jerk):
        # Fixed-height nominal collective and a reserve for the attitude-rate
        # feedback. Motor commands in the closed loop are audited separately.
        collective = self.mass*np.hypot(9.81, acceleration)
        rate_reserve = np.abs(MIXER[:, 1:]) @ (np.array([.16, .16, .09])*jerk/9.81)
        return float(np.max(MIXER[:, 0]*collective + rate_reserve))

    def _dynamic_scales(self, bounds, speed, acceleration_limit):
        vmax, amax, jmax = bounds.T
        scales = np.maximum.reduce((vmax/speed, np.sqrt(amax/acceleration_limit),
                                    np.cbrt(jmax/MAX_JERK)))
        for index, (_, accel, jerk) in enumerate(bounds):
            if self._nominal_motor_peak(accel, jerk) > 9.0:
                scales[index] = max(scales[index], 1.16)
        return scales

    def _recover_segment_time(self, curve, points, fit,
                              lengths, initial_durations, speed, acceleration_limit):
        # Try shortening only inflated long legs with spare dynamic margin.
        # Each attempt re-solves the coupled curve and rechecks every segment.
        segments = np.diff(points, axis=0)
        along_wall = abs(segments[:, 0]) <= .30*abs(segments[:, 1])
        minimum_durations = np.maximum(initial_durations, .80*curve.durations)
        for _ in range(3):
            scales = self._dynamic_scales(curve.segment_derivative_bounds(), speed, acceleration_limit)
            spare = curve.durations - minimum_durations
            eligible = along_wall & (lengths >= 3.) & (spare >= .15) & (scales < .92)
            if not np.any(eligible):
                break
            segment = int(np.argmax(np.where(eligible, spare, -1.)))
            durations = curve.durations.copy()
            reduction = min(spare[segment], durations[segment]*.12)
            for factor in (1., .5):
                durations[segment] = curve.durations[segment] - factor*reduction
                trial = fit(durations, regularize_progress=True)
                if trial is not None and np.all(self._dynamic_scales(
                        trial.segment_derivative_bounds(), speed, acceleration_limit) <= 1.001):
                    curve = trial
                    break
            else:
                break
        return curve

    def _solve(self, points, velocity, acceleration, leaves, speed, *, lateral_margin=ROUTE_LATERAL_M,
               segment_timing=False, wall_shape=False, free_mask=None, start_jerk=None):
        corridors = build_corridors(points, self.arena, leaves, margin=FLIGHT_MARGIN)
        if corridors is None:
            return None, "corridor"
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        for index, (a, b) in enumerate(corridors):
            # Snap alone can send a curve far across a wide room while
            # joining two short gate legs. Bound its deviation from the route.
            along = segments[index]/lengths[index]
            across = np.array([-along[1], along[0]])
            directions = np.vstack((across, -across, -along, along))
            bounds = directions @ points[index]
            bounds += (lateral_margin, lateral_margin, ROUTE_END_MARGIN_M,
                       lengths[index]+ROUTE_END_MARGIN_M)
            corridors[index] = np.vstack((a, directions)), np.r_[b, bounds]
        speed = max(speed, float(np.linalg.norm(velocity))*1.05)
        acceleration_limit = max(MAX_ACCEL, float(np.linalg.norm(acceleration))*1.05)
        initial_durations = np.maximum(.55, 1.5*lengths/speed)
        def fit_baseline(durations, **kwargs):
            return solve_minimum_snap(points, durations, velocity, acceleration, corridors, **kwargs)

        # Preserve each movable milestone's room-side classification even
        # with a user-supplied trust radius larger than the default.
        regions = {}
        if free_mask is not None:
            for index in np.flatnonzero(free_mask):
                x = points[index, 0]
                lower = max((wall for wall in self.arena.partitions if wall <= x), default=-np.inf)
                upper = min((wall for wall in self.arena.partitions if wall >= x), default=np.inf)
                limits = [(sign, bound) for sign, bound in ((1., upper), (-1., -lower)) if np.isfinite(bound)]
                if limits:
                    sign, bound = np.asarray(limits).T
                    regions[int(index)] = (np.column_stack((sign, np.zeros_like(sign))), bound)

        def fit_joint(durations, **kwargs):
            result = solve_joint_waypoint_minimum_snap(
                points, durations, velocity, acceleration, corridors, free_mask=free_mask,
                trust_radius=self.trust_radius, deviation_weight=self.deviation_weight, waypoint_regions=regions,
                start_jerk=start_jerk, **kwargs)
            curve = result.trajectory
            certified = curve is not None and certify_controls(
                curve.control_points, self.arena, leaves, margin=FLIGHT_MARGIN)
            self.qp_attempts.append(dict(ms=1000.*result.solve_time_s,
                                         status=result.status if curve is None or certified else "geometry_failed",
                                         reason=result.reason, solver_status=result.solver_status,
                                         max_residual=result.max_residual))
            return curve if certified else None

        def time_adjust(fit):
            for per_segment in ((True, False) if segment_timing and len(lengths) > 1 else (False,)):
                durations = initial_durations.copy()
                for _ in range(6):
                    curve = fit(durations, regularize_progress=per_segment)
                    if curve is None:
                        if not per_segment:
                            return None, "boundary"
                        break
                    bounds = curve.segment_derivative_bounds()
                    if not per_segment:
                        bounds = bounds.max(axis=0, keepdims=True)
                    scales = self._dynamic_scales(bounds, speed, acceleration_limit)
                    exceeded = scales > 1.001
                    if not np.any(exceeded):
                        if per_segment:
                            curve = self._recover_segment_time(curve, points, fit, lengths,
                                                               initial_durations, speed, acceleration_limit)
                        if (self.wall_smoothing and wall_shape and np.any(
                                (abs(segments[:, 1]) >= 2.5) & (abs(segments[:, 0]) <= .30*abs(segments[:, 1])))
                                and self._wall_sway(curve) >= .025):
                            trial = fit(curve.durations, regularize_progress=per_segment,
                                        wall_velocity_weight=WALL_VELOCITY_WEIGHT)
                            if trial is not None and self._prefer_wall_shape(
                                    curve, trial, leaves, speed, acceleration_limit):
                                curve = trial
                        return curve, ""
                    durations = durations * np.where(exceeded, np.maximum(1.06, 1.035*scales), 1.)
            return None, "dynamic"

        use_joint = self.opt_mode == "joint" and (
            (free_mask is not None and np.any(free_mask)) or start_jerk is not None)
        if use_joint:
            curve, why = time_adjust(fit_joint)
            if curve is not None:
                return PlannedTrajectory(curve.control_points, curve.durations, points.copy(),
                                         tuple(corridors), "joint",
                                         tuple(map(int, np.flatnonzero(free_mask)))
                                         if free_mask is not None else (), tuple(leaves)), ""
            self.joint_fallbacks += 1
        curve, why = time_adjust(fit_baseline)
        if curve is None:
            return None, why
        if self.opt_mode == "joint" and not certify_controls(curve.control_points, self.arena, leaves, margin=FLIGHT_MARGIN):
            return None, "corridor"
        return PlannedTrajectory(curve.control_points, curve.durations, points.copy(),
                                 tuple(corridors), "baseline", (), tuple(leaves)), ""

    def _brake(self, t, position, velocity, acceleration, revision, leaves):
        speed = float(np.linalg.norm(velocity))
        if speed < .07:
            return None
        direction = velocity/speed
        distance = max(.08, .55*speed + speed*speed/(2*MAX_ACCEL))
        finish = position + distance*direction
        curve, _ = self._solve(np.vstack((position, finish)), velocity, acceleration, leaves,
                               max(.5, speed*1.08))
        if curve is None:
            return None
        self.brakes += 1
        return ScheduledFlight(curve, t, revision, (), False, braking=True)

    def _set(self, scheduled, mode):
        self.active = scheduled
        self.mode = mode
        self.failure_cause = ""
        self.hold = None
        self.preview = (tuple(tuple(scheduled.curve.sample(t)[0])
                              for t in np.linspace(0., scheduled.curve.duration, 35))
                        if scheduled else ())

    def _matching_knot(self, waypoint):
        if self.active is None or self.active.braking or waypoint is None:
            return None
        for anchor, knot in zip(self.active.curve.reference_points[1:-1], self.active.seamless):
            if np.linalg.norm(waypoint-anchor) < .015:
                return knot
        return None

    def execution_waypoint(self, revision, waypoint):
        """Resolve a task anchor to the current schedule's local crossing point."""
        if self.active is not None and self.active.revision == revision and self.active.curve.solver == "joint":
            match = self._matching_knot(waypoint)
            if match is not None:
                return match[0]
        return None

    def can_pass(self, t, waypoint):
        match = self._matching_knot(waypoint)
        return match is not None and abs(t-self.active.start-match[1]) < .55

    def reference(self, t):
        if self.active is not None:
            return self.active.reference(t)
        if self.hold is not None:
            return self.hold, np.zeros(2), np.zeros(2), np.zeros(2)
        return None

    @staticmethod
    def _geometry_key(leaves):
        return tuple((leaf.gate.name, leaf.side, round(leaf.angle, 6)) for leaf in leaves)

    def _plan(self, t, position, velocity, acceleration, revision, waypoints, phase, leaves, *,
              old, same_path=False, recovering=False, retreat_bypass=False, roles=None):
        started = perf_counter()
        previous = old.reference(t) if old is not None and not old.braking else None
        use_previous = previous is not None and np.linalg.norm(position-previous[0]) < .09
        if use_previous and not same_path:
            use_previous = (np.linalg.norm(velocity-previous[1]) <= HANDOFF_VELOCITY_ERROR_MPS
                            and certify_controls(np.vstack((position, previous[0])), self.arena,
                                                 leaves, margin=FLIGHT_MARGIN))
        initial_acceleration = np.clip(acceleration, -.45, .45) if recovering else acceleration
        state = previous[:3] if use_previous else (position, velocity, initial_acceleration)
        points, extend, free = _window(state[0], waypoints, include_bypass_exit=retreat_bypass, roles=roles)
        start_jerk = previous[3] if use_previous and self.opt_mode == "joint" else None
        moving_handoff = (use_previous and not same_path
                          and (phase == "cross" or (phase == "approach" and len(points) == 2)))
        lateral_margin = HANDOFF_LATERAL_M if moving_handoff else ROUTE_LATERAL_M
        curve, why = (self._solve(points, state[1], state[2], leaves, SPEEDS[phase],
                                  lateral_margin=lateral_margin, segment_timing=retreat_bypass,
                                  wall_shape=phase == "bypass", free_mask=free, start_jerk=start_jerk)
                      if len(points) > 1 else (None, "at_target"))
        if curve is None and use_previous and why != "at_target":
            points, extend, free = _window(position, waypoints, include_bypass_exit=retreat_bypass, roles=roles)
            if len(points) > 1:
                curve, why = self._solve(points, velocity, acceleration, leaves, SPEEDS[phase],
                                         lateral_margin=lateral_margin, segment_timing=retreat_bypass,
                                         wall_shape=phase == "bypass", free_mask=free)
        if curve is None and why == "boundary" and old is None and phase == "release":
            lateral_margin = HANDOFF_LATERAL_M
            curve, why = self._solve(points, velocity, initial_acceleration, leaves, SPEEDS[phase],
                                     lateral_margin=lateral_margin, free_mask=free)
        self.solve_ms.append((perf_counter()-started)*1000.)
        if curve is None:
            return None, why
        return ScheduledFlight(curve, t, revision, _seamless(curve), extend, route_margin=lateral_margin), ""

    def _accept(self, scheduled):
        self._set(scheduled, "tracking")
        self.next_attempt = scheduled.start + .10
        self.plans += 1

    @staticmethod
    def _wall_sway(curve):
        sway = 0.
        knots = np.r_[0., np.cumsum(curve.durations)]
        for control, start, end in zip(curve.control_points, knots, knots[1:]):
            dx, dy = control[-1]-control[0]
            if abs(dy) >= 2. and abs(dx) <= .30*abs(dy):
                x = [curve.sample(t)[0][0] for t in np.linspace(start, end, 25)]
                sway += np.abs(np.diff(x)).sum()-abs(dx)
        return sway

    def _trajectory_metrics(self, curve, phase, leaves, *, sample_count=121):
        samples = np.array([curve.sample(t)[:3] for t in np.linspace(0., curve.duration, sample_count)])
        position, velocity, acceleration = samples.transpose(1, 0, 2)
        speed = np.linalg.norm(velocity, axis=1)
        cross = abs(velocity[:, 0]*acceleration[:, 1]-velocity[:, 1]*acceleration[:, 0])
        peak_turn_rate = float((cross/np.maximum(speed, .30)**2).max())
        knots = np.r_[0., np.cumsum(curve.durations)]
        alignment = 0.
        for index, knot in enumerate(knots[1:-1], start=1):
            p, v, _, _ = curve.sample(knot)
            onward = curve.control_points[index, -1]-p
            magnitude = np.linalg.norm(v)*np.linalg.norm(onward)
            if np.linalg.norm(v) > .30 and magnitude > 1e-6:
                alignment += 1.-float(np.clip((v @ onward)/magnitude, -1., 1.))
        sway = self._wall_sway(curve) if phase == "bypass" else 0.
        score = curve.duration + .8*peak_turn_rate + 3.*sway + .15*alignment
        return score, sampled_clearance(position, self.arena, leaves), peak_turn_rate, sway

    def _prefer_wall_shape(self, original, trial, leaves, speed, acceleration_limit):
        bounds = trial.segment_derivative_bounds()
        if np.any(self._dynamic_scales(bounds, speed, acceleration_limit) > 1.001):
            return False
        _, clearance, turn, sway = self._trajectory_metrics(original, "bypass", leaves, sample_count=61)
        _, trial_clearance, trial_turn, trial_sway = self._trajectory_metrics(trial, "bypass", leaves, sample_count=61)
        return (trial_sway < sway-.015 and trial_turn <= turn*1.10+.03
                and trial_clearance >= clearance-.01
                and bounds[:, 2].max() <= original.derivative_bounds()[2]*1.10+.10)

    def _compare_handoff(self, scheduled, waypoints, alternatives, phase, leaves, *, retreat_bypass=False, role_factory=None):
        """Compare complete executable curves from the same state to the same stop."""
        if not alternatives:
            return scheduled, waypoints
        self.handoff_comparisons[phase] = self.handoff_comparisons.get(phase, 0)+1
        started = perf_counter()
        position, velocity, acceleration, jerk = scheduled.curve.sample(0.)
        finish = scheduled.curve.sample(scheduled.curve.duration)[0]
        # When the horizon clips the route, only move knots already inside it.
        used = []
        if scheduled.extend:
            first = 0
            for point in scheduled.curve.reference_points[1:-1]:
                index = next((i for i in range(first, len(waypoints))
                              if np.linalg.norm(point-waypoints[i]) < .015), None)
                if index is None:
                    return scheduled, waypoints
                used.append(index)
                first = index+1
        best, selected = scheduled, waypoints
        best_score, clearance, turn_rate, sway = self._trajectory_metrics(scheduled.curve, phase, leaves)
        for candidate in alternatives:
            if perf_counter()-started >= HANDOFF_COMPARISON_BUDGET_S:
                self.handoff_budget_limits += 1
                break
            if scheduled.extend:
                if (len(candidate) != len(waypoints)
                        or any(np.linalg.norm(np.asarray(a)-b) > .01 for i, (a, b) in
                               enumerate(zip(candidate, waypoints)) if i not in used)):
                    continue
                inner = [candidate[i] for i in used]
            else:
                if np.linalg.norm(finish-candidate[-1]) > .015:
                    continue
                inner = candidate[:-1]
            points = np.asarray((position, *inner, finish))
            if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < .025):
                continue
            roles = role_factory(candidate) if role_factory is not None else ()
            indices = used if scheduled.extend else range(len(inner))
            free = [False, *(bool(roles) and roles[i] == "transition" for i in indices), False]
            curve, _ = self._solve(points, velocity, acceleration, leaves, SPEEDS[phase],
                                   lateral_margin=scheduled.route_margin, segment_timing=retreat_bypass,
                                   free_mask=free, start_jerk=jerk if scheduled.curve.solver == "joint" else None)
            if curve is None:
                continue
            score, trial_clearance, trial_turn, trial_sway = self._trajectory_metrics(curve, phase, leaves)
            if (trial_clearance < clearance-.025 or trial_turn > turn_rate*1.10
                    or trial_sway > sway+.04 or score >= best_score-.08):
                continue
            best = ScheduledFlight(curve, scheduled.start, scheduled.revision,
                                   _seamless(curve), scheduled.extend,
                                   route_margin=scheduled.route_margin)
            selected, best_score = candidate, score
        self.solve_ms[-1] += (perf_counter()-started)*1000.
        if best is not scheduled:
            self.handoff_adjustments[phase] = self.handoff_adjustments.get(phase, 0)+1
        return best, selected

    def _bypass_metrics(self, curve, gate, leaves):
        samples = np.array([curve.sample(t)[:3] for t in np.linspace(0., curve.duration, 161)])
        position, velocity, acceleration = samples.transpose(1, 0, 2)
        local = ((abs(position[:, 1]-gate.bypass_middle) < 2.)
                 & (gate.x-1.35 < position[:, 0]) & (position[:, 0] < gate.x+2.))
        speed = np.linalg.norm(velocity, axis=1)
        turning = local & (speed > .30)
        curvature = abs(velocity[:, 0]*acceleration[:, 1]-velocity[:, 1]*acceleration[:, 0])
        curvature /= np.maximum(speed, .30)**3
        peak = float(curvature[turning].max(initial=0.))
        exit_peak = float(curvature[turning & (position[:, 0] >= gate.x)].max(initial=0.))
        clearance = sampled_clearance(position[local], self.arena, leaves) if np.any(local) else 0.
        return peak, exit_peak, clearance

    def _choose_bypass_outlet(self, scheduled, waypoints, outlet, gate, leaves):
        _, velocity, acceleration, jerk = scheduled.curve.sample(0.)
        points = scheduled.curve.reference_points
        free = np.zeros(len(points), dtype=bool)
        free[list(scheduled.curve.free_indices)] = True
        matches = np.flatnonzero(np.linalg.norm(points[1:-1]-waypoints[outlet], axis=1) < .015)
        if not len(matches):
            return scheduled

        index = int(matches[0])+1
        outbound_side = np.sign(waypoints[-1][1]-gate.bypass_middle)
        peak, exit_peak, clearance = self._bypass_metrics(scheduled.curve, gate, leaves)
        best_score = .7*peak + .3*exit_peak + .12*scheduled.curve.duration
        best, best_point = scheduled, None
        for forward, lateral in BYPASS_OUTLET_OFFSETS:
            trial_points = points.copy()
            trial_points[index] += (forward, lateral*outbound_side)
            curve, _ = self._solve(trial_points, velocity, acceleration, leaves, SPEEDS["bypass"],
                                   segment_timing=True, free_mask=free,
                                   start_jerk=jerk if scheduled.curve.solver == "joint" else None)
            if curve is None:
                continue
            trial_peak, trial_exit, trial_clearance = self._bypass_metrics(curve, gate, leaves)
            score = .7*trial_peak + .3*trial_exit + .12*curve.duration
            if trial_clearance < clearance-.025 or trial_peak > peak*1.10 or score >= best_score-.03:
                continue
            best = ScheduledFlight(curve, scheduled.start, scheduled.revision,
                                   _seamless(curve), scheduled.extend)
            best_score, best_point = score, trial_points[index].copy()
        if best_point is not None:
            waypoints[outlet] = best_point
            self.outlet_adjustments += 1
        return best

    def try_handoff(self, t, position, velocity, acceleration, revision, waypoints, phase, leaves, *,
                    retreat_bypass=False, alternatives=(), role_factory=None):
        """Install a certified candidate only when it can replace the current flight."""
        leaves = tuple(leaves)
        old = self._recertify(t, leaves)
        candidates = (tuple(waypoints), *alternatives[:MAX_HANDOFF_CANDIDATES-1])
        for index, route in enumerate(candidates):
            scheduled, _ = self._plan(t, position, velocity, acceleration, revision, route, phase, leaves,
                                      old=old, retreat_bypass=retreat_bypass,
                                      roles=role_factory(route) if role_factory is not None else None)
            if scheduled is not None:
                break
        else:
            self.handoff_rejections += 1
            return False
        scheduled, selected = self._compare_handoff(scheduled, route, candidates[index+1:], phase, leaves,
                                                    retreat_bypass=retreat_bypass, role_factory=role_factory)
        if isinstance(waypoints, list):
            waypoints[:] = selected
        self._accept(scheduled)
        return True

    def _recertify(self, t, leaves):
        key = self._geometry_key(leaves)
        old = self.active
        if old is not None and key != self.leaves_key:
            self.rechecks += 1
            if not all(certify_controls(piece, self.arena, leaves, margin=FLIGHT_MARGIN)
                       for piece in old.remaining(t)):
                self.invalidations += 1
                self._set(None, "invalidated")
                self.next_attempt = t
                old = None
        self.leaves_key = key
        return old

    def update(self, t, position, velocity, acceleration, revision, waypoints, phase, leaves, *,
               separating=False, retreat_bypass=False, bypass_outlet=None, release_alternatives=(),
               role_factory=None):
        if phase not in FLIGHT_PHASES or not waypoints:
            self._set(None, "inactive")
            return
        leaves = tuple(leaves)
        old = self._recertify(t, leaves)
        same_path = old is not None and old.revision == revision
        previous = old.reference(t) if old is not None else None
        error = float(np.linalg.norm(position-previous[0])) if previous is not None else 0.
        time_left = old.start + old.curve.duration - t if old is not None else 0.
        extension = (same_path and old.extend
                     and time_left < max(REFRESH_S, .45*old.curve.duration))
        check_outlet = (same_path and bypass_outlet is not None and not old.braking
                        and self.outlet_checked_revision != revision
                        and abs(position[1]-bypass_outlet[1].bypass_middle) <= 3.2
                        and position[0] < bypass_outlet[1].x)
        retry_after_brake = old is not None and old.braking and time_left <= .01
        if same_path and not extension and not retry_after_brake and not check_outlet and error < TRACK_ERROR_M:
            return
        if t < self.next_attempt and (old is None or same_path):
            return
        recovering = self.mode == "holding" and self.hold is not None
        if recovering and (np.linalg.norm(velocity) > .11 or np.linalg.norm(position-self.hold) > .10):
            return

        # The current contact-to-free handoff first follows the verified separating
        # direction. A normal collision-free corridor starts outside the leaf.
        if (old is None and separating
                and not certify_controls(position.reshape(1, 2), self.arena, leaves, margin=FLIGHT_MARGIN)):
            self._set(None, "separating")
            return

        scheduled, why = self._plan(t, position, velocity, acceleration, revision, waypoints, phase, leaves,
                                    old=old, same_path=same_path, recovering=recovering,
                                    retreat_bypass=retreat_bypass, roles=role_factory(waypoints) if role_factory else None)
        if scheduled is None and old is None:
            for candidate in release_alternatives:
                scheduled, _ = self._plan(t, position, velocity, acceleration, revision, candidate,
                                          phase, leaves, old=None, recovering=recovering,
                                          roles=role_factory(candidate) if role_factory else None)
                if scheduled is not None:
                    waypoints[:] = candidate
                    release_alternatives = ()
                    break
        if scheduled is not None:
            if old is None and release_alternatives:
                scheduled, selected = self._compare_handoff(scheduled, waypoints, release_alternatives,
                                                            phase, leaves, role_factory=role_factory)
                waypoints[:] = selected
            if check_outlet:
                self.outlet_checked_revision = revision
                started = perf_counter()
                chosen = self._choose_bypass_outlet(scheduled, waypoints, *bypass_outlet, leaves)
                self.solve_ms[-1] += (perf_counter()-started)*1000.
                if chosen is scheduled and not extension and error < TRACK_ERROR_M:
                    return
                scheduled = chosen
            self._accept(scheduled)
            self.extensions += int(extension)
            return

        self.next_attempt = t + RETRY_S
        if why != "at_target":
            self.failures += 1
        if same_path and error < TRACK_ERROR_M and time_left > .10 and not old.braking:
            self.mode = "validated_prefix"
            self.failure_cause = why
            return
        brake = self._brake(t, position, velocity, acceleration, revision, leaves)
        previous_hold = self.hold if recovering else None
        self._set(brake, "braking" if brake else ("arrived" if why == "at_target" else "holding"))
        self.failure_cause = why
        if brake is None:
            self.hold = position.copy() if previous_hold is None else previous_hold

    def diagnostics(self):
        successful = [row["ms"] for row in self.qp_attempts if row["status"] == "solved"]
        qp = dict(backend_ready=self.qp_backend_ready, attempts=len(self.qp_attempts), successes=len(successful), fallbacks=self.joint_fallbacks,
                  status_counts={state: sum(row["status"] == state for row in self.qp_attempts)
                                 for state in sorted({row["status"] for row in self.qp_attempts})})
        if self.qp_attempts:
            times = [row["ms"] for row in self.qp_attempts]
            qp.update(p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)), max_ms=max(times))
        return dict(qp=qp, plans=self.plans, horizon_extensions=self.extensions, geometry_rechecks=self.rechecks,
                    invalidations=self.invalidations, solve_failures=self.failures, brakes=self.brakes,
                    handoff_rejections=self.handoff_rejections,
                    outlet_adjustments=self.outlet_adjustments,
                    handoff_adjustments=dict(self.handoff_adjustments),
                    handoff_comparisons=dict(self.handoff_comparisons),
                    handoff_budget_limits=self.handoff_budget_limits,
                    median_solve_ms=round(float(np.median(self.solve_ms)), 3) if self.solve_ms else None,
                    max_solve_ms=round(max(self.solve_ms), 3) if self.solve_ms else None)
