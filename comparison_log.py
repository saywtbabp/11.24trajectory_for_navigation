"""Offline instrumentation for paired runs. No inputs to the controller."""
import numpy as np

from joint_minimum_snap import SNAP_DIFFERENCE_FACTOR, physical_snap
from minimum_snap import _BINOMIAL


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)


def executed_snap(curve, elapsed):
    """Integrate squared snap exactly on the prefix executed before replacement."""
    if elapsed >= curve.duration:
        return physical_snap(curve)
    total = 0.
    for control, duration in zip(curve.control_points, curve.durations):
        fraction = np.clip(elapsed/duration, 0., 1.)
        if fraction <= 0.:
            break
        s = (.5*fraction*(_GAUSS_X + 1.))[:, None]
        power = np.arange(4)
        basis = _BINOMIAL[3] * s**power * (1.-s)**power[::-1]
        scaled_snap = basis @ (SNAP_DIFFERENCE_FACTOR*np.diff(control, n=4, axis=0))
        total += .5*fraction*float(_GAUSS_W @ np.sum(scaled_snap**2, axis=1))/duration**7
        elapsed -= duration
    return total


class ComparisonLog:
    def __init__(self):
        self.frames = []
        self.plans = []
        self.astar = []
        self._active = None
        self._route = None
        self._previous_contacts = set()
        self.collisions = {}
        self.snap = 0.
        self.qp_attempts = []

    def _close_plan(self, t):
        if self._active is not None:
            span = executed_snap(self._active.curve, t-self._active.start)
            self.plans[-1].update(end=t, executed_snap=span)
            self.snap += span

    def record_scan(self, t, truth, nav, flight, reference):
        route = nav.route_edges
        if route is not self._route:
            self._route = route
            if route:
                origin = nav.route[0]
                self.astar.append(dict(t=t, points=np.asarray(
                    (origin, *(p for edge in route for p in edge.waypoints))).tolist()))
        active = flight.active if flight is not None else None
        if flight is not None:
            self.qp_attempts = flight.qp_attempts
        if active is not self._active:
            self._close_plan(t)
            self._active = active
            if active is not None:
                curve = active.curve
                points = np.vstack((curve.control_points[0, 0], curve.control_points[:, -1]))
                self.plans.append(dict(t=active.start, phase=nav.phase, door=nav.active or nav.pending,
                                       revision=active.revision, solver=curve.solver,
                                       braking=active.braking, original=curve.reference_points.tolist(),
                                       optimized=points.tolist(), free_indices=list(curve.free_indices),
                                       durations=curve.durations.tolist(), controls=curve.control_points.tolist(),
                                       leaves=[dict(gate=leaf.gate.name, side=leaf.side, angle=leaf.angle)
                                               for leaf in curve.leaves],
                                       corridors=[(a.tolist(), b.tolist()) for a, b in curve.corridors]))
        row = dict(t=t, phase=nav.phase, door=nav.active or nav.pending, actual=truth.tolist())
        if reference is not None and active is not None:
            row["reference"] = [value.tolist() for value in reference]
        self.frames.append(row)

    def record_contacts(self, contacts):
        for name in contacts-self._previous_contacts:
            self.collisions[name] = self.collisions.get(name, 0)+1
        self._previous_contacts = contacts

    def finish(self, t):
        self._close_plan(t)
        self._active = None
        refs = [row["reference"] for row in self.frames if "reference" in row]
        summary = dict(executed_snap_m2_s7=self.snap, collision_episodes=sum(self.collisions.values()),
                       collision_episodes_by_geom=self.collisions)
        if refs:
            ref = np.asarray(refs)
            speed = np.linalg.norm(ref[:, 1], axis=1)
            jerk = np.linalg.norm(ref[:, 3], axis=1)
            cross = abs(ref[:, 1, 0]*ref[:, 2, 1]-ref[:, 1, 1]*ref[:, 2, 0])
            summary.update(jerk_rms=float(np.sqrt(np.mean(jerk**2))),
                           jerk_p95=float(np.percentile(jerk, 95)), jerk_peak=float(max(jerk)),
                           curvature_peak=float(max(cross[speed > .30]/speed[speed > .30]**3, default=0.)))
        moves = [float(distance) for p in self.plans if p["free_indices"] for distance in
                 np.linalg.norm(np.array(p["optimized"])-np.array(p["original"]), axis=1)[p["free_indices"]]]
        summary.update(moved_knots=sum(move > 1e-5 for move in moves),
                       max_waypoint_move_m=max(moves, default=0.),
                       mean_waypoint_move_m=float(np.mean(moves)) if moves else None)
        return summary
