import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from comparison_log import executed_snap
from door_geometry import Leaf
from flight_corridors import FLIGHT_MARGIN, build_corridors, certify_controls
from joint_minimum_snap import physical_snap, solve_joint_waypoint_minimum_snap as solve
from minimum_snap import solve_minimum_snap
from planner import Navigator
from rolling_flight import RollingFlight
from scene import Arena, Gate


ZERO = np.zeros(2)
FREE = [False, True, False]
POINTS = np.array([[0., 0.], [.8, .2], [3., 0.]])


def box(xmin, xmax, ymin, ymax):
    return np.vstack((np.eye(2), -np.eye(2))), np.array([xmax, ymax, -xmin, -ymin])


def check(test, result, corridors):
    test.assertEqual(result.status, "solved", result)
    test.assertLessEqual(result.max_residual, 1e-8)
    curve = result.trajectory
    test.assertTrue(np.isfinite(curve.control_points).all())
    for controls, (a, b) in zip(curve.control_points, corridors):
        test.assertLessEqual(np.max(controls @ a.T-b, initial=0.), 2e-8)
    for polygon in curve._derivatives:
        assert_allclose(polygon[1:, 0], polygon[:-1, -1], rtol=1e-7, atol=1e-7)
    for derivative in curve.sample(curve.duration)[1:]:
        assert_array_equal(derivative, ZERO)
    return curve


class JointTests(unittest.TestCase):
    def test_01_fixed_matches_baseline_including_optional_penalties(self):
        corridors = [box(-.1, 1.18, -.35, .45), box(.35, 3.1, -.35, .45)]
        points = np.array([[0., 0.], [.8, .2], [3., 0.]])
        for opts in ({}, {"regularize_progress": True}):
            base = solve_minimum_snap(points, [1., 1.1], ZERO, ZERO, corridors, **opts)
            result = solve(points, [1., 1.1], ZERO, ZERO, corridors, **opts)
            curve = check(self, result, corridors)
            assert_allclose(curve.control_points, base.control_points, atol=3e-7)
            for t in np.linspace(0., curve.duration, 15):
                assert_allclose(curve.sample(t), base.sample(t), atol=3e-5)
            self.assertAlmostEqual(physical_snap(base)/result.snap, 1., places=6)
            self.assertAlmostEqual(executed_snap(curve, curve.duration)/result.snap, 1.)

        wall = np.array([[0., 0.], [.2, 4.], [1.4, 5.]])
        opts = dict(regularize_progress=True, wall_velocity_weight=.25)
        wide = [box(-3, 4, -1, 7)]*2
        base = solve_minimum_snap(wall, [5., 2.], ZERO, ZERO, wide, **opts)
        joint = check(self, solve(wall, [5., 2.], ZERO, ZERO, wide, **opts), wide)
        assert_allclose(joint.control_points, base.control_points, atol=3e-7)

    def test_02_wide(self):
        corridors = [box(-1, 4, -1, 1)]*2
        result = solve(POINTS, [1., 1.], ZERO, ZERO, corridors, free_mask=FREE)
        check(self, result, corridors)
        assert_array_equal(result.waypoints[[0, -1]], POINTS[[0, -1]])
        self.assertTrue(np.all(abs(result.offsets) <= .18+1e-8))
        self.assertGreater(np.linalg.norm(result.offsets), .1)

    def test_03_overlap_and_oblique_region(self):
        corridors = [box(-1, 1.15, -.5, 1.), box(.82, 4., -.5, .6)]
        region = {1: ([[1., -1.]], [.65])}
        result = solve(POINTS, [1.3, 1.], ZERO, ZERO, corridors, free_mask=FREE,
                       trust_radius=.5, waypoint_regions=region)
        check(self, result, corridors)
        self.assertGreater(result.waypoints[1, 0], .82-1e-8)
        self.assertLessEqual(result.waypoints[1, 0]-result.waypoints[1, 1], .65+1e-8)
        self.assertGreater(np.linalg.norm(result.offsets), .02)

    def test_04_disjoint_corridors(self):
        corridors = [box(-1, .7, -1, 1), box(1.2, 4., -1, 1)]
        result = solve(POINTS, [1, 1], ZERO, ZERO, corridors, free_mask=FREE, trust_radius=None)
        self.assertEqual(result.status, "infeasible")
        self.assertIsNone(result.trajectory)

    def test_05_narrow_and_degenerate(self):
        for width in (.002, 0.):
            corridors = [box(-.1, 1.6, -width, width), box(1.0, 3.1, -width, width)]
            points = POINTS.copy()
            points[1, 1] = 0.
            result = solve(points, [1, 1], ZERO, ZERO, corridors, free_mask=FREE, trust_radius=.9)
            check(self, result, corridors)
            self.assertLessEqual(abs(result.waypoints[1, 1]), width+1e-8)

    def test_06_fixed_contact_and_actual_local_knot(self):
        gate = Gate("T", 5., -1.2, 1.2, 2., 3.4)
        leaf = Leaf(gate, 1, 0.)
        approach = leaf.midpoint-.48*leaf.tangent
        arena = Arena((gate,), (-2., -3., 8., 5.), np.array([1., 0.]), np.array([7., 0.]))
        flight = RollingFlight(arena, 1., opt_mode="joint")
        anchors = [np.array([2.2, .17]), approach.copy()]
        factory = lambda route: Navigator.flight_waypoint_roles("approach", len(route))
        self.assertTrue(flight.try_handoff(0., arena.start, ZERO, ZERO, 3, anchors, "approach", (leaf,),
                                           role_factory=factory))
        assert_array_equal(flight.active.curve.control_points[-1, -1], approach)
        assert_array_equal(anchors[0], [2.2, .17])
        point, stamp = flight.active.seamless[0]
        self.assertEqual(flight.active.curve.solver, "joint")
        self.assertGreater(np.linalg.norm(point-anchors[0]), 1e-3)
        assert_allclose(flight.execution_waypoint(3, anchors[0]), point)
        assert_allclose(flight.reference(stamp)[0], point)
        self.assertTrue(flight.can_pass(stamp, anchors[0]))
        self.assertIsNone(flight.execution_waypoint(4, anchors[0]))

    def test_07_motion_and_fixed_initial_jerk(self):
        corridors = [box(-1, 4, -1, 1)]*2
        v, a, j = [.28, -.05], [.1, .12], [.7, -.3]
        for jerk in (None, j):
            curve = check(self, solve(POINTS, [1.3, 1.7], v, a, corridors, free_mask=FREE,
                                      start_jerk=jerk), corridors)
            state = curve.sample(0.)
            assert_allclose(state[:3], [POINTS[0], v, a], atol=1e-10)
            if jerk is not None:
                assert_allclose(state[3], j, atol=1e-10)

    def test_08_dynamic_leaf_invalidates_remaining_joint_trajectory(self):
        gate = Gate("T", 0., -1.2, 1.2, 2., 3.4)
        leaf = Leaf(gate, 1, np.pi/2)
        closed = replace(leaf, angle=0.)
        arena = Arena((gate,), (-4., -3., 6., 5.), np.array([-2., 0.]), np.array([4., 0.]))
        flight = RollingFlight(arena, 1., opt_mode="joint")
        anchors = [np.array([-1., .03]), arena.goal]
        permissions = lambda _: ("transition", "fixed")
        flight.update(0., arena.start, ZERO, ZERO, 3, anchors, "cross", (leaf,), role_factory=permissions)
        old = flight.active
        self.assertIsNotNone(old)
        self.assertEqual(old.curve.solver, "joint")
        self.assertTrue(certify_controls(old.curve.control_points, arena, (leaf,), margin=FLIGHT_MARGIN))
        self.assertIsNone(build_corridors(np.vstack((arena.start, anchors)), arena, (closed,), margin=FLIGHT_MARGIN))
        flight.update(.1, arena.start, ZERO, ZERO, 3, anchors, "cross", (closed,), role_factory=permissions)
        self.assertIsNot(flight.active, old)
        self.assertEqual(flight.invalidations, 1)
        self.assertEqual(flight.mode, "holding")

    def test_09_numerics_and_iteration_limit(self):
        corridors = [box(-1, 4, -1, 1)]*2
        close = np.array([[0., 0.], [1e-4, 1e-4], [3., 0.]])
        result = solve(close, [.1, 4.], ZERO, ZERO, corridors, free_mask=FREE, trust_radius=1e-5)
        check(self, result, corridors)
        self.assertLessEqual(abs(result.offsets).max(), 1e-5+1e-8)
        result = solve(POINTS, [1e-12, 1.], ZERO, ZERO, corridors, free_mask=FREE)
        self.assertEqual(result.status, "ill_conditioned")
        result = solve(POINTS, [1., 1.], ZERO, ZERO, corridors, free_mask=FREE, max_iter=1)
        self.assertEqual(result.status, "iteration_limit")
        self.assertIsNone(result.trajectory)
        with patch("joint_minimum_snap.np.linalg.svd", return_value=(
                np.zeros((10, 5)), np.array([1., 1., 1., 1., 1e-16]), np.eye(5))):
            result = solve(POINTS, [1., 1.], ZERO, ZERO, corridors, free_mask=FREE)
        self.assertEqual(result.status, "ill_conditioned")

    def test_10_fallback_and_no_route_mutation(self):
        arena = Arena((), (-4., -3., 6., 5.), POINTS[0], POINTS[-1])
        flight = RollingFlight(arena, 1., opt_mode="joint")
        anchors = [point.copy() for point in POINTS[1:]]
        with patch("rolling_flight.solve_joint_waypoint_minimum_snap") as joint:
            from joint_minimum_snap import JointResult
            joint.return_value = JointResult("iteration_limit", .01, reason="test injection")
            flight.update(0., POINTS[0], ZERO, ZERO, 2, anchors, "fly", (),
                          role_factory=lambda _: ("transition", "fixed"))
        self.assertEqual(flight.active.curve.solver, "baseline")
        self.assertGreater(flight.joint_fallbacks, 0)
        assert_array_equal(anchors, POINTS[1:])

    def test_11_whole_window_fixed_time_objective(self):
        corridors = [box(-1, 4, -1, 1)]*2
        fixed = solve(POINTS, [1., 1.], ZERO, ZERO, corridors)
        joint = solve(POINTS, [1., 1.], ZERO, ZERO, corridors, free_mask=FREE)
        check(self, fixed, corridors)
        check(self, joint, corridors)
        self.assertLess(joint.snap + 100.*joint.deviation, fixed.snap)

    def test_12_outer_timing_reuses_reference_and_resolves_joint_qp(self):
        arena = Arena((), (-4., -3., 6., 5.), POINTS[0], POINTS[-1])
        flight = RollingFlight(arena, 1., opt_mode="joint")
        with patch("rolling_flight.solve_joint_waypoint_minimum_snap", wraps=solve) as joint:
            curve, why = flight._solve(POINTS, ZERO, ZERO, (), 1., free_mask=FREE)
        self.assertIsNotNone(curve, why)
        self.assertEqual(curve.solver, "joint")
        self.assertGreater(joint.call_count, 1)
        first_times = joint.call_args_list[0].args[1]
        last_times = joint.call_args_list[-1].args[1]
        self.assertTrue(np.any(last_times > first_times))
        for call in joint.call_args_list:
            assert_array_equal(call.args[0], POINTS)
            assert_array_equal(call.kwargs["free_mask"], FREE)

    def test_13_rolling_handoff_preserves_c3_when_accepted_from_old_reference(self):
        arena = Arena((), (-4., -3., 8., 5.), POINTS[0], np.array([5., 0.]))
        flight = RollingFlight(arena, 1., opt_mode="joint")
        roles = lambda _: ("transition", "fixed")
        flight.update(0., POINTS[0], ZERO, ZERO, 1, list(POINTS[1:]), "fly", (), role_factory=roles)
        old = flight.active
        state = old.reference(.7)
        self.assertTrue(flight.try_handoff(.7, *state[:3], 2, [POINTS[1], arena.goal], "fly", (),
                                           role_factory=roles))
        self.assertIsNot(flight.active, old)
        self.assertEqual(flight.active.curve.solver, "joint")
        assert_allclose(flight.active.reference(.7), state, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
