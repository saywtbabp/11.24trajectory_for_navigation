"""Online midpoint interaction and full-route A* decisions."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from math import dist, hypot, radians

import numpy as np

from action_graph import ActionEdge, CostWeights, EdgeFeatures, astar
from door_geometry import (
    CONTACT_OFFSET, Leaf, departure, fit_leaves, polyline_length, polar_change,
    predict_departure, segment_clear, wall_clearance, MIN_GAP,
)
from flight_corridors import FLIGHT_MARGIN, certify_controls
from scene import CAGE_RADIUS

PRIOR_FORCE = .45
PUSH_TRIAL_PROGRESS = radians(2.5)
PUSH_ACCEL_LIMIT_MPS2 = 3.2


@dataclass
class Attempt:
    leaf: Leaf
    started: float
    anchor_position: np.ndarray | None = None
    anchor_angle: float = 0.
    last_scan: float = 0.
    force: float = PRIOR_FORCE
    measured_force: float = 0.
    in_contact: bool = False
    source: str = "range"
    load_window: deque = field(default_factory=deque)
    polar_samples: int = 0

    def observe(self, position, readings, reaction, t, *, update_forecast=True):
        current = self.leaf
        load = max(0., float(reaction @ current.tangent))
        separation = float((current.hinge - position) @ current.tangent)
        near = .08 < separation < .35
        loaded = near and load > .13
        self.in_contact = loaded
        if self.anchor_position is None and loaded:
            self.anchor_position = position.copy()
            self.anchor_angle = current.angle
        valid = next((leaf for leaf in readings if leaf.side == current.side), None)
        if len(readings) == 1 and readings[0].side != current.side:
            valid = readings[0]
        if valid is not None:
            jump = valid.angle - current.angle
            if valid.side != current.side or -radians(9) < jump < radians(48) or (
                    jump > 0 and valid.support >= 24):
                angle = (valid.angle if valid.side != current.side or jump > radians(12)
                         else .65*valid.angle + .35*current.angle)
                self.leaf = replace(valid, angle=angle)
                self.anchor_position = position.copy() if loaded else None
                self.anchor_angle = angle
                self.last_scan = t
                self.source = "range+imu"
        if self.last_scan < t and loaded and self.anchor_position is not None:
            first = np.linalg.norm(self.anchor_position-current.hinge)
            last = np.linalg.norm(position-current.hinge)
            if abs(last-first) <= .12:
                estimate = self.anchor_angle + polar_change(self.anchor_position, position, current)
                angle = float(np.clip(estimate, current.angle-radians(1), current.angle+radians(4)))
                self.leaf = replace(current, angle=float(np.clip(angle, 0., np.pi/2)))
                self.source = "contact odometry"
                self.polar_samples += 1
            else:
                self.source = "held angle"
        elif self.last_scan < t:
            self.source = "held angle"
        self.load_window.append((t, self.leaf.angle, load if loaded else 0.))
        while self.load_window and t - self.load_window[0][0] > 1.05:
            self.load_window.popleft()
        # An impact can produce a larger residual than subsequent pushing.
        # Update the work forecast only after a mostly loaded short window.
        recent = [force for stamp, _, force in self.load_window if t-stamp < .35]
        self.measured_force = float(np.median(
            [force for stamp, _, force in self.load_window if t-stamp < .18]))
        if update_forecast and sum(force > 0. for force in recent) >= 5:
            self.force = .65*self.force + .35*float(np.median(recent))

    @property
    def stalled(self):
        if len(self.load_window) < 19:
            return False
        angles = [point[1] for point in self.load_window]
        loads = [point[2] for point in self.load_window]
        return np.median(loads) > .55 and max(angles) - min(angles) < radians(2.)


@dataclass
class ForceTrial:
    started: float
    angle: float
    initial_force: float
    ramp: float
    target: float
    latest: float
    last_contact: float
    peak: float
    drive: float
    acceleration: float
    output_limited: bool = False

    def advance(self, attempt, t, output_limited):
        measured = attempt.measured_force
        self.peak = max(self.peak, measured)
        if attempt.in_contact:
            self.last_contact = t
        self.target = self.initial_force+self.ramp*(t-self.started)
        error = self.target-measured
        if attempt.in_contact and (not output_limited or error < 0.):
            self.drive = float(np.clip(self.drive + 1.65*error*(t-self.latest), 0., PUSH_ACCEL_LIMIT_MPS2))
        self.latest = t
        correction = .35*error if error >= 0. else .70*error
        self.acceleration = (float(np.clip(self.drive+correction, 0., PUSH_ACCEL_LIMIT_MPS2))
                             if attempt.in_contact else 0.)
        self.output_limited = output_limited or self.acceleration >= PUSH_ACCEL_LIMIT_MPS2

        if attempt.source == "range+imu" and attempt.leaf.angle-self.angle >= PUSH_TRIAL_PROGRESS:
            return "progress"
        if t-self.last_contact > .40:
            return "contact_lost"
        return None


class Navigator:
    def __init__(self, arena, *, effort_weight, force_ramp=1.2,
                 check=False, mode="auto", show_routes=False, smooth_paths=False):
        self.arena = arena
        self.weights = CostWeights(effort_weight)
        self.check = check
        self.mode = mode
        self.show_routes = show_routes
        self.smooth_paths = smooth_paths
        self.gates = {gate.name: gate for gate in arena.gates}
        self.wall_boxes = arena.boxes(.095)  # Only masks wall returns in perception.
        self.nodes = {"goal": tuple(arena.goal)}
        self.static_edges = []
        self.panels = {}
        self.disabled = set()
        self.compared = set()
        self.precontact_rechecked = set()
        self.flight_handoff = None
        self.finished = set()
        self.events = []
        self.metrics = None
        self.route = ()
        self.route_edges = ()
        self.phase = "fly"
        self.active = None
        self.attempt = None
        self.trial = None
        if not 0. < force_ramp or not np.isfinite(force_ramp):
            raise ValueError("force ramp must be finite and positive")
        self.force_ramp = force_ramp
        self.pending = None
        self.path = []
        self.path_revision = 0
        self.retreat_bypass_revision = None
        self.crossing_left = None
        self.target = arena.start.copy()
        self.last_search = -1.
        self._build_graph()

    def event(self, t, kind, gate, **values):
        self.events.append(dict(t=round(t, 3), event=kind, door=gate, **values))

    def room(self, xy):
        return int(np.searchsorted(self.arena.partitions, xy[0], side="right"))

    def _portal(self, name, x, y, s):
        self.nodes[name+":L"] = (x-s, y)
        self.nodes[name+":R"] = (x+s, y)

    def _build_graph(self):
        for gate in self.arena.gates:
            self._portal(gate.name+":door", gate.x, gate.middle, 1.12)
            self.nodes[gate.name+":door:R"] = (gate.x+2.55, gate.middle)
            self._portal(gate.name+":bypass", gate.x, gate.bypass_middle, .95)
            key = gate.name+":bypass"
            self.static_edges.append(self._flight(self.nodes, key+":L", key+":R",
                                                  "bypass", gate.name))
        for a, first in self.nodes.items():
            for b, second in self.nodes.items():
                if a != b and self.room(first) == self.room(second):
                    self.static_edges.append(self._flight(self.nodes, a, b))

    @staticmethod
    def _flight(nodes, a, b, action="fly", door=None):
        return ActionEdge(a, b, action, EdgeFeatures(dist(nodes[a], nodes[b])),
                          door, (nodes[b],))

    def _door_edges(self, nodes, gate, start, measurements, attempt=None, *, allow_push=True):
        end = gate.name+":door:R"
        finish = np.asarray(nodes[end])
        p = np.asarray(nodes[start])
        if not measurements:
            return [ActionEdge(start, end, "inspect", EdgeFeatures(dist(p, finish)),
                               gate.name, (nodes[start],))]
        # Direct passage requires clearance for every remaining hinge hypothesis.
        if attempt is None and all(segment_clear(p, finish, leaf) for leaf in measurements):
            return [self._flight(nodes, start, end, "cross", gate.name)]
        edges = []
        for leaf in measurements:
            if attempt is not None:
                measured_departure = departure(p, finish, leaf)
                if measured_departure and attempt.source != "held angle":
                    length = polyline_length(p, measured_departure)
                    edges.append(ActionEdge(start, end, "release", EdgeFeatures(length), gate.name,
                                            tuple(map(tuple, measured_departure)), contact_ratio=.5))
                    continue
            if not allow_push:
                continue
            prediction = predict_departure(leaf, finish)
            if prediction is None:
                continue
            opening, exit_path = prediction
            approach = leaf.midpoint - .48*leaf.tangent
            if attempt is None and not segment_clear(p, approach, leaf):
                continue
            sweep = np.linspace(leaf.angle, opening.angle, 16)
            center_points = np.asarray([replace(leaf, angle=angle).center() for angle in sweep])
            if min(wall_clearance(center_points, leaf.gate)) < MIN_GAP-CAGE_RADIUS:
                continue
            start_path = (approach, leaf.center()) if attempt is None else ()
            delta = max(0., opening.angle-leaf.angle)
            arc = leaf.radius*delta
            center_arc = hypot(leaf.radius, CONTACT_OFFSET)*delta
            geometry = polyline_length(p, start_path) + center_arc
            geometry += polyline_length(opening.center(), exit_path)
            geometry = max(geometry, dist(p, finish))
            if self.trial is not None and attempt is self.attempt:
                resistance = attempt.measured_force
            else:
                resistance = attempt.force if attempt is not None else PRIOR_FORCE
            preview = (*start_path, *center_points, *exit_path)
            edges.append(ActionEdge(
                start, end, "push", EdgeFeatures(geometry-arc, arc, resistance), gate.name,
                (tuple(approach),), tuple(map(tuple, preview)), .5))
        return edges

    def _graph(self, position, allow_push=True):
        nodes = {**self.nodes, "current": tuple(position)}
        edges = self.static_edges.copy()
        for gate in self.arena.gates:
            name = gate.name
            if name in self.disabled or self.mode == "detour":
                continue
            if gate.x < position[0]-.05 and self.active != name:
                continue
            if name in self.finished:
                continue
            if self.active == name:
                edges.extend(self._door_edges(nodes, gate, "current", (self.attempt.leaf,),
                                              self.attempt, allow_push=allow_push))
            else:
                edges.extend(self._door_edges(nodes, gate, name+":door:L",
                                              self.panels.get(name, ())))
        if self.active is None:
            for key, xy in self.nodes.items():
                if self.room(xy) == self.room(position):
                    edges.append(self._flight(nodes, "current", key))
        else:
            gate = self.gates[self.active]
            nodes["retreat"] = (min(position[0], gate.x-.82), float(position[1]))
            edges.append(self._flight(nodes, "current", "retreat", "retreat", self.active))
            for key, xy in self.nodes.items():
                if self.room(xy) == self.room(nodes["retreat"]) and key != gate.name+":door:L":
                    edges.append(self._flight(nodes, "retreat", key))
        return nodes, edges

    def search(self, position, t, gate=None, allow_push=True):
        nodes, edges = self._graph(position, allow_push)
        result = astar(nodes, edges, "current", "goal", self.weights)
        if self.check:
            check = astar(nodes, edges, "current", "goal", self.weights, heuristic=False)
            assert (result is None) == (check is None)
            assert result is None or abs(result.total_cost-check.total_cost) < 1e-8
        self.route = self._route_points(nodes, result)
        self.route_edges = result.edges if result else ()
        if gate is not None:
            avoid = astar(nodes, [edge for edge in edges if edge.opening_id != gate.name
                                  or edge.action in ("bypass", "retreat")],
                          "current", "goal", self.weights)
            door = astar(nodes, [edge for edge in edges if edge.opening_id != gate.name
                                 or edge.action not in ("bypass", "retreat")],
                         "current", "goal", self.weights)
            self.metrics = self._metrics(avoid, door, gate.name)
            selected = next((edge.action for edge in result.edges
                             if edge.opening_id == gate.name), None) if result else None
            leaves = (self.attempt.leaf,) if self.active else self.panels.get(gate.name, ())
            kind = ("remaining" if self.active else
                    "precontact_update" if gate.name in self.compared else "comparison")
            details = dict(selected=selected, effort_weight=self.weights.effort_m_per_j,
                           angle_deg=np.rad2deg(leaves[0].angle) if leaves else None,
                           source=self.attempt.source if self.active else "range",
                           observed_load=self.attempt.measured_force if self.active else None,
                           force_target=self.trial.target if self.trial else None,
                           output_limited=self.trial.output_limited if self.trial else False,
                           decision_phase=self.phase,
                           hinge_options=len(leaves), continuation_blocked=not allow_push)
            if self.show_routes:
                details.update(door_route=self._route_points(nodes, door),
                               avoid_route=self._route_points(nodes, avoid),
                               decision_position=nodes["current"])
            self.event(t, kind, gate.name, **self.metrics, **details)
            self.compared.add(gate.name)
        if not result or not result.edges:
            return None
        first = result.edges[0]
        if len(result.edges) > 1 and first.action == "fly" and dist(position, nodes[first.target]) < .12:
            return result.edges[1]
        return first

    @staticmethod
    def _route_points(nodes, route):
        if not route:
            return ()
        return (nodes["current"], *(point for edge in route.edges
                                    for point in edge.preview or edge.waypoints))

    def _metrics(self, avoid, door, name):
        def length(result):
            if result is None:
                return None
            return sum(edge.features.geometry_m for edge in result.edges)

        def total_work(result):
            if result is None:
                return None
            return sum(edge.features.push_work_j for edge in result.edges)

        path_avoid, path_door = length(avoid), length(door)
        push = next((edge for edge in door.edges if edge.opening_id == name and edge.action == "push"),
                    None) if door else None
        return dict(path_avoid=path_avoid, path_door=path_door,
                    gain=None if path_avoid is None or path_door is None else path_avoid-path_door,
                    force=push.features.resistance_n if push else None,
                    remaining_arc=push.features.push_arc if push else None,
                    work=push.features.push_work_j if push else 0. if door else None,
                    door_route_work=total_work(door), avoid_route_work=total_work(avoid),
                    net=None if avoid is None or door is None else avoid.total_cost-door.total_cost,
                    avoid_cost=avoid.total_cost if avoid else None, door_cost=door.total_cost if door else None)

    def _target_path(self, phase, points, gate=None, *, after_crossing=()):
        self.phase, self.path = phase, [np.asarray(point) for point in points]
        self.crossing_left = len(self.path) if after_crossing else None
        self.path.extend(np.asarray(point) for point in after_crossing)
        self.path_revision += 1
        self.pending = gate
        self.target = self.path[0]
        return self.target

    def _complete_path(self, t):
        self.crossing_left = None
        if self.phase == "release":
            self.event(t, "cross_after_push", self.active, angle_deg=np.rad2deg(self.attempt.leaf.angle),
                       odometry_frames=self.attempt.polar_samples)
            if self.smooth_paths:
                self.panels[self.active] = (self.attempt.leaf,)
            self.finished.add(self.active)
            self.attempt = self.active = None
        if self.phase == "retreat":
            self.attempt = self.active = None
        if self.phase in ("cross", "bypass") and self.pending:
            self.event(t, self.phase, self.pending)
            self.finished.add(self.pending)
        if self.phase == "approach":
            if self.smooth_paths:
                self.attempt.started = t
            self.phase = "push"
            return
        self.phase, self.pending = "fly", None

    @staticmethod
    def flight_waypoint_roles(phase, count, *, suffix_count=0):
        """Identify movable free-flight transitions in the constructed action path."""
        action_count = max(0, count-suffix_count)
        if phase in ("fly", "approach"):
            # Lead points precede the observation or push preparation anchor.
            return ("transition",)*max(0, count-1) + ("fixed",)*min(1, count)
        if phase in ("bypass", "release"):
            # The suffix belongs to the following action. Geometry must also
            # certify release before its transitions can move.
            return ("transition",)*action_count + ("fixed",)*suffix_count
        return ("fixed",)*count

    def flight_leaves(self, phase):
        leaves = dict(self.panels)
        if self.attempt is not None and phase != "approach":
            leaves[self.active] = (self.attempt.leaf,)
        return tuple(leaf for name in sorted(leaves) for leaf in leaves[name])

    def _can_handoff(self, t, phase, points, *, retreat_bypass=False, position=None, alternatives=(), suffix_count=0):
        if self.flight_handoff is None:
            return True
        leaves = self.flight_leaves(phase)
        if (phase == "release" and position is not None and self.attempt is not None
                and not certify_controls(position.reshape(1, 2), self.arena, leaves, margin=FLIGHT_MARGIN)):
            # Contact needs a separating motion before a free-flight corridor can start.
            target = np.asarray(points[0])
            other_leaves = tuple(leaf for leaf in leaves if leaf.gate.name != self.active)
            return (segment_clear(position, target, self.attempt.leaf, separating=True)
                    and certify_controls(np.asarray((position, target)), self.arena,
                                         other_leaves, margin=FLIGHT_MARGIN))
        return self.flight_handoff(t, phase, points, self.path_revision+1, leaves,
                                   retreat_bypass, alternatives, suffix_count)

    def _crossing_path(self, edge, prefix=(), *, position=None, handoff_time=None, retreat_bypass=False,
                       other_prefixes=()):
        points = (*prefix, *edge.waypoints)
        alternatives = tuple((*other, *edge.waypoints) for other in other_prefixes)
        suffix = ()
        if self.smooth_paths:
            for current, following in zip(self.route_edges, self.route_edges[1:]):
                if current is edge and current.target == following.source and following.action == "fly":
                    suffix = following.waypoints
                    break
        if edge.action == "release":
            original = points
            points = self._smooth_departure(points, suffix, position)
            if len(points) != len(original) or any(dist(a, b) > .015 for a, b in zip(points, original)):
                alternatives = (original,)
        elif edge.action == "cross" and position is not None:
            gate = self.gates[edge.opening_id]
            for offset in (-.52, .42):
                entry = (gate.x+offset, gate.middle)
                if entry[0] > position[0]+.20:
                    alternatives += ((entry, *edge.waypoints),)
        elif edge.action == "bypass" and position is not None and len(prefix) in (1, 2) and suffix:
            variants = [path for candidate in (points, *alternatives)
                        for path in self._bypass_paths(candidate, suffix, position, self.gates[edge.opening_id])]
            if retreat_bypass:
                points, *alternatives = variants
            else:
                alternatives = tuple(variants[:-1])
        route = [*points, *suffix]
        if handoff_time is not None and not self._can_handoff(
                handoff_time, edge.action, route, retreat_bypass=retreat_bypass, position=position,
                alternatives=tuple((*candidate, *suffix) for candidate in alternatives),
                suffix_count=len(suffix)):
            return None
        return self._target_path(edge.action, route[:len(route)-len(suffix)], edge.opening_id,
                                 after_crossing=suffix)

    def _join_visible_door(self, position, velocity, gate, t, *, allow_bypass=False):
        if self.phase != "fly" or len(self.path) != 1 or len(self.route_edges) < 2:
            return None
        approach, action = self.route_edges[:2]
        if approach.action != "fly" or action.opening_id != gate.name:
            return None
        leaves = tuple(leaf for hypotheses in self.panels.values() for leaf in hypotheses)
        if action.action == "bypass" and allow_bypass:
            points = np.asarray((position, *approach.waypoints, *action.waypoints))
            legs = np.stack((points[:-1], points[1:]), axis=1)
            if certify_controls(legs, self.arena, leaves, margin=FLIGHT_MARGIN):
                return self._crossing_path(action, approach.waypoints, position=position, handoff_time=t)
        if dist(self.path[0], approach.waypoints[0]) >= .015:
            return None
        if action.action == "push":
            staging = np.asarray(action.waypoints[0])
            # An ambiguous closed door can still have one usable staging point.
            if any(dist(staging, leaf.midpoint-.48*leaf.tangent) > .065
                   for leaf in self.panels[gate.name]):
                return None
            heading = staging-position
            distance = np.linalg.norm(heading)
            speed = np.linalg.norm(velocity)
            if speed > .55 and (distance < 1.6 or velocity @ heading < .90*speed*distance):
                return None
            if certify_controls(np.asarray((position, staging)), self.arena, leaves,
                                margin=FLIGHT_MARGIN):
                return self._start_approach(action, t, position, velocity)
        if action.action != "cross":
            return None
        # Aim for the opening itself when the diagonal has enough space.
        # Keep the conservative entry waypoint as a fallback.
        exit_point = np.asarray(action.waypoints[0])
        prefixes = []
        for entry in ((gate.x+.42, gate.middle), (gate.x-.52, gate.middle), self.path[0]):
            legs = np.asarray(((position, entry), (entry, exit_point)))
            if certify_controls(legs, self.arena, leaves, margin=FLIGHT_MARGIN):
                prefixes.append((entry,))
        if not prefixes:
            return self._crossing_path(action, self.path, handoff_time=t)
        return self._crossing_path(action, prefixes[0], handoff_time=t, other_prefixes=prefixes[1:])

    @staticmethod
    def _lead_paths(position, velocity, target):
        delta = np.asarray(target)-position
        length = np.linalg.norm(delta)
        speed = np.linalg.norm(velocity)
        if length < .90 or speed < .30:
            return ()
        along = delta/length
        sideways = velocity-(velocity @ along)*along
        if np.linalg.norm(sideways) < .15:
            return ()
        origin = position + min(.75, .35*length)*along
        return tuple((origin + seconds*sideways, target) for seconds in (.25, .55))

    def _start_approach(self, edge, t, position, velocity):
        route = list(edge.waypoints)
        alternatives = self._lead_paths(position, velocity, route[-1])
        if not self._can_handoff(t, "approach", route, alternatives=alternatives):
            return None
        leaf = min(self.panels[edge.opening_id], key=lambda item:
                   dist(item.midpoint-.48*item.tangent, edge.waypoints[0]))
        self.active = edge.opening_id
        self.attempt = Attempt(leaf, t, last_scan=t)
        self.last_search = t
        self.event(t, "push", self.active, angle_deg=np.rad2deg(leaf.angle),
                   hinge_options=len(self.panels[self.active]), midpoint=.5)
        return self._target_path("approach", route, self.active)

    def _bypass_paths(self, points, suffix, position, gate):
        entry = np.asarray(points[-2])
        prefix = points[:-2]
        turn = prefix[-1] if prefix else position
        # Start turning before the jamb and bias the exit toward the outgoing leg.
        exit_side = np.sign(suffix[0][1]-gate.bypass_middle)
        leaves = self.flight_leaves("bypass")
        for advance, lead, exit_x, exit_y in ((.40, .65, .52, .07), (.18, .28, .42, 0.)):
            lead_in = entry + (advance, np.clip(turn[1]-entry[1], -lead, lead))
            exit_point = np.array((gate.x+exit_x, gate.bypass_middle+exit_y*exit_side))
            path = np.asarray((position, *prefix, lead_in, exit_point, suffix[0]))
            legs = np.stack((path[:-1], path[1:]), axis=1)
            if certify_controls(legs, self.arena, leaves, margin=FLIGHT_MARGIN):
                yield (*prefix, lead_in, exit_point)
        yield points

    def _smooth_departure(self, points, suffix, position):
        # Keep the separating leg. Once clear of the leaf, try joining the
        # same-room route directly instead of turning back to the door centerline.
        if not suffix:
            return points
        leaves = {**self.panels, self.active: (self.attempt.leaf,)}
        obstacles = tuple(leaf for hypotheses in leaves.values() for leaf in hypotheses)
        if len(points) == 1 and position is not None:
            finish = np.asarray(points[0])
            if certify_controls(np.asarray((position, finish)), self.arena, obstacles, margin=FLIGHT_MARGIN):
                return points
            # A nearly open panel can still clip the straight chord near its
            # tip. Guide the curve around that tip before joining the exit.
            for offset in (.10, .18, .27):
                lead = .45*position + .55*finish + (0., self.attempt.leaf.side*offset)
                legs = np.asarray(((position, lead), (lead, finish)))
                if certify_controls(legs, self.arena, obstacles, margin=FLIGHT_MARGIN):
                    return (lead, finish)
            return points
        if len(points) != 3:
            return points
        release, bridge = map(np.asarray, points[:2])
        # Prefer less lateral travel before joining the next confirmed leg.
        # Keep the original lead as a fallback if the leaf blocks the shortcut.
        offsets = ((.65, .12), (.65, .07), (.60, .07), (.55, .07), (.55, .04),
                   (.55, 0.), (.40, 0.), (.30, 0.), (.15, 0.), (0., 0.))
        if position is None:
            offsets = ((0., 0.),)
        for fraction in (0., .25, .5, .75, 1.):
            for advance, trim in offsets:
                lead = release + (advance, trim*(suffix[0][1]-release[1]))
                if advance and not (segment_clear(position, lead, self.attempt.leaf, separating=True)
                                    or segment_clear(position, lead, self.attempt.leaf)):
                    continue
                join = lead + fraction*(bridge-lead)
                legs = (np.asarray((lead, join)), np.asarray((join, suffix[0])))
                if certify_controls(np.asarray(legs), self.arena, obstacles, margin=FLIGHT_MARGIN):
                    return (lead, join) if fraction else (lead,)
        return points

    def release_alternatives(self, position):
        """Generate small post-separation adjustments without changing the route endpoint."""
        if self.phase != "release" or not self.crossing_left or len(self.path) < 2:
            return ()
        entry = self.path[0]
        direction = self.path[1]-entry
        length = np.linalg.norm(direction)
        if length < .4 or np.linalg.norm(entry-position) < .18:
            return ()
        along = direction/length
        sideways = np.array((-along[1], along[0]))
        return tuple((entry+forward*along+lateral*sideways, *self.path[1:])
                     for forward, lateral in ((.15, 0.), (.25, .10), (.25, -.10)))

    def _stop(self, position, t, why):
        attempt = self.attempt
        resistance = attempt.measured_force if self.trial is not None else attempt.force
        if self.trial is not None:
            self._finish_trial(t, why)
        self.disabled.add(self.active)
        self.event(t, "retreat", self.active, reason=why, force=resistance,
                   observed_load=attempt.measured_force, angle_deg=np.rad2deg(attempt.leaf.angle),
                   net=self.metrics["net"] if self.metrics else None,
                   odometry_frames=attempt.polar_samples)
        gate = self.gates[self.active]
        retreat = (min(position[0], gate.x-.82), float(position[1]))
        return self._target_path("retreat", (retreat,))

    def _join_retreat_to_bypass(self, position, t):
        # Once the cage clears the leaf, plan both legs in one flight window.
        anchor = self.path[0]
        approach = position - anchor
        approach_length = np.linalg.norm(approach)
        if approach_length <= .12:
            return None
        leaves = {**self.panels, self.active: (self.attempt.leaf,)}
        obstacles = tuple(leaf for hypotheses in leaves.values() for leaf in hypotheses)
        if not certify_controls(np.asarray((position, anchor)), self.arena, obstacles,
                                margin=FLIGHT_MARGIN):
            return None
        self.search(position, t)
        if len(self.route_edges) < 3:
            return None
        retreat, connection, bypass = self.route_edges[:3]
        if (retreat.action != "retreat" or connection.action != "fly"
                or bypass.action != "bypass" or bypass.opening_id != self.active):
            return None
        onward = np.asarray(connection.waypoints[0]) - anchor
        onward_length = np.linalg.norm(onward)
        turn = anchor + min(.30, .45*approach_length)*approach/approach_length
        if onward_length > .12:
            turn += min(.30, .45*onward_length)*onward/onward_length
        prefixes = []
        for corner in (turn, anchor):
            # Cutting the corner is optional. Certify the incoming and outgoing
            # legs against the current leaf and wall surfaces before committing.
            prefix = (corner, *connection.waypoints)
            points = np.asarray((position, *prefix, *bypass.waypoints))
            legs = np.stack((points[:-1], points[1:]), axis=1)
            if certify_controls(legs, self.arena, obstacles, margin=FLIGHT_MARGIN):
                prefixes.append(prefix)
        if not prefixes:
            return None
        target = self._crossing_path(bypass, prefixes[0], position=position, handoff_time=t,
                                     retreat_bypass=True, other_prefixes=prefixes[1:])
        if target is not None:
            self.panels[self.active] = leaves[self.active]
            self.attempt = self.active = None
            self.retreat_bypass_revision = self.path_revision
        return target

    def _start_trial(self, t, planar_command):
        force = self.attempt.measured_force
        drive = float(np.clip(np.dot(planar_command, self.attempt.leaf.tangent), 0., PUSH_ACCEL_LIMIT_MPS2))
        self.trial = ForceTrial(
            started=t, angle=self.attempt.leaf.angle, initial_force=force,
            ramp=self.force_ramp, target=force,
            latest=t, last_contact=t, peak=force, drive=drive, acceleration=drive)
        self.phase = "boost"
        self.event(t, "boost_start", self.active, force=force, force_target=force, net=self.metrics["net"])

    def _finish_trial(self, t, outcome):
        trial = self.trial
        self.event(t, "boost_end", self.active, reason=outcome,
                   force=self.attempt.measured_force, peak_force=trial.peak,
                   force_target=trial.target, output_limited=trial.output_limited,
                   net=self.metrics["net"] if self.metrics else None,
                   progress_deg=np.rad2deg(self.attempt.leaf.angle-trial.angle))
        self.trial = None

    def _release(self, edge, t, position):
        target = self._crossing_path(edge, position=position, handoff_time=t)
        if target is None:
            return self._stop(position, t, "release_path_unavailable")
        if self.trial is not None:
            self._finish_trial(t, "progress")
        attempt = self.attempt
        self.event(t, "release", self.active, angle_deg=np.rad2deg(attempt.leaf.angle),
                   source=attempt.source)
        return target

    def repair_door_path(self, position, t):
        """Change a blocked door approach or departure using the current leaf."""
        if self.phase == "release" and self.attempt is not None and self.attempt.source != "held angle":
            finish = np.asarray(self.nodes[self.active+":door:R"])
            path = departure(position, finish, self.attempt.leaf)
            if path:
                suffix = tuple(self.path[self.crossing_left:]) if self.crossing_left is not None else ()
                path = self._smooth_departure(path, suffix, position)
                if len(path)+len(suffix) != len(self.path) or any(
                        dist(a, b) > .06 for a, b in zip(path, self.path)):
                    if self._can_handoff(t, "release", (*path, *suffix), position=position, suffix_count=len(suffix)):
                        return self._target_path("release", path, self.active, after_crossing=suffix)
        if self.phase == "cross" and self.pending is not None:
            gate = self.gates[self.pending]
            leaves = self.panels.get(gate.name, ())
            entry = np.asarray(self.nodes[gate.name+":door:L"])
            finish = np.asarray(self.nodes[gate.name+":door:R"])
            if position[0] < gate.x-.40 and leaves and not all(
                    segment_clear(entry, finish, leaf) for leaf in leaves):
                return self._target_path("fly", (entry,))
        return self.target

    def decide(self, position, velocity, points, reaction, t, *,
               planar_command=(0., 0.), output_limited=False, fly_through=False, flight_waypoint=None):
        tracked_door = self.active or (self.pending if self.smooth_paths and self.phase == "cross" else None)
        near_gate = next((gate for gate in self.arena.gates
                          if gate.x-3.3 <= position[0] <= gate.x+(1.5 if gate.name == tracked_door else 0.)
                          and gate.name not in self.finished), None)
        readings = ()
        if near_gate is not None and near_gate.name not in self.disabled:
            readings = fit_leaves(near_gate, points, self.wall_boxes)
            if readings and self.active is None:
                self.panels[near_gate.name] = readings
            if readings and self.active is None and near_gate.name not in self.compared:
                self.search(position, t, near_gate)
                if self.smooth_paths:
                    joined = self._join_visible_door(position, velocity, near_gate, t)
                    if joined is not None:
                        return joined
            elif (readings and self.active is None and self.smooth_paths
                  and self.phase == "fly" and len(self.path) == 1
                  and near_gate.name not in self.precontact_rechecked
                  and dist(self.path[0], self.nodes[near_gate.name+":door:L"]) < .015
                  and dist(position, self.path[0]) <= max(1., np.linalg.norm(velocity) + .25)):
                # Re-evaluate the approach while the previous trajectory still
                # has room to turn or stop short of its door-front endpoint.
                self.precontact_rechecked.add(near_gate.name)
                self.search(position, t, near_gate)
                joined = self._join_visible_door(position, velocity, near_gate, t, allow_bypass=True)
                if joined is not None:
                    return joined
        if self.smooth_paths and self.phase == "release" and readings and near_gate.name == self.active:
            self.attempt.observe(position, readings, reaction, t, update_forecast=False)
        if self.path:
            if self.smooth_paths and self.phase == "retreat":
                joined = self._join_retreat_to_bypass(position, t)
                if joined is not None:
                    return joined
            speed = np.linalg.norm(velocity)
            tight_target = self.phase == "release" or (self.phase == "approach" and len(self.path) == 1)
            threshold = .065 if tight_target else .12
            flying_through = fly_through and len(self.path) > 1
            settle_speed = .12 if self.smooth_paths and self.phase == "approach" else .23
            execution_point = self.path[0] if flight_waypoint is None else flight_waypoint
            if dist(position, execution_point) > threshold or (speed > settle_speed and not flying_through):
                return self.target
            self.path.pop(0)
            if self.crossing_left is not None:
                self.crossing_left -= 1
            if not self.path or self.crossing_left == 0:
                self._complete_path(t)
            if self.path:
                self.target = self.path[0]
                return self.target
        if self.active is not None:
            attempt = self.attempt
            gate = self.gates[self.active]
            attempt.observe(position, readings, reaction, t, update_forecast=self.trial is None)
            if t-attempt.last_scan > 1.5 or (self.trial is None and t-attempt.started > 14.):
                why = "angle_unobservable" if t-attempt.last_scan > 1.5 else "push_timeout"
                return self._stop(position, t, why)
            if self.trial is not None:
                outcome = self.trial.advance(attempt, t, output_limited)
                if outcome == "progress":
                    self._finish_trial(t, outcome)
                    attempt.load_window.clear()
                    attempt.started = t
                    self.phase = "push"
                    self.last_search = -1.
                elif outcome is not None:
                    edge = self.search(position, t, gate, allow_push=False)
                    if edge is not None and edge.action == "release":
                        return self._release(edge, t, position)
                    return self._stop(position, t, outcome)
            stalled = self.trial is None and attempt.stalled
            if stalled or t-self.last_search >= .19:
                self.last_search = t
                edge = self.search(position, t, gate)
                if edge is None or edge.action == "retreat":
                    return self._stop(position, t, "remaining_cost")
                if edge.action == "release":
                    return self._release(edge, t, position)
                if stalled and self.metrics["net"] == 0:
                    return self._stop(position, t, "remaining_cost")
                if stalled and attempt.in_contact and t-attempt.last_scan <= .3:
                    self._start_trial(t, planar_command)
            self.target = attempt.leaf.midpoint - .045*attempt.leaf.tangent
            return self.target
        gate = (near_gate if near_gate is not None and near_gate.name in self.panels
                and near_gate.name not in self.disabled and near_gate.name not in self.finished
                and self.mode != "detour" else None)
        edge = self.search(position, t, gate)
        if edge is None:
            self.phase = "no_path"
            return position
        if edge.action == "inspect":
            self.disabled.add(edge.opening_id)
            self.event(t, "unobserved", edge.opening_id)
            return position
        if edge.action == "push":
            target = self._start_approach(edge, t, position, velocity)
            return target if target is not None else position
        if self.smooth_paths and edge.action == "fly" and len(self.route_edges) > 1:
            first, following = self.route_edges[:2]
            if edge is first and following.action in ("cross", "bypass"):
                target = self._crossing_path(following, edge.waypoints, position=position, handoff_time=t)
                if target is not None:
                    return target
        if edge.action in ("cross", "bypass"):
            target = self._crossing_path(edge, position=position, handoff_time=t)
            return target if target is not None else position
        if edge.action == "fly" and self.flight_handoff is not None:
            route = list(edge.waypoints)
            alternatives = self._lead_paths(position, velocity, route[-1])
            if alternatives:
                if self._can_handoff(t, "fly", route, alternatives=alternatives):
                    return self._target_path("fly", route, edge.opening_id)
                return position
        return self._target_path(edge.action, edge.waypoints, edge.opening_id)


