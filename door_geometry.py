"""Door fitting and clearance against the original wall and leaf surfaces."""
from dataclasses import dataclass, replace
from math import atan2, ceil, cos, radians, sin

import numpy as np

from scene import CAGE_RADIUS, DOOR_HALF, WALL_HALF

CLEARANCE = .025
CONTACT_OFFSET = .22
MIN_GAP = 2 * CAGE_RADIUS + CLEARANCE


@dataclass(frozen=True)
class Leaf:
    gate: object
    side: int
    angle: float
    score: float = 0.
    support: int = 0

    @property
    def hinge(self):
        return np.array([self.gate.x, self.gate.low if self.side == 1 else self.gate.high])

    @property
    def direction(self):
        return np.array([sin(self.angle), self.side*cos(self.angle)])

    @property
    def tangent(self):
        return np.array([cos(self.angle), -self.side*sin(self.angle)])

    @property
    def radius(self):
        return self.gate.length / 2

    @property
    def midpoint(self):
        return self.hinge + self.radius*self.direction

    def center(self, offset=CONTACT_OFFSET):
        return self.midpoint - offset*self.tangent


ANGLES = np.deg2rad(np.arange(-2, 91))
AXES = {side: np.array([np.sin(ANGLES), side*np.cos(ANGLES)]) for side in (-1, 1)}


def fit_leaves(gate, points, wall_boxes):
    """Score both door-frame ends independently using only range points."""
    near = points[(abs(points[:, 0] - gate.x) < 2.)
                  & (points[:, 1] > gate.low - .1) & (points[:, 1] < gate.high + .1)]
    if not len(near):
        return ()
    in_wall = np.any(np.all((near[:, None] >= wall_boxes[:, :2]) &
                            (near[:, None] <= wall_boxes[:, 2:]), axis=2), axis=1)
    near = near[~in_wall]
    fits = []
    for side in (-1, 1):
        hinge = [gate.x, gate.low if side == 1 else gate.high]
        d = near - hinge
        a = AXES[side]
        along = d @ a
        transverse = abs(d[:, 0, None]*a[1] - d[:, 1, None]*a[0])
        good = (along > .15) & (along < gate.length + .035) & (transverse < .055)
        weight = np.sum(good * along * (1 - np.minimum(transverse/.065, 1)), axis=0)
        index = int(np.argmax(weight))
        support = int(good[:, index].sum())
        if support >= 8:
            fits.append(Leaf(gate, side, max(0., float(ANGLES[index])), float(weight[index]), support))
    fits.sort(key=lambda item: item.score, reverse=True)
    if len(fits) == 2 and fits[1].score < .65 * fits[0].score:
        return (fits[0],)
    return tuple(fits)


def chord_gap(leaf):
    return 2 * leaf.gate.length * sin(leaf.angle / 2) - DOOR_HALF * sin(leaf.angle)


def wall_clearance(points, gate):
    """Distance to the two raw wall slabs adjacent to the doorway."""
    x, y = points.T
    dx = np.maximum(abs(x-gate.x) - WALL_HALF, 0.)
    dy = np.maximum(np.minimum(y-gate.low, gate.high-y), 0.)
    return np.hypot(dx, dy)


def leaf_clearance(points, leaf):
    delta = points - leaf.hinge
    along = delta @ leaf.direction
    normal = delta @ leaf.tangent
    dr = np.maximum.reduce((.065 - along, along - leaf.gate.length, np.zeros(len(points))))
    dn = np.maximum(abs(normal) - DOOR_HALF, 0.)
    return np.hypot(dr, dn)


def segment_clear(start, finish, leaf, *, separating=False):
    steps = 1 + max(2, ceil(np.linalg.norm(finish-start) / .035))
    points = np.linspace(start, finish, steps)
    if min(wall_clearance(points, leaf.gate)) < CAGE_RADIUS + CLEARANCE:
        return False
    gap = leaf_clearance(points, leaf)
    if separating:
        # The initial sample can be touching the leaf. The first leg must
        # increase the signed separation and end clear of the actual leaf.
        normal = (points - leaf.hinge) @ leaf.tangent
        return normal[0] < -.09 and normal[-1] < normal[0] and gap[-1] >= CAGE_RADIUS + CLEARANCE
    return min(gap) >= CAGE_RADIUS + CLEARANCE


def departure(start, finish, leaf):
    if chord_gap(leaf) < MIN_GAP:
        return None
    if segment_clear(start, finish, leaf):
        return (finish,)
    gate = leaf.gate
    # Begin lateral motion only once the center has cleared the jamb plane.
    if start[0] < gate.x + .39:
        return None
    tip = leaf.hinge + gate.length*leaf.direction
    lane_y = leaf.side * (max(leaf.side*finish[1], leaf.side*tip[1]) + .40)
    bridge = np.array([max(finish[0], tip[0]+.45), lane_y])
    for move in (0., .07, .15, .28):
        release = np.array([max(start[0], gate.x+.40+move), lane_y])
        if (segment_clear(start, release, leaf, separating=True)
                and segment_clear(release, bridge, leaf)
                and segment_clear(bridge, finish, leaf)):
            return (release, bridge, finish)
    return None


def predict_departure(leaf, finish):
    """First feasible one-degree opening step for a midpoint push."""
    for degrees in range(ceil(np.rad2deg(leaf.angle)), 91):
        candidate = replace(leaf, angle=radians(degrees))
        path = departure(candidate.center(), finish, candidate)
        if path:
            return candidate, path
    return None


def polyline_length(start, path):
    return sum(float(np.linalg.norm(b-a)) for a, b in zip((start, *path), path))


def polar_change(anchor, position, leaf):
    first, last = anchor - leaf.hinge, position - leaf.hinge
    cross = first[0]*last[1] - first[1]*last[0]
    return -leaf.side * atan2(cross, first @ last)

