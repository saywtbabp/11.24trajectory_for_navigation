"""Coarse map and independent MuJoCo test fixtures."""
from dataclasses import dataclass

import numpy as np

WALL_HALF = .05
DOOR_HALF = .025
CAGE_RADIUS = .266
END_GAP = .03


@dataclass(frozen=True)
class Gate:
    name: str
    x: float
    low: float
    high: float
    bypass_low: float
    bypass_high: float

    @property
    def length(self):
        return self.high - self.low - END_GAP

    @property
    def middle(self):
        return (self.low + self.high) / 2

    @property
    def bypass_middle(self):
        return (self.bypass_low + self.bypass_high) / 2


@dataclass(frozen=True)
class Arena:
    gates: tuple
    bounds: tuple
    start: np.ndarray
    goal: np.ndarray

    @property
    def partitions(self):
        return [gate.x for gate in self.gates]

    @property
    def walls(self):
        left, bottom, right, top = self.bounds
        lines = [(left, bottom, right, bottom), (left, top, right, top),
                 (left, bottom, left, top), (right, bottom, right, top)]
        for gate in self.gates:
            x = gate.x
            first, second = sorted(((gate.low, gate.high), (gate.bypass_low, gate.bypass_high)))
            lines.extend(((x, bottom, x, first[0]),
                          (x, first[1], x, second[0]),
                          (x, second[1], x, top)))
        return np.asarray(lines, dtype=float)

    def boxes(self, pad):
        walls = self.walls
        return np.column_stack((np.minimum(walls[:, :2], walls[:, 2:]) - pad,
                                np.maximum(walls[:, :2], walls[:, 2:]) + pad))


def make_scene():
    # The navigator receives only the coarse map. The last four columns are
    # scene parameters (initial angle, hinge side, profile and display name).
    cases = (
        ('D1', 4.90, 6.40,  7.,  1, 'easy', 'door shortcut'),
        ('D2', -5.30, -3.80, 78., -1, 'easy', 'open door'),
        ('D3', 7.00, 8.60, 25.,  1, 'easy', 'weak door returns'),
        ('D4', -6.95, -5.45, 15., -1, 'effort_ramp', 'rising effort'),
        ('D5', 6.20, 7.80, 22., -1, 'binding', 'hidden travel stop'),
        ('D6', -8.60, -7.00, 25.,  1, 'very_stiff', 'high static load'),
        ('D7', 7.00, 8.60, 35., -1, 'snap_open', 'light door swings clear'),
        ('D8', -11.50, -10.00, 0., 1, 'easy', 'closed with ambiguous hinge'),
        ('D9', 12.20, 13.70, 25., -1, 'breakaway', 'opens after force ramp'),
    )
    gates = tuple(Gate(name, index * 8., -.90, .90, low, high)
                  for index, (name, low, high, *_) in enumerate(cases))
    fixtures = tuple((gate.name, gate.x, gate.low if side == 1 else gate.high,
                      gate.length, angle, profile, side)
                     for gate, (_, _, _, angle, side, profile, _) in zip(gates, cases))
    arena = Arena(gates, (-4., -12.1, gates[-1].x + 8., 13.8),
                  np.array([-3., 0.]), np.array([gates[-1].x + 6., 0.]))
    return arena, fixtures, {case[0]: case[-1] for case in cases}

