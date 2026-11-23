"""Convex free-flight corridors against the simulated wall and leaf surfaces."""
from dataclasses import dataclass

import numpy as np

from door_geometry import CLEARANCE
from scene import CAGE_RADIUS, DOOR_HALF, WALL_HALF


FLIGHT_MARGIN = .045
_TOLERANCE = 1e-9
_LEAF_START = .065  # Distance from hinge to the near end of the MuJoCo leaf.


@dataclass(frozen=True)
class _Rectangles:
    centers: np.ndarray
    axes: np.ndarray
    half_extents: np.ndarray

    def select(self, mask):
        return _Rectangles(self.centers[mask], self.axes[mask], self.half_extents[mask])


def _obstacles(arena, leaves):
    walls = np.asarray(arena.walls, dtype=float).reshape(-1, 2, 2)
    edges = walls[:, 1] - walls[:, 0]
    lengths = np.linalg.norm(edges, axis=1)
    present = lengths > _TOLERANCE
    directions = edges[present] / lengths[present, None]
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    centers = list(walls[present].mean(axis=1))
    axes = list(np.stack((directions, normals), axis=1))
    half_extents = [(length / 2, WALL_HALF) for length in lengths[present]]
    for leaf in leaves:
        direction = leaf.direction
        length = leaf.gate.length
        centers.append(leaf.hinge + (length + _LEAF_START) / 2 * direction)
        axes.append((direction, leaf.tangent))
        half_extents.append(((length - _LEAF_START) / 2, DOOR_HALF))
    return _Rectangles(np.asarray(centers).reshape(-1, 2),
                       np.asarray(axes).reshape(-1, 2, 2),
                       np.asarray(half_extents).reshape(-1, 2))


def sampled_clearance(points, arena, leaves):
    """Smallest cage clearance at supplied points, for ranking certified curves."""
    rectangles = _obstacles(arena, leaves)
    local = np.einsum("nij,tnj->tni", rectangles.axes, points[:, None]-rectangles.centers)
    outside = np.maximum(abs(local)-rectangles.half_extents, 0.)
    return float(np.linalg.norm(outside, axis=-1).min()) - CAGE_RADIUS


def _closest_segment(start, finish, rectangles):
    """Return distance and the local displacement from each box to the segment."""
    local = np.einsum("nij,nj->ni", rectangles.axes, start - rectangles.centers)
    motion = np.einsum("nij,j->ni", rectangles.axes, finish - start)
    extent = rectangles.half_extents

    # Squared distance to a box is piecewise quadratic along the segment.
    # Breakpoints are the crossings of its four face lines.
    crossings = np.concatenate((-extent-local, extent-local), axis=1)
    slopes = np.tile(motion, (1, 2))
    cuts = np.divide(crossings, slopes, out=np.zeros_like(crossings), where=slopes != 0.)
    count = len(local)
    breaks = np.sort(np.column_stack((np.zeros(count), np.ones(count), cuts.clip(0., 1.))), axis=1)
    left = breaks[:, :-1]
    right = breaks[:, 1:]
    mid = (left+right) / 2
    base = local[:, None, :]
    direction = motion[:, None, :]
    bounds = extent[:, None, :]
    middle = base + mid[:, :, None] * direction
    active = np.abs(middle) > bounds
    offset = base - np.sign(middle)*bounds
    numerator = -np.sum(active * direction * offset, axis=2)
    denominator = np.sum(active * direction**2, axis=2)
    time = np.divide(numerator, denominator, out=mid, where=denominator > 0.)
    closest = base + np.clip(time, left, right)[:, :, None] * direction
    displacement = closest - np.clip(closest, -bounds, bounds)
    squared = np.sum(displacement**2, axis=2)
    best = np.argmin(squared, axis=1)
    indices = np.arange(count)
    return np.sqrt(squared[indices, best]), displacement[indices, best]


def _cross(origin, point, other):
    first = point - origin
    second = other - origin
    return first[0]*second[1] - first[1]*second[0]


def _clip_polygon(vertices, normal, offset):
    clipped = []
    for start, finish in zip(vertices, np.roll(vertices, -1, axis=0)):
        start_error = start @ normal - offset
        finish_error = finish @ normal - offset
        if start_error <= 0.:
            clipped.append(start)
        if (start_error > 0.) != (finish_error > 0.):
            fraction = start_error / (start_error-finish_error)
            clipped.append(start + fraction*(finish-start))
    return np.asarray(clipped)


def _segment_corridor(start, finish, rectangles, bounds, clearance):
    distances, differences = _closest_segment(start, finish, rectangles)
    if np.any(distances < clearance - _TOLERANCE):
        return None

    # The closest points of two convex sets define a separating support plane.
    # Offset this support into free space so every point in the resulting
    # halfspace clears the raw obstacle by the cage radius and margin.
    local_away = differences / distances[:, None]
    normals = -np.einsum("ni,nij->nj", local_away, rectangles.axes)
    supports = np.sum(np.abs(local_away)*rectangles.half_extents, axis=1)
    offsets = np.sum(normals*rectangles.centers, axis=1) - supports - clearance
    xmin, ymin, xmax, ymax = bounds
    rows = [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)]
    limits = [xmax, -xmin, ymax, -ymin]
    if np.any(np.vstack((start, finish)) @ np.asarray(rows).T > limits):
        return None
    vertices = np.array(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
    # Nearby planes tend to make the farther ones redundant. Clipping keeps the
    # number of optimizer constraints small regardless of map length.
    for index in np.argsort(distances):
        normal, offset = normals[index], offsets[index]
        if np.all(vertices @ normal <= offset + _TOLERANCE):
            continue
        vertices = _clip_polygon(vertices, normal, offset)
        if not len(vertices):
            return None
        rows.append(normal)
        limits.append(offset)
    return np.asarray(rows), np.asarray(limits)


def _clearance(margin):
    if not np.isfinite(margin) or margin < 0.:
        raise ValueError("flight corridor margin must be finite and nonnegative")
    return CAGE_RADIUS + margin


def build_corridors(points, arena, leaves, *, margin=CLEARANCE):
    """Return (A, b) per segment for the center constraints A @ p <= b."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("corridor points must contain at least two finite planar positions")
    clearance = _clearance(margin)
    rectangles = _obstacles(arena, leaves)
    corridors = []
    for start, finish in zip(points[:-1], points[1:]):
        corridor = _segment_corridor(start, finish, rectangles, arena.bounds, clearance)
        if corridor is None:
            return None
        corridors.append(corridor)
    return corridors


def _convex_hull(points):
    points = [np.asarray(point) for point in sorted(set(map(tuple, points)))]
    if len(points) < 3:
        return np.asarray(points)
    chains = []
    for ordered in (points, reversed(points)):
        chain = []
        for point in ordered:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], point) <= 0.:
                chain.pop()
            chain.append(point)
        chains.extend(chain[:-1])
    return np.asarray(chains)


def _certify_hull(controls, rectangles, clearance, bounds):
    if np.any(controls < bounds[:2]) or np.any(controls > bounds[2:]):
        return False

    half_width = np.einsum("ni,nij->nj", rectangles.half_extents, np.abs(rectangles.axes))
    gap = np.maximum(rectangles.centers-half_width-controls.max(axis=0),
                     controls.min(axis=0)-rectangles.centers-half_width)
    candidates = np.linalg.norm(np.maximum(gap, 0.), axis=1) < clearance
    if not np.any(candidates):
        return True

    near = rectangles.select(candidates)
    hull = _convex_hull(controls)
    next_points = np.roll(hull, -1, axis=0)
    if len(hull) >= 3:
        # A rectangle entirely inside the hull can miss every hull edge.
        edges = next_points - hull
        to_centers = near.centers[:, None, :] - hull
        cross = edges[:, 0]*to_centers[:, :, 1] - edges[:, 1]*to_centers[:, :, 0]
        if np.any(np.all(cross >= -_TOLERANCE, axis=1)):
            return False
    for start, finish in zip(hull, next_points):
        distances, _ = _closest_segment(start, finish, near)
        if np.any(distances < clearance - _TOLERANCE):
            return False
    return True


def certify_controls(control_points, arena, leaves, *, margin=CLEARANCE):
    """Certify the convex hull of each supplied Bézier control polygon.

    A false result means the hull does not certify clearance. A curve can be
    split before checking its smaller hulls if a tighter certificate is needed.
    """
    controls = np.asarray(control_points, dtype=float)
    if controls.ndim == 2:
        controls = controls[None, ...]
    if (controls.ndim != 3 or controls.shape[-1] != 2 or controls.size == 0
            or not np.isfinite(controls).all()):
        return False
    clearance = _clearance(margin)
    rectangles = _obstacles(arena, leaves)
    return all(_certify_hull(piece, rectangles, clearance, arena.bounds) for piece in controls)
