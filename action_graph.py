"""A* over directed flight and contact actions with interpretable edge costs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import heapq
from itertools import count
import math

@dataclass(frozen=True)
class CostWeights:
    effort_m_per_j: float

    def __post_init__(self):
        if not math.isfinite(self.effort_m_per_j) or self.effort_m_per_j < 0:
            raise ValueError("effort weight must be finite and nonnegative")


@dataclass(frozen=True)
class EdgeFeatures:
    # Distances and penalties are expressed in metres of equivalent flight.
    flight_distance: float
    push_arc: float = 0.
    resistance_n: float = 0.

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0
                     for value in (self.flight_distance, self.push_arc, self.resistance_n)):
            raise ValueError("edge distances and penalties must be finite and nonnegative")


    @property
    def push_work_j(self) -> float:
        return self.push_arc * self.resistance_n

    @property
    def geometry_m(self) -> float:
        return self.flight_distance + self.push_arc

    def cost(self, weights: CostWeights) -> float:
        return self.geometry_m + weights.effort_m_per_j * self.push_work_j

@dataclass(frozen=True)
class ActionEdge:
    source: str
    target: str
    action: str
    features: EdgeFeatures
    opening_id: str | None = None
    waypoints: tuple[tuple[float, float], ...] = ()
    preview: tuple[tuple[float, float], ...] = ()
    contact_ratio: float | None = None

    def cost(self, weights: CostWeights) -> float:
        return self.features.cost(weights)


@dataclass(frozen=True)
class SearchResult:
    nodes: tuple[str, ...]
    edges: tuple[ActionEdge, ...]
    total_cost: float
    expanded_nodes: int


def astar(nodes: Mapping[str, Sequence[float]], edges: Iterable[ActionEdge],
          start: str, goal: str, weights: CostWeights, *,
          heuristic: bool = True) -> SearchResult | None:
    """Return the least-cost directed route, or None if the goal is disconnected."""
    if start not in nodes or goal not in nodes:
        raise ValueError("start and goal must be graph nodes")
    if any(len(xy) != 2 or not all(math.isfinite(value) for value in xy) for xy in nodes.values()):
        raise ValueError("graph nodes require finite planar coordinates")

    adjacency = {node: [] for node in nodes}
    for edge in edges:
        if edge.source not in nodes or edge.target not in nodes:
            raise ValueError("edge endpoints must be graph nodes")
        cost = edge.cost(weights)
        # The straight-line heuristic requires each action to cost at least
        # the physical displacement between its endpoints.
        distance = math.dist(nodes[edge.source], nodes[edge.target])
        if not math.isfinite(cost) or cost + 1e-9 < distance:
            raise ValueError(f"invalid cost for edge {edge.source!r} -> {edge.target!r}")
        adjacency[edge.source].append((edge, cost))

    goal_distance = ({key: math.dist(xy, nodes[goal]) for key, xy in nodes.items()}
                     if heuristic else dict.fromkeys(nodes, 0.))
    tie_breaker = count()
    queue = [(goal_distance[start], next(tie_breaker), 0., start)]
    best_cost = {start: 0.}
    incoming = {}
    expanded = 0
    while queue:
        _, _, cost, node = heapq.heappop(queue)
        if cost != best_cost[node]:
            continue
        if node == goal:
            path = []
            while node != start:
                edge = incoming[node]
                path.append(edge)
                node = edge.source
            route = tuple(reversed(path))
            return SearchResult((start, *(edge.target for edge in route)), route, cost, expanded)
        expanded += 1
        for edge, edge_cost in adjacency[node]:
            candidate = cost + edge_cost
            if candidate < best_cost.get(edge.target, math.inf):
                best_cost[edge.target] = candidate
                incoming[edge.target] = edge
                heapq.heappush(queue, (candidate + goal_distance[edge.target], next(tie_breaker),
                                       candidate, edge.target))
    return None
