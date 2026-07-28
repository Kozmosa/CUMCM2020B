"""Only conservative, deterministic Q3 pruning helpers."""

from __future__ import annotations

from collections import deque

from .data import Q3Config


def shortest_distance_to_end(cfg: Q3Config) -> dict[int, int]:
    """Optimistic BFS distance, ignoring weather and all player interaction."""
    distance = {node: 10**9 for node in cfg.nodes}
    distance[cfg.end] = 0
    queue: deque[int] = deque([cfg.end])
    while queue:
        node = queue.popleft()
        for neighbor in cfg.adj[node]:
            if distance[neighbor] > distance[node] + 1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def success_is_deadline_impossible(
    position: int, *, day: int, cfg: Q3Config, distance: dict[int, int]
) -> bool:
    """Prove only that success is impossible; do not discard cash-seeking actions."""
    return distance[position] > cfg.deadline - day
