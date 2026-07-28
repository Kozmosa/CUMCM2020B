"""Compact exact enumeration of feasible integer water/food inventories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Q3Config


@dataclass(frozen=True)
class ResourceIndex:
    water: np.ndarray
    food: np.ndarray
    id_grid: np.ndarray

    @classmethod
    def build(cls, cfg: Q3Config) -> "ResourceIndex":
        w_max = cfg.weight_limit // cfg.water_weight
        f_max = cfg.weight_limit // cfg.food_weight
        water: list[int] = []
        food: list[int] = []
        grid = np.full((w_max + 1, f_max + 1), -1, dtype=np.int32)
        for w in range(w_max + 1):
            max_f = (cfg.weight_limit - cfg.water_weight * w) // cfg.food_weight
            for f in range(max_f + 1):
                grid[w, f] = len(water)
                water.append(w)
                food.append(f)
        return cls(
            water=np.asarray(water, dtype=np.int16),
            food=np.asarray(food, dtype=np.int16),
            id_grid=grid,
        )

    def encode(self, water: int, food: int) -> int:
        if water < 0 or food < 0:
            return -1
        if water >= self.id_grid.shape[0] or food >= self.id_grid.shape[1]:
            return -1
        return int(self.id_grid[water, food])

    def decode(self, resource_id: int) -> tuple[int, int]:
        return int(self.water[resource_id]), int(self.food[resource_id])
