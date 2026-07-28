"""Compact state codecs and date-layer containers used by Q3 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import JointState, PlayerState, StateValue, Status
from .resource_index import ResourceIndex


@dataclass(frozen=True)
class CompactStateCodec:
    resources: ResourceIndex
    n_players: int

    def encode(self, state: JointState) -> tuple[int, ...]:
        if len(state) != self.n_players:
            raise ValueError("joint state has the wrong player count")
        out: list[int] = []
        for player in state:
            resource_id = self.resources.encode(player.water, player.food)
            out.extend(
                (
                    int(player.status),
                    player.position,
                    resource_id,
                    player.cash_scaled,
                    player.fixed_payoff_scaled,
                )
            )
        return tuple(out)

    def decode(self, key: tuple[int, ...]) -> JointState:
        width = 5
        if len(key) != width * self.n_players:
            raise ValueError("compact state key has the wrong width")
        players: list[PlayerState] = []
        for offset in range(0, len(key), width):
            status, position, resource_id, cash, payoff = key[offset : offset + width]
            water, food = self.resources.decode(resource_id)
            players.append(
                PlayerState(
                    Status(status),
                    position=position,
                    water=water,
                    food=food,
                    cash_scaled=cash,
                    fixed_payoff_scaled=payoff,
                )
            )
        return tuple(players)


@dataclass
class LayerCache:
    """Append-only compact rows with a hash index only for row lookup."""

    codec: CompactStateCodec
    keys: list[tuple[int, ...]] = field(default_factory=list)
    values: list[tuple[float, ...]] = field(default_factory=list)
    success: list[tuple[float, ...]] = field(default_factory=list)
    index: dict[tuple[int, ...], int] = field(default_factory=dict)

    def put(self, state: JointState, value: StateValue) -> int:
        key = self.codec.encode(state)
        row = self.index.get(key)
        if row is None:
            row = len(self.keys)
            self.index[key] = row
            self.keys.append(key)
            self.values.append(value.value)
            self.success.append(value.success)
        else:
            self.values[row] = value.value
            self.success[row] = value.success
        return row

    def get(self, state: JointState) -> StateValue | None:
        row = self.index.get(self.codec.encode(state))
        if row is None:
            return None
        return StateValue(self.values[row], self.success[row])

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width = 5 * self.codec.n_players
        keys = np.asarray(self.keys, dtype=np.int64).reshape((-1, width))
        values = np.asarray(self.values, dtype=np.float64).reshape(
            (-1, self.codec.n_players)
        )
        success = np.asarray(self.success, dtype=np.float64).reshape(
            (-1, self.codec.n_players)
        )
        return keys, values, success
