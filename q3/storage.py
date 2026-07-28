"""Compact state codecs and date-layer containers used by Q3 checkpoints."""

from __future__ import annotations

import struct
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


@dataclass(frozen=True)
class PackedStateCodec:
    """Fixed-width byte keys for hot value and policy dictionaries.

    A three-player level-6 key occupies 68 bytes including the day, versus a
    nested tuple of PlayerState objects and their individual Python integers.
    The representation is lossless for the supported Q3 map and resource
    index and remains directly decodable for checkpoint serialization.
    """

    resources: ResourceIndex
    n_players: int

    _DAY = struct.Struct("<H")
    _PLAYER = struct.Struct("<BBIqq")

    def __post_init__(self) -> None:
        if not 1 <= self.n_players <= 3:
            raise ValueError("packed Q3 state keys support one to three players")
        if len(self.resources.water) >= 1 << 32:
            raise ValueError("resource ids do not fit the packed Q3 key")

    @property
    def key_size(self) -> int:
        return self._DAY.size + self.n_players * self._PLAYER.size

    def encode(self, day: int, state: JointState) -> bytes:
        if not 0 <= day < 1 << 16:
            raise ValueError("day does not fit the packed Q3 key")
        if len(state) != self.n_players:
            raise ValueError("joint state has the wrong player count")
        output = bytearray(self.key_size)
        self._DAY.pack_into(output, 0, day)
        offset = self._DAY.size
        for player in state:
            if not 0 <= player.position < 1 << 8:
                raise ValueError("position does not fit the packed Q3 key")
            resource_id = self.resources.encode(player.water, player.food)
            if resource_id < 0:
                raise ValueError("state inventory is outside the resource index")
            self._PLAYER.pack_into(
                output,
                offset,
                int(player.status),
                player.position,
                resource_id,
                player.cash_scaled,
                player.fixed_payoff_scaled,
            )
            offset += self._PLAYER.size
        return bytes(output)

    def decode(self, key: bytes) -> tuple[int, JointState]:
        if len(key) != self.key_size:
            raise ValueError("packed state key has the wrong width")
        day = int(self._DAY.unpack_from(key, 0)[0])
        players: list[PlayerState] = []
        offset = self._DAY.size
        for _ in range(self.n_players):
            status, position, resource_id, cash, payoff = self._PLAYER.unpack_from(
                key, offset
            )
            water, food = self.resources.decode(int(resource_id))
            players.append(
                PlayerState(
                    Status(int(status)),
                    position=int(position),
                    water=water,
                    food=food,
                    cash_scaled=int(cash),
                    fixed_payoff_scaled=int(payoff),
                )
            )
            offset += self._PLAYER.size
        return day, tuple(players)


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
