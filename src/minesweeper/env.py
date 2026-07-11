from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from minesweeper.board import GameStatus, MinesweeperBoard


@dataclass(frozen=True)
class RewardConfig:
    reveal_cell: float = 0.10
    reveal_zero_bonus: float = 0.05
    win: float = 10.0
    hit_mine: float = -10.0
    invalid_action: float = -1.0
    step_cost: float = -0.01


class MinesweeperEnv:
    """DDQN-friendly Minesweeper environment with a Gym-like step API."""

    channel_names = ("hidden", "revealed", "flagged", "number", "frontier")

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        mine_count: int = 4,
        seed: int | None = None,
        rewards: RewardConfig | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.mine_count = mine_count
        self.rewards = rewards or RewardConfig()
        self.board = MinesweeperBoard(width, height, mine_count, seed)
        self.last_action: int | None = None

    @property
    def action_space_size(self) -> int:
        return self.width * self.height

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return len(self.channel_names), self.height, self.width

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.board.reset(seed)
        self.last_action = None
        return self.observation()

    def decode_action(self, action: int) -> tuple[int, int]:
        return divmod(int(action), self.width)

    def encode_action(self, row: int, col: int) -> int:
        if not self.board.in_bounds(row, col):
            raise ValueError("cell is out of bounds")
        return row * self.width + col

    def valid_actions_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.bool_)
        if self.board.status != GameStatus.ONGOING:
            return mask

        for row, col in self.board.cells():
            action = self.encode_action(row, col)
            mask[action] = not self.board.is_revealed(row, col) and not self.board.is_flagged(
                row, col
            )
        return mask

    def observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_shape, dtype=np.float32)

        for row, col in self.board.cells():
            hidden = not self.board.is_revealed(row, col)
            flagged = self.board.is_flagged(row, col)
            revealed = self.board.is_revealed(row, col)

            obs[0, row, col] = 1.0 if hidden and not flagged else 0.0
            obs[1, row, col] = 1.0 if revealed else 0.0
            obs[2, row, col] = 1.0 if flagged else 0.0

            if revealed and not self.board.is_mine(row, col):
                obs[3, row, col] = self.board.counts[row][col] / 8.0

            if hidden and self._touches_revealed_cell(row, col):
                obs[4, row, col] = 1.0

        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if action < 0 or action >= self.action_space_size:
            return self._invalid_step(action, "out_of_bounds")

        row, col = self.decode_action(action)
        self.last_action = action

        if not self.valid_actions_mask()[action]:
            return self._invalid_step(action, "not_revealable")

        result = self.board.reveal(row, col)
        reward = self.rewards.step_cost
        if result.invalid:
            reward += self.rewards.invalid_action
        elif result.hit_mine:
            reward += self.rewards.hit_mine
        else:
            reward += result.revealed_count * self.rewards.reveal_cell
            if result.revealed_count > 1:
                reward += self.rewards.reveal_zero_bonus
            if result.won:
                reward += self.rewards.win

        done = self.board.status != GameStatus.ONGOING
        info = {
            "status": self.board.status.value,
            "revealed_count": result.revealed_count,
            "hit_mine": result.hit_mine,
            "won": result.won,
            "invalid": result.invalid,
            "valid_actions": self.valid_actions_mask(),
        }
        return self.observation(), reward, done, info

    def _invalid_step(
        self, action: int, reason: str
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        done = self.board.status != GameStatus.ONGOING
        info = {
            "status": self.board.status.value,
            "revealed_count": 0,
            "hit_mine": False,
            "won": self.board.status == GameStatus.WON,
            "invalid": True,
            "invalid_reason": reason,
            "valid_actions": self.valid_actions_mask(),
        }
        self.last_action = int(action)
        return self.observation(), self.rewards.invalid_action, done, info

    def _touches_revealed_cell(self, row: int, col: int) -> bool:
        return any(self.board.is_revealed(n_row, n_col) for n_row, n_col in self.board.neighbors(row, col))
