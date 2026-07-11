from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Iterable


Cell = tuple[int, int]


class GameStatus(str, Enum):
    ONGOING = "ongoing"
    WON = "won"
    LOST = "lost"


@dataclass(frozen=True)
class RevealResult:
    revealed_count: int
    hit_mine: bool
    won: bool
    invalid: bool = False


class MinesweeperBoard:
    """Pure Minesweeper rules with delayed mine placement for first-click safety."""

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        mine_count: int = 4,
        seed: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if mine_count < 0:
            raise ValueError("mine_count must be non-negative")
        if mine_count >= width * height:
            raise ValueError("mine_count must leave at least one safe cell")

        self.width = width
        self.height = height
        self.mine_count = mine_count
        self._rng = random.Random(seed)
        self._initial_seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
        elif self._initial_seed is not None:
            self._rng.seed(self._initial_seed)

        self.mines: set[Cell] = set()
        self.flags: set[Cell] = set()
        self.revealed: set[Cell] = set()
        self.counts: list[list[int]] = [
            [0 for _ in range(self.width)] for _ in range(self.height)
        ]
        self.status = GameStatus.ONGOING
        self.mines_placed = False

    @property
    def safe_cell_count(self) -> int:
        return self.width * self.height - self.mine_count

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def cells(self) -> Iterable[Cell]:
        for row in range(self.height):
            for col in range(self.width):
                yield row, col

    def neighbors(self, row: int, col: int) -> Iterable[Cell]:
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                next_row = row + d_row
                next_col = col + d_col
                if self.in_bounds(next_row, next_col):
                    yield next_row, next_col

    def place_mines(self, safe_cell: Cell) -> None:
        if self.mines_placed:
            return
        if not self.in_bounds(*safe_cell):
            raise ValueError("safe_cell must be in bounds")

        candidates = [cell for cell in self.cells() if cell != safe_cell]
        if self.mine_count > len(candidates):
            raise ValueError("mine_count is too high for first-click safety")

        self.mines = set(self._rng.sample(candidates, self.mine_count))
        self._compute_counts()
        self.mines_placed = True

    def _compute_counts(self) -> None:
        self.counts = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for row, col in self.mines:
            for neighbor_row, neighbor_col in self.neighbors(row, col):
                self.counts[neighbor_row][neighbor_col] += 1

    def reveal(self, row: int, col: int) -> RevealResult:
        cell = (row, col)
        if (
            self.status != GameStatus.ONGOING
            or not self.in_bounds(row, col)
            or cell in self.flags
            or cell in self.revealed
        ):
            return RevealResult(0, False, self.status == GameStatus.WON, invalid=True)

        if not self.mines_placed:
            self.place_mines(cell)

        if cell in self.mines:
            self.revealed.add(cell)
            self.status = GameStatus.LOST
            return RevealResult(1, True, False)

        revealed_count = self._flood_reveal(row, col)
        if len(self.revealed - self.mines) == self.safe_cell_count:
            self.status = GameStatus.WON
            return RevealResult(revealed_count, False, True)

        return RevealResult(revealed_count, False, False)

    def _flood_reveal(self, row: int, col: int) -> int:
        revealed_before = len(self.revealed)
        stack = [(row, col)]

        while stack:
            current = stack.pop()
            if current in self.revealed or current in self.flags or current in self.mines:
                continue

            self.revealed.add(current)
            current_row, current_col = current
            if self.counts[current_row][current_col] != 0:
                continue

            for neighbor in self.neighbors(current_row, current_col):
                if neighbor not in self.revealed and neighbor not in self.flags:
                    stack.append(neighbor)

        return len(self.revealed) - revealed_before

    def toggle_flag(self, row: int, col: int) -> bool:
        cell = (row, col)
        if self.status != GameStatus.ONGOING or not self.in_bounds(row, col):
            return False
        if cell in self.revealed:
            return False

        if cell in self.flags:
            self.flags.remove(cell)
        else:
            self.flags.add(cell)
        return True

    def is_revealed(self, row: int, col: int) -> bool:
        return (row, col) in self.revealed

    def is_flagged(self, row: int, col: int) -> bool:
        return (row, col) in self.flags

    def is_mine(self, row: int, col: int) -> bool:
        return (row, col) in self.mines
