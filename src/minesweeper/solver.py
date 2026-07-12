"""A sound logical solver used to certify that a board needs no guessing.

The solver never looks at hidden mine locations directly. It only uses the
numbers of cells it has logically revealed, exactly like a human player. If it
can clear every safe cell using pure deduction, the board is guaranteed to be
solvable without guessing.

Because every deduction rule below is *sound* (it never marks a cell safe unless
it provably is), any board the solver fully clears is a genuine no-guess board.
"""

from __future__ import annotations

from typing import Iterable

Cell = tuple[int, int]

# Bounds that keep the (rare) exhaustive enumeration step cheap. Frontier
# components larger than this are skipped; the generator simply tries another
# layout instead of spending time on an expensive search.
_MAX_ENUM_CELLS = 22
_SOLUTION_CAP = 20000


class _Simulation:
    def __init__(
        self,
        width: int,
        height: int,
        mine_count: int,
        counts: list[list[int]],
    ) -> None:
        self.width = width
        self.height = height
        self.mine_count = mine_count
        self.counts = counts
        self.revealed: set[Cell] = set()
        self.known_mines: set[Cell] = set()
        self.known_safe: set[Cell] = set()

    def neighbors(self, row: int, col: int) -> Iterable[Cell]:
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < self.height and 0 <= n_col < self.width:
                    yield n_row, n_col

    def all_cells(self) -> Iterable[Cell]:
        for row in range(self.height):
            for col in range(self.width):
                yield row, col

    def reveal(self, cell: Cell) -> None:
        """Reveal a cell that is known to be safe, flooding through zeros."""
        stack = [cell]
        while stack:
            current = stack.pop()
            if current in self.revealed or current in self.known_mines:
                continue
            self.revealed.add(current)
            self.known_safe.discard(current)
            row, col = current
            if self.counts[row][col] == 0:
                for neighbor in self.neighbors(row, col):
                    if neighbor not in self.revealed:
                        stack.append(neighbor)

    def _flush_safe(self) -> None:
        for cell in list(self.known_safe):
            self.reveal(cell)

    def _constraints(self) -> list[tuple[frozenset[Cell], int]]:
        constraints: list[tuple[frozenset[Cell], int]] = []
        for row, col in self.revealed:
            count = self.counts[row][col]
            if count == 0:
                continue
            unknown: list[Cell] = []
            known = 0
            for neighbor in self.neighbors(row, col):
                if neighbor in self.known_mines:
                    known += 1
                elif neighbor not in self.revealed:
                    unknown.append(neighbor)
            if unknown:
                constraints.append((frozenset(unknown), count - known))
        return constraints

    def _deduce_simple(self) -> bool:
        progress = False
        for cells, remaining in self._constraints():
            if remaining == 0:
                for cell in cells:
                    if cell not in self.known_safe:
                        self.known_safe.add(cell)
                        progress = True
            elif remaining == len(cells):
                for cell in cells:
                    if cell not in self.known_mines:
                        self.known_mines.add(cell)
                        progress = True
        return progress

    def _deduce_global(self) -> bool:
        hidden_unknown = [
            cell
            for cell in self.all_cells()
            if cell not in self.revealed and cell not in self.known_mines
        ]
        remaining_mines = self.mine_count - len(self.known_mines)
        progress = False
        if remaining_mines == 0:
            for cell in hidden_unknown:
                if cell not in self.known_safe:
                    self.known_safe.add(cell)
                    progress = True
        elif len(hidden_unknown) == remaining_mines:
            for cell in hidden_unknown:
                if cell not in self.known_mines:
                    self.known_mines.add(cell)
                    progress = True
        return progress

    def _deduce_subset(self) -> bool:
        constraints = self._constraints()
        progress = False
        for a_cells, a_remaining in constraints:
            for b_cells, b_remaining in constraints:
                if a_cells is b_cells or not a_cells < b_cells:
                    continue
                diff = b_cells - a_cells
                diff_remaining = b_remaining - a_remaining
                if diff_remaining == 0:
                    for cell in diff:
                        if cell not in self.known_safe:
                            self.known_safe.add(cell)
                            progress = True
                elif diff_remaining == len(diff):
                    for cell in diff:
                        if cell not in self.known_mines:
                            self.known_mines.add(cell)
                            progress = True
        return progress

    def _deduce_enumerate(self) -> bool:
        constraints = self._constraints()
        if not constraints:
            return False

        cell_to_constraints: dict[Cell, list[int]] = {}
        for index, (cells, _remaining) in enumerate(constraints):
            for cell in cells:
                cell_to_constraints.setdefault(cell, []).append(index)

        visited: set[Cell] = set()
        progress = False
        for start in cell_to_constraints:
            if start in visited:
                continue

            component_cells: set[Cell] = set()
            component_constraints: set[int] = set()
            queue = [start]
            while queue:
                cell = queue.pop()
                if cell in component_cells:
                    continue
                component_cells.add(cell)
                for constraint_index in cell_to_constraints[cell]:
                    component_constraints.add(constraint_index)
                    for other in constraints[constraint_index][0]:
                        if other not in component_cells:
                            queue.append(other)

            visited |= component_cells
            if len(component_cells) > _MAX_ENUM_CELLS:
                continue

            cells_list = sorted(component_cells)
            local_constraints = [constraints[i] for i in component_constraints]
            solutions = self._enumerate_component(cells_list, local_constraints)
            if not solutions:
                continue

            total = len(solutions)
            for position, cell in enumerate(cells_list):
                mine_solutions = sum(1 for solution in solutions if solution[position])
                if mine_solutions == 0:
                    if cell not in self.known_safe:
                        self.known_safe.add(cell)
                        progress = True
                elif mine_solutions == total:
                    if cell not in self.known_mines:
                        self.known_mines.add(cell)
                        progress = True
        return progress

    def _enumerate_component(
        self,
        cells: list[Cell],
        constraints: list[tuple[frozenset[Cell], int]],
    ) -> list[tuple[int, ...]]:
        index = {cell: position for position, cell in enumerate(cells)}
        indexed = [([index[cell] for cell in group], remaining) for group, remaining in constraints]
        assignment: list[int | None] = [None] * len(cells)
        solutions: list[tuple[int, ...]] = []

        def consistent() -> bool:
            for positions, remaining in indexed:
                assigned_sum = 0
                unassigned = 0
                for position in positions:
                    value = assignment[position]
                    if value is None:
                        unassigned += 1
                    else:
                        assigned_sum += value
                if assigned_sum > remaining or assigned_sum + unassigned < remaining:
                    return False
            return True

        def backtrack(position: int) -> None:
            if len(solutions) > _SOLUTION_CAP:
                return
            if position == len(cells):
                solutions.append(tuple(value or 0 for value in assignment))
                return
            for value in (0, 1):
                assignment[position] = value
                if consistent():
                    backtrack(position + 1)
            assignment[position] = None

        backtrack(0)
        return solutions

    def solve(self, safe_cell: Cell) -> bool:
        self.reveal(safe_cell)
        safe_total = self.width * self.height - self.mine_count

        while True:
            if len(self.revealed) == safe_total:
                return True
            if self._deduce_simple() or self._deduce_global():
                self._flush_safe()
                continue
            if self._deduce_subset():
                self._flush_safe()
                continue
            if self._deduce_enumerate():
                self._flush_safe()
                continue
            return False


def is_no_guess_solvable(
    width: int,
    height: int,
    mine_count: int,
    counts: list[list[int]],
    safe_cell: Cell,
) -> bool:
    """Return True if the board can be fully cleared using only logic."""
    return _Simulation(width, height, mine_count, counts).solve(safe_cell)
