from minesweeper.board import MinesweeperBoard
from minesweeper.solver import is_no_guess_solvable


def test_generated_no_guess_board_is_solvable_by_solver() -> None:
    for seed in range(10):
        board = MinesweeperBoard(9, 9, 10, seed=seed, no_guess=True)
        first_cell = (4, 4)
        board.place_mines(first_cell)

        assert is_no_guess_solvable(
            board.width, board.height, board.mine_count, board.counts, first_cell
        )


def test_no_guess_first_click_opens_a_region() -> None:
    board = MinesweeperBoard(9, 9, 10, seed=3, no_guess=True)
    first_cell = (4, 4)
    board.place_mines(first_cell)

    assert board.counts[first_cell[0]][first_cell[1]] == 0
    assert not board.is_mine(*first_cell)


def test_solver_rejects_a_board_that_needs_guessing() -> None:
    # 1x5 strip with mines at columns 1 and 3. After the safe first click on
    # column 0 the player learns one mine, but the remaining mine could sit in
    # any of the three far cells: a genuine forced guess.
    width, height, mine_count = 5, 1, 2
    mines = {(0, 1), (0, 3)}
    counts = [[0] * width]
    for _, col in mines:
        for d_col in (-1, 1):
            n_col = col + d_col
            if 0 <= n_col < width and (0, n_col) not in mines:
                counts[0][n_col] += 1

    assert not is_no_guess_solvable(width, height, mine_count, counts, (0, 0))
