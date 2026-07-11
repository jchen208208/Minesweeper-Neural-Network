import numpy as np
import pytest

from minesweeper.board import GameStatus, MinesweeperBoard
from minesweeper.env import MinesweeperEnv


def test_first_click_is_always_safe() -> None:
    env = MinesweeperEnv(width=5, height=5, mine_count=10, seed=7)
    action = env.encode_action(2, 2)

    _, reward, done, info = env.step(action)

    assert not done
    assert not info["hit_mine"]
    assert reward > 0
    assert not env.board.is_mine(2, 2)
    assert env.board.is_revealed(2, 2)


def test_observation_shape_and_valid_action_mask_update_after_reveal() -> None:
    env = MinesweeperEnv(width=4, height=3, mine_count=2, seed=1)
    obs = env.reset()

    assert obs.shape == (5, 3, 4)
    assert obs.dtype == np.float32
    assert env.valid_actions_mask().shape == (12,)
    assert env.valid_actions_mask().all()

    action = env.encode_action(0, 0)
    env.step(action)

    assert not env.valid_actions_mask()[action]


def test_invalid_action_gets_penalty_and_does_not_end_game() -> None:
    env = MinesweeperEnv(width=3, height=3, mine_count=1, seed=5)

    _, reward, done, info = env.step(100)

    assert reward < 0
    assert not done
    assert info["invalid"]
    assert info["invalid_reason"] == "out_of_bounds"


def test_hitting_mine_ends_game_with_penalty() -> None:
    env = MinesweeperEnv(width=3, height=3, mine_count=1, seed=2)
    env.step(env.encode_action(0, 0))
    mine = next(iter(env.board.mines))

    _, reward, done, info = env.step(env.encode_action(*mine))

    assert done
    assert reward < -1
    assert info["hit_mine"]
    assert env.board.status == GameStatus.LOST


def test_revealing_all_safe_cells_wins() -> None:
    env = MinesweeperEnv(width=2, height=2, mine_count=1, seed=4)
    env.step(env.encode_action(0, 0))

    done = False
    info = {}
    reward = 0.0
    for row, col in list(env.board.cells()):
        if not env.board.is_mine(row, col) and not env.board.is_revealed(row, col):
            _, reward, done, info = env.step(env.encode_action(row, col))

    assert done
    assert reward > 1
    assert info["won"]
    assert env.board.status == GameStatus.WON


def test_zero_cell_flood_reveal_expands_region() -> None:
    board = MinesweeperBoard(width=4, height=4, mine_count=1, seed=0)
    board.mines = {(3, 3)}
    board.mines_placed = True
    board._compute_counts()

    result = board.reveal(0, 0)

    assert not result.hit_mine
    assert result.revealed_count > 1
    assert board.is_revealed(0, 0)


def test_invalid_board_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        MinesweeperEnv(width=2, height=2, mine_count=4)
