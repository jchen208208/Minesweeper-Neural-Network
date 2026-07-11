# Minesweeper Neural Network

A DDQN-friendly Minesweeper environment with a playable Pygame UI.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Play

```powershell
python -m minesweeper.ui
```

Optional board settings:

```powershell
python -m minesweeper.ui --width 9 --height 9 --mines 10 --seed 42
```

Controls:

- Left click reveals a cell.
- Right click toggles a flag.
- Press `R` to reset.

## Use From DDQN Code

The environment has one discrete action per cell. For a board with width `w`,
decode an action with `row = action // w` and `col = action % w`.

```python
from minesweeper import MinesweeperEnv

env = MinesweeperEnv(width=5, height=5, mine_count=4, seed=123)
obs = env.reset()

valid_actions = env.valid_actions_mask()
action = valid_actions.nonzero()[0][0]

next_obs, reward, done, info = env.step(action)
```

Observations are NumPy arrays shaped `(5, height, width)` with these channels:

1. Hidden cells
2. Revealed cells
3. Flagged cells
4. Revealed neighbor count normalized to `0.0-1.0`
5. Hidden frontier cells touching revealed information

The first reveal is guaranteed safe. This gives a DDQN a cleaner learning
problem while preserving the key Minesweeper uncertainty: hidden mine locations
are never exposed in the observation.

## Test

```powershell
pytest
```
