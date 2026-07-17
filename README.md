# Building a Neural Network That Learns Minesweeper

This guide walks you through building an AI that teaches *itself* to play
Minesweeper, using the game environment already in this repository. You will
type the code in yourself, file by file. Every step explains **what** you are
building and **why** it works, in plain language.

The technique we use is called **Double Deep Q-Network (DDQN)** combined with a
**Convolutional Neural Network (CNN)**. Don't worry if those words mean nothing yet — Part 1 explains everything before we write any code. **Head to instruction_manual after reading this file.**

---



## Part 1: The Big Ideas (read this first)



### 1.1 How does an AI "learn" a game?

There is no list of Minesweeper rules programmed into the AI. Instead, it
learns the same way you might learn a game with no instructions:

1. It clicks a square (mostly at random at first).
2. The game gives it a **reward** — a number that says "that was good" or
  "that was bad". In our environment:
  - Revealing a safe square: small positive reward (+0.10 per square)
  - Winning the game: big positive reward (+10)
  - Clicking a mine: big negative reward (−10)
  - Clicking a square that's already revealed: penalty (−1)
3. It slowly adjusts itself so that moves which led to good rewards become
  more likely, and moves that led to bad rewards become less likely.

Repeat this over tens of thousands of games and the AI gets genuinely good.
This whole field is called **Reinforcement Learning** (RL) — learning by
trial, error, and reward.

### 1.2 What is a Q-value?

Imagine, for every square on the board, a number that answers:

> "If I click this square now, how much total reward can I expect to
> collect from here to the end of the game?"

That number is called a **Q-value** ("Q" for *quality* of an action). If the
AI knew the true Q-value of every square, playing perfectly would be easy:
just always click the square with the highest Q-value.

Of course, it doesn't know the true Q-values. **The entire point of training
is to learn a good estimate of them.** The thing doing the estimating is a
neural network — which is why this method is called a **Deep Q-Network (DQN)**.

### 1.3 What does the neural network actually see?

The environment (already written for you in `src/minesweeper/env.py`) turns
the board into a stack of 5 grids of numbers, each the same size as the board.
Think of them as 5 transparent overlays:


| Layer | Name       | What it shows                                      |
| ----- | ---------- | -------------------------------------------------- |
| 1     | `hidden`   | 1 where a square is still covered, 0 elsewhere     |
| 2     | `revealed` | 1 where a square has been opened                   |
| 3     | `flagged`  | 1 where a flag has been placed                     |
| 4     | `number`   | The number shown on opened squares (scaled 0 to 1) |
| 5     | `frontier` | 1 on covered squares that touch an opened square   |


So for a 9×9 board, the input to the network is a 5×9×9 block of numbers.
This is exactly like a small image with 5 color channels instead of 3 (RGB) —
which is why an image-processing network (a CNN) is the perfect fit.

### 1.4 What is a CNN and why use one here?

A **Convolutional Neural Network** looks at an image through a small sliding
window (usually 3×3) and learns patterns inside that window. It slides the
same window over every position on the board, so **a pattern learned in one
corner automatically works everywhere else**.

This matters a lot for Minesweeper, because Minesweeper logic is *local*:
whether a square is safe depends on the numbers right next to it, and the
same logic applies anywhere on the board. Examples of patterns the network
can learn:

- "A revealed 1 that touches exactly one covered square → that square is a mine."
- "A revealed 1 whose mine is already accounted for → its other neighbors are safe."

A bonus: because a CNN only ever looks through its small sliding window, the
**same trained network works on any board size** — train on 9×9, then run it
on 16×16 without changing anything. We design our network carefully (no
"fully connected" layers) to keep this property.

### 1.5 The four supporting tricks that make DQN work

Plain "neural network + rewards" is unstable on its own. Four standard tricks
fix that, and all four appear in our code:

**Trick 1 — Replay Buffer (the memory).**
Instead of learning from each move immediately and then throwing it away, the
AI saves every experience — *(what the board looked like, what I clicked, what
reward I got, what the board looked like after)* — into a big memory bank
holding ~100,000 experiences. To learn, it grabs a **random batch** of old
memories and studies those. Random sampling breaks up the "streakiness" of
consecutive moves, which otherwise confuses learning. It also means each
experience gets reused many times, so the AI learns more from every game.

**Trick 2 — Target Network (the stable teacher).**
Training a network toward targets computed by *itself* is like chasing your
own shadow — the target moves whenever you move. So we keep **two copies** of
the network:

- The **online network**: the student. Updated constantly.
- The **target network**: the teacher. A frozen snapshot of the student,
refreshed only once every ~1,000 learning steps.

The student learns toward answers from the frozen teacher, which holds still
long enough to actually be learned from.

**Trick 3 — Double DQN (the honest scorekeeper).**
This is the "Double" in DDQN. Regular DQN asks the teacher network both "which
next move is best?" *and* "how good is it?" — and using one network for both
questions systematically **overestimates** how good moves are (any random
error in a Q-value makes that move look better than it is, and the max picks
it up). The Double DQN fix splits the two questions:

- The **online** network picks *which* next move looks best.
- The **target** network scores *how good* that move is.

One line of code different from regular DQN, and it noticeably improves
learning stability.

**Trick 4 — Epsilon-Greedy Exploration (curiosity).**
If the AI always played its current best guess, it would never discover
better moves it hasn't tried. So with probability **epsilon (ε)** it makes a
random move instead. We start at ε = 1.0 (100% random — pure exploration) and
slowly lower it to ε = 0.05 (5% random) over training. Early on it explores;
later it mostly exploits what it knows.

### 1.6 One more important idea: action masking

The board has 81 squares (on 9×9), and the network outputs 81 Q-values — one
per square. But many squares are already revealed and clicking them is
useless. The environment gives us a **mask**: a list of true/false saying
which squares are still legal to click.

Everywhere in our code, before picking a move, we set the Q-values of illegal
squares to negative infinity so they can never be chosen. This saves the AI
from wasting thousands of games learning the obvious lesson "don't click
squares that are already open."

### 1.7 The full learning loop, in one picture

```
                 ┌──────────────────────────────────────────┐
                 │            One training step             │
                 └──────────────────────────────────────────┘

   board state ──► ONLINE NETWORK ──► Q-values ──► mask illegal moves
                                                        │
                              ε chance of random ◄──────┘
                                                        │
                                                   pick a move
                                                        │
                                                 GAME ENVIRONMENT
                                                        │
                                     (reward, new board state, done?)
                                                        │
                                                 save to REPLAY BUFFER
                                                        │
                             sample a random batch of 128 old memories
                                                        │
                     compute targets using TARGET NETWORK (Double DQN)
                                                        │
                          nudge ONLINE NETWORK toward those targets
                                                        │
                every ~1000 steps: copy online ──► target (refresh teacher)
```

---



## Part 2: Setup



### Step 2.1 — Install PyTorch

PyTorch is the library we use to build and train neural networks. Since you
have a 4GB NVIDIA GPU, install the CUDA (GPU-enabled) version. In your
terminal, from the project folder:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then verify the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

It should print `True`. If it prints `False`, the CPU-only version got
installed or your GPU driver needs updating — training will still work on
CPU, just slower.

Also add `torch` on its own line inside `requirements.txt` so the dependency
is recorded.

### Step 2.2 — Create the agent folder

We keep all the AI code separate from the game code. Create this structure
inside `src/`:

```
src/
└── agent/
    ├── __init__.py        (empty file — marks this folder as a Python package)
    ├── model.py           (the CNN brain)
    ├── replay_buffer.py   (the memory)
    ├── agent.py           (the decision-maker: DDQN logic)
    ├── train.py           (the training loop — runs thousands of games)
    └── evaluate.py        (tests how good the trained AI is)
```

Create the folder and an empty `__init__.py` first. The next sections fill in
each file, one at a time, in an order where each file only depends on files
you've already written.

---



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

Or start from a difficulty preset:

```powershell
python -m minesweeper.ui --difficulty Expert
```

Controls:

- Left click reveals a cell.
- Right click toggles a flag.
- Press `R` to reset.
- Press `1` / `2` / `3` to switch board size (Beginner / Intermediate / Expert).
- Press arrow keys to resize board width/height in custom mode.
- Press `[` / `]` to decrease/increase mine count in custom mode.



## No-Guess Boards

By default every board is guaranteed to be solvable with pure logic, so you
never have to guess. On the first click the game repeatedly generates layouts
and keeps only the ones a sound logical solver (`minesweeper.solver`) can clear
completely. The first click also always opens an empty region.

Pass `--allow-guessing` to the UI to disable this and play classic random
boards instead. The environment exposes the same option:

```python
from minesweeper import MinesweeperEnv

env = MinesweeperEnv(width=16, height=16, mine_count=40, no_guess=True)
```



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

