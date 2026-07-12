# Building a Neural Network That Learns Minesweeper

This guide walks you through building an AI that teaches *itself* to play
Minesweeper, using the game environment already in this repository. You will
type the code in yourself, file by file. Every step explains **what** you are
building and **why** it works, in plain language.

The technique we use is called **Double Deep Q-Network (DDQN)** combined with a
**Convolutional Neural Network (CNN)**. Don't worry if those words mean nothing
yet — Part 1 explains everything before we write any code.

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

| Layer | Name       | What it shows                                          |
|-------|------------|--------------------------------------------------------|
| 1     | `hidden`   | 1 where a square is still covered, 0 elsewhere         |
| 2     | `revealed` | 1 where a square has been opened                       |
| 3     | `flagged`  | 1 where a flag has been placed                         |
| 4     | `number`   | The number shown on opened squares (scaled 0 to 1)     |
| 5     | `frontier` | 1 on covered squares that touch an opened square       |

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

## Part 3: The CNN Brain — `src/agent/model.py`

**What this file does:** defines the neural network that takes the 5-layer
board picture in and puts one Q-value per square out.

**Key design choice:** we use *only* convolution layers, never a "fully
connected" layer. Fully connected layers are locked to one input size; pure
convolutions work on any board size. The final layer is a 1×1 convolution
that squashes everything down to exactly one number per square.

Type this into `model.py`:

```python
"""The CNN that estimates Q-values: input (5, H, W) -> output (H*W,)."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two conv layers with a shortcut connection.

    The shortcut (adding the input back to the output) makes deep networks
    much easier to train: each block only has to learn a small *correction*
    to its input instead of re-building the whole signal from scratch.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x                      # remember the input
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = x + residual                  # the shortcut: add the input back
        return self.relu(x)


class QNetwork(nn.Module):
    """Fully convolutional Q-network. Works on any board size."""

    def __init__(self, in_channels: int = 5, hidden: int = 64, n_blocks: int = 4):
        super().__init__()
        # First layer: lift the 5 input layers up to 64 feature layers.
        self.input_conv = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # The tower of residual blocks. Each block lets information travel
        # 2 squares further across the board, so 4 blocks + input conv means
        # each output square "sees" roughly a 19x19 neighborhood - plenty.
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
        # Head: a 1x1 conv turns 64 feature layers into exactly 1 number
        # per square - that number is the Q-value for clicking that square.
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x):
        # x arrives as (batch, 5, height, width)
        x = self.relu(self.input_conv(x))
        x = self.blocks(x)
        x = self.head(x)                  # -> (batch, 1, height, width)
        return x.flatten(start_dim=1)     # -> (batch, height*width)
```

**Things worth understanding here:**

- `kernel_size=3, padding=1` means "look through a 3×3 window, and pad the
  board edges so the output stays the same size as the input."
- `ReLU` is the standard "activation function" — it just replaces negative
  numbers with 0. Without activations between layers, stacking layers would
  be pointless (many linear steps collapse into one linear step).
- The output is flattened from a grid to a single row of `height × width`
  numbers, because the environment numbers actions `0, 1, 2, ...` reading the
  board left-to-right, top-to-bottom. Square (row, col) = action
  `row * width + col` — the flatten produces exactly that ordering.

**Quick check before moving on.** Run this from the project root:

```powershell
python -c "import torch; from agent.model import QNetwork; net = QNetwork(); print(net(torch.zeros(2, 5, 9, 9)).shape)"
```

You should see `torch.Size([2, 81])` — 2 boards in, 81 Q-values each out.
Try `(2, 5, 16, 16)` too and you'll get `[2, 256]`: same network, bigger
board, no changes needed. *(If Python can't find `agent`, run
`pip install -e .` once so the `src/` folder is on the path.)*

---

## Part 4: The Memory — `src/agent/replay_buffer.py`

**What this file does:** stores the last ~100,000 experiences and hands back
random batches for learning.

Each experience (usually called a **transition**) has six parts:

1. `obs` — what the board looked like (the 5-layer picture)
2. `action` — which square the AI clicked
3. `reward` — the score the game gave back
4. `next_obs` — what the board looked like afterward
5. `done` — did the game end on this move? (True/False)
6. `next_mask` — which squares are legal to click in the *next* state
   (needed later for the Double DQN calculation)

We store everything in plain NumPy arrays in ordinary RAM (not GPU memory —
your 4GB of GPU memory is precious and the buffer would eat it). It's a
"ring" buffer: when full, the newest experience overwrites the oldest.

Type this into `replay_buffer.py`:

```python
"""A memory bank of past experiences, sampled randomly for learning."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple, n_actions: int):
        self.capacity = capacity
        self.size = 0        # how many experiences are stored so far
        self.pos = 0         # where the next experience will be written

        # Pre-allocate one big array per field. float16/int8 keep RAM low.
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float16)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float16)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.bool_)

    def push(self, obs, action, reward, next_obs, done, next_mask):
        """Save one experience, overwriting the oldest if full."""
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.next_masks[self.pos] = next_mask
        self.pos = (self.pos + 1) % self.capacity   # wrap around at the end
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        """Grab a random batch and move it to the GPU as PyTorch tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[idx], device=device),
            torch.as_tensor(self.next_masks[idx], device=device),
        )

    def __len__(self):
        return self.size
```

**Why random sampling matters (recap):** the 20 moves of one game are highly
related to each other. Learning from them in order is like studying flash
cards that are all near-duplicates — the network overfits to the current game
and forgets older lessons. Shuffling memories from thousands of different
games gives it a balanced diet.

---

## Part 5: The Decision-Maker — `src/agent/agent.py`

**What this file does:** the heart of the project. It owns both networks,
picks moves, and performs the DDQN learning update.

Before the code, understand the **one equation** everything revolves around.
The target we teach the network is:

```
target = reward + gamma × (value of the best next move)     [0 if game ended]
```

In words: "the true worth of this move = the reward it got right now, plus
whatever the future is worth after it." `gamma` (γ, ~0.99) slightly discounts
the future — reward now is worth a bit more than reward later, which also
nudges the AI to win *efficiently*. When the game ended (`done`), there is no
future, so the target is just the reward.

And the **Double DQN split** for "value of the best next move":

- **online network** → *chooses* the best next action (the argmax)
- **target network** → *scores* that chosen action

Type this into `agent.py`:

```python
"""The DDQN agent: picks moves and learns from replayed experience."""

import numpy as np
import torch
import torch.nn as nn

from agent.model import QNetwork


class DDQNAgent:
    def __init__(
        self,
        device: torch.device,
        gamma: float = 0.99,          # how much future reward matters
        lr: float = 1e-4,             # learning rate: size of each nudge
        target_update_every: int = 1000,  # steps between teacher refreshes
    ):
        self.device = device
        self.gamma = gamma
        self.target_update_every = target_update_every
        self.learn_steps = 0

        # The student and the teacher (identical at birth).
        self.online = QNetwork().to(device)
        self.target = QNetwork().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()  # the teacher never trains directly

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        # Huber loss: like squared error for small mistakes, but doesn't
        # explode for big ones - keeps rare -10 mine hits from destabilizing.
        self.loss_fn = nn.SmoothL1Loss()

    @torch.no_grad()
    def act(self, obs: np.ndarray, valid_mask: np.ndarray, epsilon: float) -> int:
        """Pick a square to click. With probability epsilon, explore randomly."""
        if np.random.random() < epsilon:
            # Random exploration - but only among LEGAL squares.
            return int(np.random.choice(np.flatnonzero(valid_mask)))

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q_values = self.online(obs_t.unsqueeze(0)).squeeze(0)  # (H*W,)
        # Action masking: illegal squares get -infinity, so argmax skips them.
        q_values[~torch.as_tensor(valid_mask, device=self.device)] = -torch.inf
        return int(q_values.argmax().item())

    def learn(self, batch) -> float:
        """One learning step on a random batch of memories. Returns the loss."""
        obs, actions, rewards, next_obs, dones, next_masks = batch

        # 1) What does the student CURRENTLY predict for the moves it made?
        q_pred = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2) Build the target (the "correct answer" to nudge toward).
        with torch.no_grad():   # targets are facts, not things to train on
            # --- the Double DQN split ---
            # Student CHOOSES the best next action...
            next_q_online = self.online(next_obs)
            next_q_online[~next_masks] = -torch.inf        # mask illegal moves
            best_next_action = next_q_online.argmax(dim=1)
            # ...teacher SCORES that action.
            next_q_target = self.target(next_obs)
            best_next_q = next_q_target.gather(
                1, best_next_action.unsqueeze(1)
            ).squeeze(1)
            # If the game ended, there is no future: (1 - dones) zeroes it out.
            # Some ended states have no legal moves at all -> guard against -inf.
            best_next_q = torch.nan_to_num(best_next_q, neginf=0.0)
            target = rewards + self.gamma * (1.0 - dones) * best_next_q

        # 3) Nudge the student's prediction toward the target.
        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients: caps the size of any single nudge (stability).
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 4) Every N steps, refresh the teacher with the student's weights.
        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # --- saving / loading trained brains ---

    def save(self, path: str):
        torch.save(self.online.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)
```

**Reading guide for the tricky lines:**

- `.gather(1, actions...)` — the network outputs 81 Q-values per board, but
  we only care about the one for the square that was *actually clicked*.
  `gather` plucks out that one value for each memory in the batch.
- `with torch.no_grad():` — tells PyTorch "don't track this part for
  learning." The target is the answer key; we only learn on the prediction.
- `@torch.no_grad()` on `act` — picking a move during play isn't learning
  either; this makes it faster and lighter on memory.
- `(1.0 - dones)` — `dones` is 1.0 when the game ended, so this multiplies
  the future term by zero exactly when there is no future.

---

## Part 6: The Training Loop — `src/agent/train.py`

**What this file does:** actually runs the show — plays thousands of games,
stores memories, learns after every move, prints progress, and saves the
best brain found so far to a file.

The schedule in plain words:

- Play games back to back. Every single move: act → observe → store.
- Don't start learning until the buffer has ~5,000 memories (learning from
  a nearly-empty, unrepresentative memory bank teaches nonsense).
- After warmup: **one learning step per move**.
- Epsilon slides from 1.0 down to 0.05 over the first ~150,000 moves.
- Every 500 episodes, print stats and run a quick 200-game exam with
  epsilon = 0 (no random moves). If the win rate beat the previous best,
  save the network weights to `checkpoints/best.pt`.

Type this into `train.py`:

```python
"""Train the DDQN agent. Run:  python -m agent.train"""

import argparse
import os
from collections import deque

import numpy as np
import torch

from minesweeper import MinesweeperEnv
from agent.agent import DDQNAgent
from agent.replay_buffer import ReplayBuffer


def evaluate(agent: DDQNAgent, env_kwargs: dict, n_games: int = 200) -> float:
    """Play n_games with no exploration; return the fraction won."""
    env = MinesweeperEnv(**env_kwargs)
    wins = 0
    for _ in range(n_games):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs, env.valid_actions_mask(), epsilon=0.0)
            obs, _, done, info = env.step(action)
        wins += int(info["won"])
    return wins / n_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--eps-decay-steps", type=int, default=150_000)
    parser.add_argument("--eps-final", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    env_kwargs = dict(width=args.width, height=args.height, mine_count=args.mines)
    env = MinesweeperEnv(**env_kwargs)
    agent = DDQNAgent(device=device)
    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_shape=env.observation_shape,
        n_actions=env.action_space_size,
    )

    os.makedirs("checkpoints", exist_ok=True)
    total_steps = 0
    best_win_rate = 0.0
    recent_wins = deque(maxlen=500)     # rolling stats for the printout
    recent_rewards = deque(maxlen=500)

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Epsilon slides from 1.0 to eps-final over eps-decay-steps moves.
            epsilon = max(
                args.eps_final,
                1.0 - (1.0 - args.eps_final) * (total_steps / args.eps_decay_steps),
            )

            mask = env.valid_actions_mask()
            action = agent.act(obs, mask, epsilon)
            next_obs, reward, done, info = env.step(action)

            buffer.push(obs, action, reward, next_obs, done, info["valid_actions"])
            obs = next_obs
            episode_reward += reward
            total_steps += 1

            if len(buffer) >= args.warmup:
                agent.learn(buffer.sample(args.batch_size, device))

        recent_wins.append(int(info["won"]))
        recent_rewards.append(episode_reward)

        if episode % 500 == 0:
            eval_win_rate = evaluate(agent, env_kwargs)
            print(
                f"episode {episode:>7} | steps {total_steps:>8} | "
                f"epsilon {epsilon:.3f} | "
                f"train win% {100 * np.mean(recent_wins):5.1f} | "
                f"avg reward {np.mean(recent_rewards):7.2f} | "
                f"eval win% {100 * eval_win_rate:5.1f}"
            )
            if eval_win_rate > best_win_rate:
                best_win_rate = eval_win_rate
                agent.save("checkpoints/best.pt")
                print(f"  new best! saved to checkpoints/best.pt")

    agent.save("checkpoints/final.pt")
    print(f"Done. Best evaluation win rate: {100 * best_win_rate:.1f}%")


if __name__ == "__main__":
    main()
```

**Note the two win-rate columns in the printout:** "train win%" includes
random exploration moves (it will look worse than the AI really is), while
"eval win%" is the AI playing seriously. Judge progress by eval.

---

## Part 7: Testing the Trained AI — `src/agent/evaluate.py`

**What this file does:** loads a saved brain and measures it properly. Because
the network is size-agnostic, you can also test it on board sizes it never
trained on.

Type this into `evaluate.py`:

```python
"""Evaluate a trained agent. Run:  python -m agent.evaluate --games 1000"""

import argparse

import torch

from minesweeper import MinesweeperEnv
from agent.agent import DDQNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--games", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDQNAgent(device=device)
    agent.load(args.checkpoint)

    env = MinesweeperEnv(width=args.width, height=args.height, mine_count=args.mines)
    wins = 0
    total_revealed = 0
    safe_cells = env.board.safe_cell_count

    for _ in range(args.games):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs, env.valid_actions_mask(), epsilon=0.0)
            obs, _, done, info = env.step(action)
        wins += int(info["won"])
        total_revealed += info["revealed_count"]

    print(f"Board: {args.width}x{args.height}, {args.mines} mines")
    print(f"Games played:      {args.games}")
    print(f"Win rate:          {100 * wins / args.games:.1f}%")
    print(f"Board completion:  {100 * total_revealed / (args.games * safe_cells):.1f}%")


if __name__ == "__main__":
    main()
```

---

## Part 8: Running Everything

### 8.1 Smoke test (do this first!)

Before committing to hours of training, make sure nothing crashes. Run a tiny
session:

```powershell
python -m agent.train --episodes 1000 --warmup 500 --eps-decay-steps 5000
```

You want to see the progress lines print without errors. Win rates will be
near 0% — that's expected after only 1,000 games.

### 8.2 The real training run

```powershell
python -m agent.train
```

What to expect on a 9×9 board with 10 mines:

- **First ~30 minutes:** win rate near 0%. The AI is mostly exploring
  randomly. Average reward should still creep upward — it's learning to
  survive longer even before it can win.
- **A few hours in:** eval win rate climbing through 10–30%.
- **Where it can plateau:** published DQN-style agents reach roughly 40–80%
  on beginner boards depending on tuning and training length. Also remember
  some Minesweeper games *require* a lucky guess — even a perfect player
  can't win 100% on standard boards.

You can stop training at any time with `Ctrl+C` — the best brain so far is
already saved in `checkpoints/best.pt`.

### 8.3 Measure it, and try other board sizes

```powershell
python -m agent.evaluate --games 1000
python -m agent.evaluate --games 500 --width 16 --height 16 --mines 40
```

The second command tests your 9×9-trained brain on an intermediate board it
has never seen — a genuinely fun result of the fully convolutional design.

---

## Part 9: If Learning Stalls — the Tuning Knobs

Reinforcement learning often needs a couple of iterations. If the eval win
rate is flat near zero after a few hours, try these, one at a time:

1. **Slower epsilon decay** — `--eps-decay-steps 300000`. More exploration
   time helps if the AI settled into a bad habit early.
2. **Cleaner training signal with no-guess boards** — in `train.py`, add
   `no_guess=True` to `env_kwargs`. The environment then only generates
   boards solvable by pure logic, so the AI is never punished for an
   unavoidable coin-flip. (Board generation gets slower, but training signal
   gets much cleaner.)
3. **Curriculum: start easier** — train first with `--mines 5`, then continue
   training that same checkpoint with 10 mines. Easier boards give more wins
   early, and wins are where the biggest learning signal comes from.
4. **Lower learning rate** — if the loss value jumps around wildly, change
   `lr=1e-4` to `lr=5e-5` in `agent.py`.
5. **Bigger brain** — `n_blocks=6` and/or `hidden=128` in `model.py`. Costs
   speed; still comfortably fits in 4GB.

---

## Glossary (quick reference)

| Term | Meaning |
|------|---------|
| **Episode** | One complete game, from first click to win/loss. |
| **Step / transition** | One move and its outcome. |
| **Q-value** | Estimated total future reward for clicking a given square. |
| **Policy** | The AI's strategy: "given a board, which square do I click?" |
| **Epsilon (ε)** | Probability of making a random move instead of the best-known one. |
| **Gamma (γ)** | Discount factor: how much tomorrow's reward is worth today (0.99). |
| **Replay buffer** | Memory bank of past moves, sampled randomly for learning. |
| **Online network** | The "student" network that trains constantly. |
| **Target network** | The frozen "teacher" copy, refreshed every ~1,000 steps. |
| **Double DQN** | Online net *picks* the next move, target net *scores* it. |
| **Action masking** | Setting illegal moves' Q-values to −∞ so they're never picked. |
| **Loss** | How wrong the network's predictions are; training pushes it down. |
| **Checkpoint** | A saved copy of the network's weights (`checkpoints/best.pt`). |
| **Huber loss** | An error measure that tolerates occasional big surprises. |
| **Gradient clipping** | A cap on how big any single learning nudge can be. |
