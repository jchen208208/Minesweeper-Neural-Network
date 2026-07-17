# Instruction Manual: Write Your Own Minesweeper AI

This manual guides you through writing every file of the DDQN + CNN agent
**yourself**. It tells you what each piece must do, warns you about the
classic mistakes, and gives pseudocode where the logic gets dense — but the
actual Python is yours to write.

If you get truly stuck, `finished_code_answers.md` contains a full working version of every
file. Treat it like the answer key at the back of a textbook: check it
*after* you've made an honest attempt, not before.

---

## How to work through this manual

Each part below follows the same shape:

- **Goal** — what the file must accomplish, in one sentence.
- **Spec** — the exact ingredients: class names, methods, inputs, outputs.
Follow these closely, because later files call the earlier ones and the
names need to match.
- **Pseudocode** — for the genuinely tricky logic only.
- **Watch out** — the mistakes almost everyone makes on their first try.
- **Check yourself** — a small test to run before moving to the next part.

Build the files in the order given. Each one only depends on files you've
already finished.

---



## Part 0: Setup

1. Install PyTorch with GPU support:
  ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
2. Verify the GPU is detected — this must print `True`:
  ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
  ```
3. Add `torch` to `requirements.txt`.
4. Create the folder `src/agent/` containing an empty file named
  `__init__.py`. That empty file is how Python knows the folder is a
   package you can import from.
5. If you haven't already, run `pip install -e ".[dev]"` from the project
  root so that `import minesweeper` and `import agent` work from anywhere.

You will create these files, in this order:


| Order | File                         | Role                            |
| ----- | ---------------------------- | ------------------------------- |
| 1     | `src/agent/model.py`         | The CNN brain                   |
| 2     | `src/agent/replay_buffer.py` | The memory bank                 |
| 3     | `src/agent/agent.py`         | The decision-maker (DDQN logic) |
| 4     | `src/agent/train.py`         | The training loop               |
| 5     | `src/agent/evaluate.py`      | The exam                        |


---



## Part 1: The Brain — `model.py`

**Goal:** a neural network that takes the board picture, shape
`(batch, 5, height, width)`, and returns one Q-value per square, shape
`(batch, height * width)`.

### Spec

Build two classes, both subclassing `torch.nn.Module` (the PyTorch base
class for anything with learnable weights; you define layers in `__init__`
and the data flow in a method called `forward`):

`ResidualBlock` — a reusable building block.

- Constructor takes `channels` (int).
- Contains: two `nn.Conv2d` layers (`channels -> channels`, kernel size 3,
padding 1) and a `nn.ReLU`.
- `forward(x)`: pass `x` through conv1, ReLU, conv2 — then **add the
original input** `x` **back onto the result** — then ReLU, and return.
- That "add the input back" is the *residual shortcut*. It means the block
only has to learn a small correction instead of rebuilding the whole
signal, which makes deeper networks train much more easily.

`QNetwork` — the full brain.

- Constructor takes `in_channels=5`, `hidden=64`, `n_blocks=4`.
- Contains, in order:
  1. An input conv: `nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)`
  2. A stack of `n_blocks` ResidualBlocks (hint: `nn.Sequential` can hold a
    list of blocks and run them in order)
  3. A head conv: `nn.Conv2d(hidden, 1, kernel_size=1)` — a 1×1 conv that
    squashes 64 feature layers down to exactly 1 number per square
- `forward(x)`: input conv → ReLU → blocks → head. The head's output has
shape `(batch, 1, H, W)`; flatten everything after the batch dimension so
you return `(batch, H*W)`. (Look up `torch.flatten` or the tensor method
`.flatten(start_dim=1)`.)



### The one rule you must not break

**No fully connected (**`nn.Linear`**) layers anywhere.** A Linear layer locks
the network to one specific board size. Using only convolutions is what lets
the same trained brain play 9×9, 16×16, or any other board.

### Why `kernel_size=3, padding=1`?

The window is 3×3 (a square and its 8 neighbors — exactly the neighborhood
Minesweeper numbers describe). Padding 1 adds a border of zeros so the
output grid stays the same size as the input grid. Without padding, every
conv layer would shrink the board by 2 in each direction.

### Why does the flattening order matter?

The environment numbers squares left-to-right, top-to-bottom:
action = `row * width + col`. PyTorch's flatten happens to walk the grid in
exactly that same order, so output position `i` automatically lines up with
action `i`. You don't need to do anything special — just be aware this
alignment is load-bearing.

### Check yourself

```powershell
python -c "import torch; from agent.model import QNetwork; net = QNetwork(); print(net(torch.zeros(2, 5, 9, 9)).shape)"
```

Must print `torch.Size([2, 81])`. Then try input `(2, 5, 16, 16)` — you
should get `[2, 256]` from the *same* network with no code changes.

---



## Part 2: The Memory — `replay_buffer.py`

**Goal:** a fixed-size memory bank that stores experiences and hands back
random batches as GPU tensors.

### Spec

One class, `ReplayBuffer`. It does NOT need to be an `nn.Module` — it's just
plain Python + NumPy.

- Constructor takes `capacity` (int, e.g. 100,000), `obs_shape` (tuple,
e.g. `(5, 9, 9)`), and `n_actions` (int, e.g. 81).
- Stores six parallel NumPy arrays, pre-allocated at full capacity:


| Field        | Shape                    | Suggested dtype | Holds                           |
| ------------ | ------------------------ | --------------- | ------------------------------- |
| `obs`        | `(capacity, *obs_shape)` | `float16`       | board before the move           |
| `actions`    | `(capacity,)`            | `int64`         | which square was clicked        |
| `rewards`    | `(capacity,)`            | `float32`       | reward received                 |
| `next_obs`   | `(capacity, *obs_shape)` | `float16`       | board after the move            |
| `dones`      | `(capacity,)`            | `float32`       | 1.0 if the game ended, else 0.0 |
| `next_masks` | `(capacity, n_actions)`  | `bool`          | legal moves in the next state   |


  (`float16` for the two big observation arrays halves your RAM usage; the
  observations are just 0s, 1s, and eighths, so no precision is lost.)

- Method `push(obs, action, reward, next_obs, done, next_mask)`: write one
experience, ring-buffer style.
- Method `sample(batch_size, device)`: pick random rows and return them as
six PyTorch tensors on the given device. Convert the two observation
arrays back to `float32` when you make the tensors (networks want float32).
- Support `len(buffer)` — define `__len__` returning the current count.



### Pseudocode: the ring buffer

The "ring" trick with two counters — a write position that wraps around, and
a size that grows until full:

```
in the constructor:
    pos  = 0      # index where the NEXT experience will be written
    size = 0      # how many experiences are stored so far

push(experience):
    write each field of the experience into its array at index pos
    pos  = (pos + 1) mod capacity        # wraps back to 0 when it hits the end
    size = min(size + 1, capacity)       # grows until full, then stays put
```

Once `pos` wraps around, new experiences silently overwrite the oldest ones.
That's intentional: memories from 100,000 moves ago came from a much dumber
version of the AI and are no longer worth keeping.

### Pseudocode: sampling

```
sample(batch_size, device):
    idx = batch_size random integers in the range [0, size)   # NOT capacity!
    for each of the six arrays:
        take the rows at idx
        convert to a torch tensor on `device`
        (obs and next_obs: convert dtype to float32)
    return the six tensors
```



### Watch out

- Sample indices from `[0, size)`, **not** `[0, capacity)`. Before the buffer
is full, the tail of every array is all zeros — sampling those rows feeds
fake blank experiences into training and quietly poisons it.
- Store `done` as a float (1.0 / 0.0), not a bool. You'll multiply by
`(1 - done)` in the learning math later, and floats make that painless.



### Check yourself

Write a 5-line throwaway script: create a buffer with capacity 10, push 15
fake experiences (e.g. `np.zeros((5, 9, 9))` for the observations), then
check that `len(buffer)` is 10 and that `sample(4, torch.device("cpu"))`
returns 6 tensors with 4 rows each.

---



## Part 3: The Decision-Maker — `agent.py`

**Goal:** the DDQN core. Owns both networks, picks moves with masked
epsilon-greedy, and does the Double DQN learning update.

This is the hardest file. Take it slowly, and keep the target equation in
front of you the whole time:

```
target = reward + gamma × (value of best next move)      ... or just `reward` if the game ended
```



### Spec

One class, `DDQNAgent`.

- Constructor takes `device`, `gamma=0.99`, `lr=1e-4`,
`target_update_every=1000`. It must create:
  - `self.online` — a `QNetwork`, moved to the device (`.to(device)`)
  - `self.target` — a second `QNetwork`, also on the device, then
  **initialized to be an exact copy of online** (copy the weights across:
  look up `load_state_dict` and `state_dict`)
  - An Adam optimizer over `self.online.parameters()` — and *only* online's.
  The target network is never trained directly; it only ever receives
  copies.
  - A loss function: use `nn.SmoothL1Loss()` (also called Huber loss). It
  behaves like squared error for small mistakes but doesn't blow up on big
  ones — important because a surprise mine hit (−10) would otherwise
  produce a huge, destabilizing update.
  - A step counter for deciding when to refresh the target network.
- Method `act(obs, valid_mask, epsilon) -> int`
- Method `learn(batch) -> float` (returns the loss, handy for logging)
- Methods `save(path)` / `load(path)` — save/load `state_dict` of the online
network; on load, copy into *both* networks.



### Pseudocode: `act`

```
act(obs, valid_mask, epsilon):
    with probability epsilon:
        return a random choice among the indices where valid_mask is True
        # (NumPy: flatnonzero gives you those indices)

    otherwise:
        turn obs into a float32 tensor on the device
        add a batch dimension of 1                    # network expects (batch, 5, H, W)
        q = online network forward pass               # shape (1, H*W)
        remove the batch dimension                    # shape (H*W,)
        set q[squares where valid_mask is False] = -infinity
        return the index of the largest q
```

Two notes:

- Even the *random* branch must respect the mask. Otherwise the AI spends
its exploration budget clicking already-revealed squares.
- Decorate the method with `@torch.no_grad()` (or wrap the forward pass in
`with torch.no_grad():`). This tells PyTorch "no learning happening here,
don't record bookkeeping" — faster and lighter on memory.



### Pseudocode: `learn` (the heart of the whole project)

```
learn(batch):
    unpack: obs, actions, rewards, next_obs, dones, next_masks

    # STEP 1 - what does the student currently predict?
    q_all  = online(obs)                          # (batch, H*W): Q for every square
    q_pred = pick, for each row, the entry at that row's action
             # look up tensor.gather - it does exactly this row-wise pick

    # STEP 2 - build the answer key (no gradients here!)
    with no_grad:
        # -- Double DQN: online CHOOSES, target SCORES --
        next_q_online = online(next_obs)              # (batch, H*W)
        next_q_online[illegal squares per next_masks] = -infinity
        best_action   = argmax of next_q_online along the squares axis

        next_q_target = target(next_obs)              # (batch, H*W)
        best_value    = pick, per row, next_q_target at best_action   # gather again

        # a finished game can have NO legal next moves -> best_value may be -inf
        replace any -inf in best_value with 0.0       # look up torch.nan_to_num

        target_value = rewards + gamma * (1 - dones) * best_value

    # STEP 3 - nudge the student toward the answer key
    loss = huber_loss(q_pred, target_value)
    optimizer.zero_grad()          # clear leftover gradients from last step
    loss.backward()                # compute how each weight should change
    clip gradient norm to 10.0     # look up nn.utils.clip_grad_norm_
    optimizer.step()               # apply the nudge

    # STEP 4 - refresh the teacher occasionally
    step_counter += 1
    if step_counter is a multiple of target_update_every:
        copy online's weights into target

    return loss as a plain float
```



### Watch out — the four classic DDQN bugs

1. **Forgetting** `no_grad` **around Step 2.** Symptom: training runs but the
  network drifts into nonsense, because gradients flow through the answer
   key itself. The target is a *fact* to learn toward, not a thing to train.
2. **Using the target network for the argmax.** That's regular DQN, not
  Double DQN. The split is the whole point: online picks, target scores.
3. **Forgetting to mask** `next_q_online` **before the argmax.** The "best next
  move" must be a *legal* move, or your targets are built on moves that
   could never be played.
4. **Forgetting** `(1 - dones)`**.** When a game ended, there is no future — the
  target must be just the reward. Miss this and the AI hallucinates value
   after death, and mine hits stop scaring it properly.



### Check yourself

Throwaway script: create the agent on CPU, feed `act` a zero observation
with an all-True mask and `epsilon=0` — it should return an int in `[0, 81)`.
Then build one fake batch by hand (or push a few random transitions through
your ReplayBuffer and sample) and call `learn` — it should return a finite
float, no errors, no `nan`.

---



## Part 4: The Training Loop — `train.py`

**Goal:** the script that runs everything — plays thousands of games, stores
memories, learns after every move, prints progress, saves the best brain.

### Spec

- Use `argparse` for the settings so you can experiment without editing
code. Suggested arguments and defaults:
  - `--width 9`, `--height 9`, `--mines 10`
  - `--episodes 100000`
  - `--buffer-size 100000`, `--batch-size 128`
  - `--warmup 5000` (moves stored before learning starts)
  - `--eps-decay-steps 150000`, `--eps-final 0.05`
- Pick the device: `"cuda" if torch.cuda.is_available() else "cpu"`. Print
it, so you always know whether you're actually on the GPU.
- Build the three main objects: `MinesweeperEnv`, `DDQNAgent`,
`ReplayBuffer`. The env tells you the shapes the buffer needs — use
`env.observation_shape` and `env.action_space_size` rather than computing
them yourself.
- A helper function `evaluate(agent, env_settings, n_games)`: plays
`n_games` full games with `epsilon=0` on a *fresh* env and returns the
fraction won. (The `info` dict from `env.step` has a `"won"` key.)
- Make a `checkpoints/` folder (`os.makedirs(..., exist_ok=True)`).



### Pseudocode: the main loop

```
for episode = 1 .. episodes:
    obs = env.reset()
    done = False

    while not done:
        epsilon = current value on the decay schedule        (see below)

        mask   = env.valid_actions_mask()
        action = agent.act(obs, mask, epsilon)
        next_obs, reward, done, info = env.step(action)

        buffer.push(obs, action, reward, next_obs, done, info["valid_actions"])
        #                                  the mask for the NEXT state ^^^
        obs = next_obs
        total_steps += 1

        if len(buffer) >= warmup:
            agent.learn(buffer.sample(batch_size, device))

    record whether this episode was won, and its total reward
    # a deque with maxlen=500 gives you cheap rolling statistics

    every 500 episodes:
        win_rate = evaluate(agent, env_settings, 200 games)
        print episode, total_steps, epsilon, rolling train win%, rolling avg reward, win_rate
        if win_rate > best so far:
            remember it and agent.save("checkpoints/best.pt")

after the loop: agent.save("checkpoints/final.pt")
```



### Pseudocode: the epsilon schedule

A straight line from 1.0 down to `eps_final`, then flat:

```
fraction = total_steps / eps_decay_steps
epsilon  = 1.0 - (1.0 - eps_final) * fraction
epsilon  = max(epsilon, eps_final)          # clamp so it never goes below the floor
```

Decay by *steps* (moves), not episodes — early episodes are only a few moves
long, so episode-based decay would move much faster than you intend.

### Watch out

- **Push the mask of the NEXT state**, which conveniently arrives in
`info["valid_actions"]` from the step you just took. Pushing the mask of
the *current* state (the one you used to pick the action) is an
off-by-one bug that corrupts every Double DQN target. This is the
sneakiest bug in the whole project.
- Update `obs = next_obs` at the end of the loop body. Forgetting it means
the agent acts on a stale board forever.
- The warmup gate (`len(buffer) >= warmup`) matters: learning from a nearly
empty buffer means seeing the same few experiences over and over, and the
network memorizes them like a student cramming three flash cards.



### Check yourself — the smoke test

```powershell
python -m agent.train --episodes 1000 --warmup 500 --eps-decay-steps 5000
```

Success = progress lines print, no crashes, loss values are finite. Win
rates near 0% are completely normal after only 1,000 games.

---



## Part 5: The Exam — `evaluate.py`

**Goal:** load a saved checkpoint and measure it properly, on any board size.

### Spec

- Arguments: `--checkpoint` (default `checkpoints/best.pt`), `--width`,
`--height`, `--mines`, `--games` (default 1000).
- Create the agent, call `agent.load(checkpoint)`.
- Play the games exactly like the `evaluate` helper in train.py
(`epsilon=0` always), but track two numbers:
  - **Win rate** — fraction of games won.
  - **Board completion** — total safe squares revealed across all games,
  divided by (games × safe squares per board). The env exposes
  `env.board.safe_cell_count`, and `info["revealed_count"]` gives the
  final count per game. Completion is the kinder metric early on: an AI
  that clears 60% of the board but dies is much better than one that dies
  on move two, even though both have won 0 games.
- Print both, nicely formatted.



### Check yourself

After the smoke test from Part 4 saved a checkpoint:

```powershell
python -m agent.evaluate --games 100
python -m agent.evaluate --games 100 --width 16 --height 16 --mines 40
```

The second command is the payoff of the no-Linear-layers rule: your
9×9-trained brain playing a board it has never seen. (Expect it to be bad
after a smoke-test-sized training run — the point is that it *runs*.)

---



## Part 6: The Real Training Run

```powershell
python -m agent.train
```

What to expect on 9×9 with 10 mines:

- **First ~30 minutes:** eval win rate near 0%. Watch avg reward instead —
it should creep upward as the AI learns to survive longer.
- **A few hours:** eval win rate climbing through 10–30%.
- **Plateau:** DQN-style agents typically land somewhere in 40–80% on
beginner boards depending on tuning and patience. Remember that some
games *force* a 50/50 guess — even perfect play can't win them all.

`Ctrl+C` is always safe: the best brain so far is already on disk.

### If learning stalls (try one at a time)

1. **More exploration time:** `--eps-decay-steps 300000`.
2. **Cleaner signal:** pass `no_guess=True` when creating the training env —
  the repo's board generator will then only serve boards solvable by pure
   logic, so the AI is never punished for an unavoidable coin flip.
3. **Curriculum:** train with `--mines 5` first, then load that checkpoint
  and continue with 10 mines. Early wins are where the strongest learning
   signal lives.
4. **Loss bouncing wildly?** Lower the learning rate (`lr=5e-5`).
5. **Suspect the brain is too small?** `n_blocks=6` or `hidden=128` in the
  QNetwork. Slower per step, still fine on 4GB.

---



## Debugging Guide: symptom → likely cause


| Symptom                               | First thing to check                                                                                                                            |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Shape error mentioning `(batch, ...)` | Forgot to add/remove the batch dimension in `act`                                                                                               |
| Loss is `nan` or `inf`                | `-inf` leaking out of masking into the target — check the `nan_to_num` guard, and that at least one action is legal wherever you take an argmax |
| Loss falls but win rate never moves   | Off-by-one on the pushed mask (must be the NEXT state's mask), or targets built without masking                                                 |
| Win rate rises then collapses         | Target network refreshed too often (or never) — verify the step counter logic                                                                   |
| AI keeps clicking revealed squares    | Mask not applied in `act` — check both the random branch and the greedy branch                                                                  |
| GPU out of memory                     | Buffer arrays accidentally created as torch tensors on the GPU — they must be NumPy in ordinary RAM                                             |
| `import agent` fails                  | Missing `__init__.py`, or you skipped `pip install -e .`                                                                                        |
| Trains but is very slow               | Printed device says `cpu` — the CUDA build of torch isn't installed                                                                             |


---



## Suggested pacing

- **Session 1:** Part 0 + Part 1 (setup and the brain). Short and satisfying.
- **Session 2:** Part 2 (memory). Easiest file — pure Python and NumPy.
- **Session 3:** Part 3 (decision-maker). The hard one. Budget real time,
keep the target equation visible, and test as you go.
- **Session 4:** Parts 4–5 (training loop and exam), then the smoke test.
- **Session 5:** Kick off the real run and enjoy the graphs.

