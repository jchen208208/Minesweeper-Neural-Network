# Minesweeper Solver + Neural Network — 2-Week Sprint Plan

*Goal: a constraint-logic Minesweeper solver, a CNN that tries to out-guess it, benchmarked win rates, and a live "watch it think" demo.*
*Pace: full-time summer sprint, ~40–50 hrs/week, 14 days. Cost: $0 (friend's 16GB GPU covers training).*

---

## The Learning Contract (read before every session)

This project only counts if you can defend every line in an interview. Working rules with Claude:

1. **Concepts before code.** At the start of each phase, have Claude explain the ideas until you could teach them to a friend. No code yet.
2. **You write the first attempt.** Pseudocode from Claude is allowed. Runnable Python is not.
3. **Claude reviews, points at bugs, never rewrites.** Debugging your own code is where the learning and the war stories live.
4. **45-minute rule.** Stuck longer than that → ask a *targeted* question ("why does my flood fill recurse forever?"), not "fix this."
5. **Phase-exit quiz.** End of each phase, Claude grills you on your own code like an interviewer. Can't explain it → not done. Speed compresses typing, never understanding — quizzes are non-negotiable at sprint pace.
6. **Commit daily with real messages.** Your git history is evidence of sustained effort — recruiters look.

**Slip rule:** if you're more than a day behind by Day 6, stretch to a 3-week pace — never skip quizzes or cut corners to stay on schedule. And remember the safety net: **a benchmarked solver (end of Day 6) is already a complete, shippable project.**

---

## Days 1–2 — Foundations + Game Engine

**Day 1 morning — setup & warm-ups:**
- Python 3.11+, VS Code, fresh git repo (`minesweeper-ai`) pushed to GitHub with a stub README.
- Self-check, written from scratch with no help:
  - Print a 10×10 grid of `-` characters
  - A `Counter` class with `increment()` and `reset()`
  - A recursive function summing a nested list like `[1,[2,[3]],4]`
- Shaky on any → drill that topic with Claude before proceeding. A few hours here saves days later.

**Day 1 afternoon → Day 2 — the engine, in order:**
1. `Board` class: width, height, mine count; random mine placement with a **seedable RNG** (seeding makes bugs reproducible — from day one).
2. Neighbor-count computation for every cell. Pick a `grid[row][col]` convention now and never break it.
3. **First-click safety:** place mines only *after* the first reveal, excluding that cell and its neighbors.
4. Reveal with **flood fill:** a 0-cell auto-reveals neighbors, cascading. Learn BFS with an explicit queue *and* the recursive version; know why recursion depth bites on big boards.
5. Flagging, win detection (all safe cells revealed), loss detection.
6. Terminal rendering + input loop → play your own game.
7. Unit tests (`pytest`): a seeded layout, a flood-fill case computed by hand, a first-click-safety test.

**Concepts:** 2D indexing, classes, BFS/DFS, encapsulation (true state vs. what a player may see — critical in the next phase), border/corner edge cases.
**Bugs you'll probably meet (that's the point):** infinite flood-fill recursion (no visited set), miscounted neighbors on edges, revealing flagged cells.
**Done when:** you can play and win in the terminal, tests pass, and the flood-fill quiz feels easy.

---

## Days 3–6 — Logic Solver (the intellectual heart)

**Days 3–4 — deduction:**
1. Build a `SolverView`: the solver sees **only** revealed numbers and flags — never the true mine grid. Letting it peek is the classic silent bug that fakes a 100% win rate and later poisons your training data. Write a test that proves it can't.
2. **Rule A:** a number equal to its count of unrevealed neighbors → all of them are mines (flag).
3. **Rule B:** a number equal to its count of flagged neighbors → remaining unrevealed neighbors are safe (reveal).
4. Loop both rules to a **fixpoint** (a full pass changes nothing). This alone beats most human players on beginner boards.

*(Sprint cut: the subset/1-2-pattern rule is skipped — list it in the README as future work.)*

**Days 5–6 — exact probabilities + benchmark:**
5. Collect boundary constraints; split into **independent connected components** (this is what keeps enumeration tractable — understand why before coding it).
6. Per component: backtracking enumeration of all mine assignments consistent with every constraint; count how often each cell is a mine → exact P(mine).
7. Guess policy: no certain move → open the minimum-probability cell. Include off-boundary cells (P = remaining mines / remaining hidden cells).
8. **Day 6 — benchmark harness:** 10,000 games per difficulty; report win rate + 95% confidence interval in the README. Reference points: strong solvers reach roughly ~90% beginner / ~75% intermediate / ~35–40% expert. Being in the neighborhood is success; without the subset rule you'll land somewhat below the best — that's expected and honest.

**Concepts:** constraint propagation, fixpoint iteration, backtracking search, why decomposition tames exponential blowup, basic probability, confidence intervals.
**Done when:** the benchmark table is in the README and you can walk through Rules A/B and the enumeration on a whiteboard.

---

## Days 7–8 — Dataset Generation

1. **Board encoding** as channel planes over the H×W grid: one-hot planes for revealed numbers 0–8, an "unrevealed" mask plane, a flag plane. (Know *why* one-hot beats feeding raw 0–8 values — real interview question.)
2. Instrument the solver: every time it's *forced to guess*, dump (encoded state, full ground-truth mine grid) to disk as `.npz` shards.
3. Generate **~200k guess situations** across difficulties (sprint-sized; scale up later if results warrant). Also save the solver's computed probabilities for each state — that's the baseline to beat, stored next to the data.
4. Split train/validation/test **by game**, not by position — positions within one game are correlated; splitting by position leaks.

**Concepts:** tensor thinking, data leakage, class imbalance (most cells aren't mines), reproducible pipelines.
**Done when:** one script + one seed regenerates everything, and you've visually spot-checked encodings against the boards they came from.

---

## Days 9–12 — The CNN (GPU days)

**Days 9–10 — learn PyTorch first. Do not skip.**
- Tensors, autograd, `nn.Module`, the training loop. Build the toy XOR example, then a tiny MNIST-style exercise, before touching your real data. These two days are the difference between training a model and copy-pasting one.

**Day 11 — train:**
1. Model: small **fully-convolutional** net (4–6 conv layers, 3×3 kernels, padding to preserve H×W), sigmoid output = P(mine) per cell. Fully-convolutional means one model handles all board sizes.
2. Loss: binary cross-entropy **masked to unrevealed cells only** (revealed cells are known — training on them wastes capacity and inflates metrics).
3. Train on the 16GB GPU. Log train/val loss curves; know how you'd *detect* overfitting even if it doesn't happen.

*(Sprint cut: ablation studies — future work.)*

**Day 12 — the experiment:**
4. Head-to-head on identical seeded games: (a) solver-certain moves + **solver-probability guessing** vs. (b) solver-certain moves + **NN guessing**. Same logic core, only the guess policy differs — a clean controlled experiment.
5. Report win rates with confidence intervals per difficulty, plus per-guess accuracy (NN calibration vs. the solver's exact local probabilities).
6. Whatever the outcome — NN wins, loses, or ties — it's *your* empirical finding. A rigorous negative result is still a great writeup.

**Concepts:** convnets and receptive fields, masked losses, calibration, controlled experiments.
**Done when:** the head-to-head table exists and you can explain *why* the winner wins (what global patterns can a CNN see that local constraint math can't?).

---

## Days 13–14 — Demo + Writeup

1. **Streamlit (or simple web) demo:** the board with the NN's live probability heatmap overlaid — cells glowing by mine-confidence, agent playing automatically. Your 30-second interview demo.
2. **README:** what it is, the solver benchmark table, the head-to-head result, a GIF of the heatmap, how to run it.
3. **Short writeup / blog post:** the single hardest bug of the sprint + the experiment result. One honest post beats ten feature lists.

*(Sprint cut: the VS Code extension — pure garnish; do it on a slow weekend later.)*

**Resume bullet you're building toward (fill in real numbers):**
> *Built a Minesweeper AI combining an exact constraint-satisfaction solver (X% expert win rate over 10k games) with a CNN guess-policy trained on 200k self-generated positions, changing win rate by Y%; shipped a live probability-heatmap demo.*

---

## Future Work (post-sprint, whenever)

- Subset/1-2-pattern deduction rule → pushes solver win rate toward published bests
- Ablations: depth/width/dataset-size vs. accuracy
- Scale dataset to 1M positions
- VS Code extension demo
