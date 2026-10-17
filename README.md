# MazeMind RL — Intelligent Maze Navigation Using Q-Learning

> **Hackathon Project** | Domain: Artificial Intelligence · Machine Learning · Reinforcement Learning

---

## 1. Project Overview

**MazeMind RL** is a fully interactive Reinforcement Learning demonstration that trains an autonomous agent to navigate a configurable grid maze using **tabular Q-Learning implemented entirely from scratch**. The project provides a professional Streamlit dashboard where users can:

- Configure the maze, rewards, and hyperparameters live.
- Watch the agent learn episode-by-episode with real Q-value updates.
- Inspect every Q-learning transition with the full numerical update equation.
- Compare the trained agent against a random baseline.
- Export training metrics, Q-tables, and evaluation results.

---

## 2. Problem Statement

A maze is a discrete grid where an agent starts at position (0, 0) and must reach the goal at (5, 5) while avoiding obstacles. The agent has no prior knowledge of the maze structure and must discover the optimal path purely through trial-and-error interaction with the environment.

**Challenge:** Design a learning algorithm that, given only reward signals, converges to a reliable goal-reaching policy.

---

## 3. Reinforcement Learning Formulation

| Component | Definition |
|-----------|-----------|
| **Agent** | The RL learner navigating the maze |
| **Environment** | 6×6 grid with obstacles, start, and goal |
| **State** | Current (row, col) position → linearised integer index |
| **Action space** | {Up, Down, Left, Right} — 4 discrete actions |
| **Reward** | +100 goal, −1 step, −5 wall collision |
| **Episode** | Starts at (0,0), ends when goal is reached or step limit hit |
| **Policy** | Mapping from states to actions derived from the Q-table |

---

## 4. Q-Learning Algorithm

The agent uses the standard temporal-difference update:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

**Terminal transitions** (goal reached or step limit):
```
Q(s,a) ← Q(s,a) + α · [r − Q(s,a)]
```
No future bootstrapping is applied when the episode ends.

**Action selection:** ε-greedy — random with probability ε, greedy otherwise.

**Epsilon decay:** `ε ← max(ε_min, ε × ε_decay)` after each episode.

---

## 5. Default Hyperparameters

| Parameter | Default | Justification |
|-----------|---------|---------------|
| Learning rate α | 0.1 | Stable convergence on tabular problems |
| Discount factor γ | 0.95 | Values near-future rewards highly |
| Initial epsilon | 1.0 | Full exploration at start |
| Minimum epsilon | 0.05 | Retains minimal exploration indefinitely |
| Epsilon decay | 0.995 | Gradual transition to exploitation over ~1000 eps |
| Training episodes | 1500 | Sufficient for 6×6 maze convergence |
| Max steps / episode | 100 | Prevents infinite loops (BFS optimal ≈ 10 steps) |
| Batch size | 25 | UI responsiveness vs training speed |

---

## 6. Installation

```bash
# Clone or copy the project directory
cd MazeMind

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+ (tested on 3.9.6 and 3.11).

---

## 7. Running the Application

```bash
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.

**Demo workflow (no configuration needed):**
1. Open the app — the default solvable maze loads automatically.
2. Click **▶ Train Agent** in the Maze Simulator tab.
3. Watch the training progress charts update live.
4. After training, go to **Q-Table Explorer** to inspect learned values.
5. Use **Step Once** or **Play Anim** to watch the greedy policy.
6. Go to **Evaluation** and run both baselines for comparison.
7. Export results from the **Export & History** tab.

---

## 8. Dashboard Features

### A. KPI Cards
Real-time metrics: current episode, latest reward, moving-average reward (50-episode window), success rate, average steps to goal, current ε, successful episode count, and maximum Q-value.

### B. Maze Simulator
Interactive grid showing start (S), goal (G), obstacles, agent position (★), visited cells, and evaluation path. Controls: Train / Pause / Resume / Stop / Reset Env / Reset Learning / Test Agent / Step Once / Play Animation. Optional policy-arrow overlay.

### C. Training Progress
Four interactive Plotly charts:
1. Episode reward + moving-average overlay.
2. Steps per episode.
3. Rolling goal-reaching success rate.
4. Epsilon decay curve.

### D. Q-Table Explorer
- Full Q-table (states × actions) with best-action highlighting.
- Per-action heatmap across the grid.
- Value function V(s) = max_a Q(s,a) heatmap.
- Cell inspector: select any grid cell to view its four Q-values and best action.

### E. Learning Process Inspector
Select any recorded transition (up to 200 stored) and see:
- State, action, reward, next state.
- Old Q-value, target, TD error, new Q-value.
- Full numerical update equation with real values.
- Before/after Q-value bar chart for that state.

### F. Performance Evaluation
Independent evaluation of:
1. **Random policy:** uniform random action selection.
2. **Trained Q-agent:** greedy policy from learned Q-table.

Reports: success rate, avg steps (successes), avg reward, avg episode length.  
Side-by-side comparison bar chart and relative improvement table.

### G. Configuration
All hyperparameters, maze dimensions, obstacle layout, reward values, and evaluation settings are adjustable from the sidebar. Invalid configurations are validated and rejected with error messages.

### H. Export & History
Download: training metrics CSV, Q-table CSV, agent JSON (save/load), evaluation JSON/CSV.
Upload and restore a previously saved agent JSON (compatibility validated).

---

## 9. Evaluation Methodology

- Both policies run on **identical** environment instances (same maze, start, goal, rewards, step limit).
- The Q-table is **never updated** during evaluation.
- Success rate = (episodes reaching goal) / (total evaluation episodes) × 100%.
- This is not a classification metric — it measures navigation performance.

---

## 10. Libraries Used

| Library | Purpose |
|---------|---------|
| Streamlit | Interactive web dashboard |
| NumPy | Q-table computation, array operations |
| Pandas | Metrics DataFrames, CSV export |
| Plotly | Interactive charts and heatmaps |
| Python stdlib | BFS, session management, JSON I/O |

---

## 11. Limitations

- Tabular Q-Learning does not scale to large state spaces (large grids).
- Training runs synchronously in the Streamlit process (no true background threads).
- Animation uses `time.sleep` — not a true async animation.
- Maze editing requires rebuilding the environment (Q-table is reset).
- Only cardinal movement — no diagonal actions.

---

## 12. Future Enhancements

- Deep Q-Network (DQN) for larger/image-based mazes.
- Multi-agent scenarios.
- A* comparison as an additional baseline.
- SARSA and Double Q-Learning variants.
- Maze generation algorithms (recursive backtracking, Prim's).
- Persistent experiment database (SQLite).
- True asynchronous training via threading or multiprocessing.

---

## 13. Suggested Division of Work

| Member | Responsibility |
|--------|---------------|
| Member 1 | Environment design, reward structure, BFS solvability |
| Member 2 | Q-Learning algorithm, agent, training loop |
| Member 3 | Streamlit dashboard, KPI cards, maze visualization |
| Member 4 | Evaluation framework, charts, export, README |

---

## 14. Project Structure

```
MazeMind/
├── app.py            # Streamlit dashboard (main entry point)
├── environment.py    # Grid maze environment (MDP)
├── agent.py          # Q-Learning agent
├── training.py       # Training loop and metrics
├── evaluation.py     # Random baseline and policy evaluation
├── visualization.py  # Plotly chart helpers
├── requirements.txt  # Python dependencies
├── README.md         # This document
└── results/          # Output directory for exports
```

---

*All Q-values and training metrics displayed in the application are computed from real agent–environment interactions. No results are fabricated or hardcoded.*
