"""
environment.py — MazeMind RL
Grid-based maze environment for Q-Learning navigation.

Provides a fully observable, discrete, grid-based MDP:
  - States  : (row, col) tuples, linearised to integer indices.
  - Actions : 0=Up, 1=Down, 2=Left, 3=Right.
  - Rewards : configurable per-move, per-collision, and goal reward.
  - Terminal: reaching the goal cell ends the episode.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional

# ── Action constants ──────────────────────────────────────────────────────────
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
ACTION_NAMES  = {UP: "Up", DOWN: "Down", LEFT: "Left", RIGHT: "Right"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
N_ACTIONS = 4


class MazeEnvironment:
    """
    Grid maze environment.

    Parameters
    ----------
    rows, cols   : grid dimensions (default 6×6).
    start        : starting cell (row, col).
    goal         : goal cell (row, col).
    obstacles    : list of obstacle cells.
    reward_goal  : reward for reaching the goal.
    reward_step  : reward per ordinary move.
    reward_wall  : reward for hitting a wall / obstacle.
    max_steps    : step limit per episode (prevents infinite loops).
    seed         : RNG seed for reproducibility (None = random).
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 6,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (5, 5),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        reward_goal: float = 100.0,
        reward_step: float = -1.0,
        reward_wall: float = -5.0,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start = tuple(start)
        self.goal  = tuple(goal)
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_wall = reward_wall
        self.max_steps   = max_steps

        # Default obstacle layout for a 6×6 grid (a solvable maze).
        if obstacles is None:
            obstacles = self._default_obstacles()
        self.obstacles: set = set(map(tuple, obstacles))

        self._validate_config()

        self.rng = np.random.default_rng(seed)

        # Episode state
        self.agent_pos: Tuple[int, int] = self.start
        self.step_count: int = 0
        self.done: bool = False
        self.visited: List[Tuple[int, int]] = [self.start]

    # ── Configuration ─────────────────────────────────────────────────────────

    @staticmethod
    def _default_obstacles() -> List[Tuple[int, int]]:
        """Solvable 6×6 maze — verified via BFS from (0,0) to (5,5)."""
        return [
            (0, 3), (0, 4),
            (1, 1), (1, 2),
            (2, 4), (2, 5),
            (3, 1), (3, 2), (3, 3),
            (4, 3), (4, 4),
        ]


    def _validate_config(self) -> None:
        errors = []
        if not (0 <= self.start[0] < self.rows and 0 <= self.start[1] < self.cols):
            errors.append(f"Start {self.start} is out of bounds ({self.rows}×{self.cols}).")
        if not (0 <= self.goal[0] < self.rows and 0 <= self.goal[1] < self.cols):
            errors.append(f"Goal {self.goal} is out of bounds ({self.rows}×{self.cols}).")
        if self.start == self.goal:
            errors.append("Start and goal must differ.")
        if tuple(self.start) in self.obstacles:
            errors.append("Start cell must not be an obstacle.")
        if tuple(self.goal) in self.obstacles:
            errors.append("Goal cell must not be an obstacle.")
        if errors:
            raise ValueError("\n".join(errors))

    # ── State encoding ────────────────────────────────────────────────────────

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to a flat integer index."""
        return state[0] * self.cols + state[1]

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat integer index to (row, col)."""
        return (index // self.cols, index % self.cols)

    # ── Episode control ───────────────────────────────────────────────────────

    def reset(self) -> int:
        """Reset environment to start; return start state index."""
        self.agent_pos  = self.start
        self.step_count = 0
        self.done       = False
        self.visited    = [self.start]
        return self.state_to_index(self.agent_pos)

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute action; return (next_state_index, reward, done, info).

        Terminal transitions must NOT bootstrap from next-state Q-values.
        The `done` flag signals this to the agent.
        """
        if self.done:
            raise RuntimeError("Episode is already done. Call reset() first.")

        prev_pos = self.agent_pos          # capture BEFORE moving
        self.step_count += 1
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # ── Boundary / obstacle collision ─────────────────────────────────
        if (
            nr < 0 or nr >= self.rows
            or nc < 0 or nc >= self.cols
            or (nr, nc) in self.obstacles
        ):
            # Agent stays in place; penalise.
            next_pos  = self.agent_pos
            reward    = self.reward_wall
            done      = False
        else:
            next_pos = (nr, nc)

            if next_pos == self.goal:
                reward = self.reward_goal
                done   = True
            else:
                reward = self.reward_step
                done   = False

        # Step limit reached (failed episode, not goal).
        if not done and self.step_count >= self.max_steps:
            done = True

        self.agent_pos = next_pos
        self.done      = done
        if next_pos not in self.visited:
            self.visited.append(next_pos)

        info = {
            "step":         self.step_count,
            "prev_pos":     prev_pos,
            "next_pos":     next_pos,
            "action":       action,
            "done":         done,
            "hit_wall":     next_pos == prev_pos,
            "reached_goal": next_pos == self.goal and done,
        }

        return self.state_to_index(next_pos), reward, done, info


    # ── Utility ───────────────────────────────────────────────────────────────

    def get_valid_actions(self, state: Optional[Tuple[int, int]] = None) -> List[int]:
        """Return list of actions that do not immediately hit a wall/obstacle."""
        pos = state if state is not None else self.agent_pos
        valid = []
        for a, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if (
                0 <= nr < self.rows
                and 0 <= nc < self.cols
                and (nr, nc) not in self.obstacles
            ):
                valid.append(a)
        return valid

    def is_solvable(self) -> bool:
        """BFS check: can the agent reach the goal from start?"""
        from collections import deque
        visited = set()
        queue   = deque([self.start])
        visited.add(self.start)
        while queue:
            pos = queue.popleft()
            if pos == self.goal:
                return True
            for a, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = pos[0] + dr, pos[1] + dc
                npos   = (nr, nc)
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and npos not in self.obstacles
                    and npos not in visited
                ):
                    visited.add(npos)
                    queue.append(npos)
        return False

    def shortest_path(self) -> Optional[List[Tuple[int, int]]]:
        """BFS shortest path from start to goal. Returns None if unsolvable."""
        from collections import deque
        parent  = {self.start: None}
        queue   = deque([self.start])
        while queue:
            pos = queue.popleft()
            if pos == self.goal:
                path = []
                while pos is not None:
                    path.append(pos)
                    pos = parent[pos]
                return list(reversed(path))
            for a, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = pos[0] + dr, pos[1] + dc
                npos   = (nr, nc)
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and npos not in self.obstacles
                    and npos not in parent
                ):
                    parent[npos] = pos
                    queue.append(npos)
        return None

    def config_dict(self) -> Dict:
        """Serialisable configuration snapshot."""
        return {
            "rows":         self.rows,
            "cols":         self.cols,
            "start":        list(self.start),
            "goal":         list(self.goal),
            "obstacles":    [list(o) for o in sorted(self.obstacles)],
            "reward_goal":  self.reward_goal,
            "reward_step":  self.reward_step,
            "reward_wall":  self.reward_wall,
            "max_steps":    self.max_steps,
        }
