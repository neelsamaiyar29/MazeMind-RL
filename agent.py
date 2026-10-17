"""
agent.py — MazeMind RL
Tabular Q-Learning agent with epsilon-greedy exploration.

Implements the standard update equation:
  Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

For terminal transitions (done=True) the target is simply r — no
bootstrapping from s'.

Parameters are fully configurable. The Q-table and hyperparameters are
stored together so that saving/loading remains self-consistent.
"""

from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Optional, Tuple


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    n_states  : total number of discrete states.
    n_actions : number of actions (4 for UP/DOWN/LEFT/RIGHT).
    alpha     : learning rate (0 < α ≤ 1).
    gamma     : discount factor (0 ≤ γ < 1).
    epsilon   : initial exploration rate.
    epsilon_min : minimum exploration rate.
    epsilon_decay : multiplicative decay applied after each episode.
    seed      : RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        self.n_states     = n_states
        self.n_actions    = n_actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        self._seed = seed

        # Q-table: shape (n_states, n_actions), initialised to zero.
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

        # Last update record — used by the Learning Process Inspector.
        self.last_transition: Optional[Dict] = None

    # ── Policy ─────────────────────────────────────────────────────────────

    def select_action(self, state: int, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state   : current state index.
        greedy  : if True, always pick the best action (no exploration).
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._greedy_action(state)

    def _greedy_action(self, state: int) -> int:
        """Return the action with the highest Q-value; break ties uniformly."""
        q_values = self.q_table[state]
        max_q    = np.max(q_values)
        # All actions tied at max_q — pick uniformly among them.
        best     = np.where(q_values == max_q)[0]
        return int(self.rng.choice(best))

    # ── Learning ────────────────────────────────────────────────────────────

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> Dict:
        """
        Apply one Q-learning update.

        Returns a dict with all values used in the update — shown in the
        Learning Process Inspector.
        """
        old_q = self.q_table[state, action]

        if done:
            # Terminal transition: target = immediate reward only.
            max_next_q = 0.0
            target     = reward
        else:
            max_next_q = float(np.max(self.q_table[next_state]))
            target     = reward + self.gamma * max_next_q

        td_error = target - old_q
        new_q    = old_q + self.alpha * td_error

        self.q_table[state, action] = new_q

        record = {
            "state":      state,
            "action":     action,
            "reward":     reward,
            "next_state": next_state,
            "done":       done,
            "old_q":      old_q,
            "max_next_q": max_next_q,
            "target":     target,
            "td_error":   td_error,
            "new_q":      new_q,
            "alpha":      self.alpha,
            "gamma":      self.gamma,
        }
        self.last_transition = record
        return record

    def decay_epsilon(self) -> None:
        """Decay epsilon after each training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Utilities ───────────────────────────────────────────────────────────

    def reset_qtable(self) -> None:
        """Zero the Q-table and restore epsilon to its initial value."""
        self.q_table  = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.epsilon  = 1.0   # reset to full exploration
        self.last_transition = None

    def get_greedy_policy(self) -> np.ndarray:
        """Return a (n_states,) array of greedy actions."""
        max_q  = np.max(self.q_table, axis=1)
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            best = np.where(self.q_table[s] == max_q[s])[0]
            policy[s] = int(self.rng.choice(best))
        return policy

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialise agent state for saving."""
        return {
            "n_states":      self.n_states,
            "n_actions":     self.n_actions,
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "seed":          self._seed,
            "q_table":       self.q_table.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "QLearningAgent":
        """Restore an agent from a saved dict."""
        agent = cls(
            n_states      = d["n_states"],
            n_actions     = d["n_actions"],
            alpha         = d["alpha"],
            gamma         = d["gamma"],
            epsilon       = d["epsilon"],
            epsilon_min   = d["epsilon_min"],
            epsilon_decay = d["epsilon_decay"],
            seed          = d.get("seed"),
        )
        agent.q_table = np.array(d["q_table"], dtype=np.float64)
        return agent

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "QLearningAgent":
        return cls.from_dict(json.loads(s))
