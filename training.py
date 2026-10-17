"""
training.py — MazeMind RL
Training loop and metric collection for the Q-Learning agent.

Supports:
  - Episode-by-episode or batch training.
  - Pause / resume / stop via shared control flags (stored in session state).
  - Detailed transition history for the Learning Process Inspector.
  - Rolling metric computations (moving average reward, success rate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent import QLearningAgent
from environment import MazeEnvironment

# Maximum recent transitions kept in memory (prevent unbounded growth).
MAX_TRANSITION_HISTORY = 200


@dataclass
class EpisodeMetrics:
    """Metrics recorded at the end of each training episode."""
    episode:        int
    total_reward:   float
    steps:          int
    reached_goal:   bool
    epsilon:        float
    timestamp:      float = field(default_factory=time.time)


@dataclass
class TrainingState:
    """
    Mutable training state shared between the training loop and the UI.

    Stored in st.session_state so it persists across Streamlit reruns.
    """
    metrics:            List[EpisodeMetrics] = field(default_factory=list)
    transition_history: List[Dict]           = field(default_factory=list)
    current_episode:    int                  = 0
    total_episodes:     int                  = 0
    is_training:        bool                 = False
    is_paused:          bool                 = False
    stop_requested:     bool                 = False
    training_complete:  bool                 = False
    current_path:       List[Tuple]          = field(default_factory=list)
    last_eval_path:     List[Tuple]          = field(default_factory=list)

    def reset(self) -> None:
        """Clear all recorded metrics and paths."""
        self.metrics.clear()
        self.transition_history.clear()
        self.current_episode   = 0
        self.is_training       = False
        self.is_paused         = False
        self.stop_requested    = False
        self.training_complete = False
        self.current_path.clear()
        self.last_eval_path.clear()

    def add_transition(self, record: Dict) -> None:
        self.transition_history.append(record)
        if len(self.transition_history) > MAX_TRANSITION_HISTORY:
            self.transition_history.pop(0)

    def rolling_mean_reward(self, window: int = 50) -> List[float]:
        rewards = [m.total_reward for m in self.metrics]
        if not rewards:
            return []
        result = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            result.append(float(np.mean(rewards[start : i + 1])))
        return result

    def rolling_success_rate(self, window: int = 50) -> List[float]:
        successes = [1.0 if m.reached_goal else 0.0 for m in self.metrics]
        if not successes:
            return []
        result = []
        for i in range(len(successes)):
            start = max(0, i - window + 1)
            result.append(float(np.mean(successes[start : i + 1])) * 100.0)
        return result

    @property
    def success_rate(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(1 for m in self.metrics if m.reached_goal) / len(self.metrics) * 100.0

    @property
    def latest_reward(self) -> Optional[float]:
        return self.metrics[-1].total_reward if self.metrics else None

    @property
    def moving_avg_reward(self) -> Optional[float]:
        rma = self.rolling_mean_reward(50)
        return rma[-1] if rma else None

    @property
    def avg_steps_success(self) -> Optional[float]:
        good = [m.steps for m in self.metrics if m.reached_goal]
        return float(np.mean(good)) if good else None

    @property
    def n_successful(self) -> int:
        return sum(1 for m in self.metrics if m.reached_goal)


def run_episode(
    env: MazeEnvironment,
    agent: QLearningAgent,
    training_state: TrainingState,
    greedy: bool = False,
    record_transitions: bool = True,
) -> EpisodeMetrics:
    """
    Run one complete episode.

    Parameters
    ----------
    greedy              : if True, use greedy policy (no exploration, no updates).
    record_transitions  : if True, append each transition to training_state.transition_history.

    Returns EpisodeMetrics for the episode.
    """
    state = env.reset()
    total_reward = 0.0
    reached_goal = False
    path = [env.agent_pos]

    for _ in range(env.max_steps):
        action      = agent.select_action(state, greedy=greedy)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        if not greedy:
            update_record = agent.update(state, action, reward, next_state, done)
            update_record["pos"]      = info.get("prev_pos", env.agent_pos)
            update_record["next_pos"] = info.get("next_pos", env.agent_pos)
            if record_transitions:
                training_state.add_transition(update_record)

        path.append(env.agent_pos)

        if done:
            if info.get("reached_goal"):
                reached_goal = True
            break

        state = next_state

    training_state.current_path = path

    return EpisodeMetrics(
        episode      = training_state.current_episode,
        total_reward = total_reward,
        steps        = env.step_count,
        reached_goal = reached_goal,
        epsilon      = agent.epsilon,
    )


def train_batch(
    env: MazeEnvironment,
    agent: QLearningAgent,
    training_state: TrainingState,
    batch_size: int = 10,
) -> bool:
    """
    Run up to `batch_size` episodes, updating training_state.

    Returns True if training finished (all episodes done or stop requested).
    Called repeatedly by the Streamlit UI with st.rerun() between batches so
    the interface stays responsive.
    """
    if training_state.stop_requested:
        training_state.is_training = False
        return True

    episodes_left = training_state.total_episodes - training_state.current_episode
    n = min(batch_size, episodes_left)

    for _ in range(n):
        if training_state.stop_requested or training_state.is_paused:
            break

        training_state.current_episode += 1
        metrics = run_episode(env, agent, training_state, greedy=False)
        metrics.episode = training_state.current_episode
        training_state.metrics.append(metrics)
        agent.decay_epsilon()

    finished = (
        training_state.current_episode >= training_state.total_episodes
        or training_state.stop_requested
    )

    if finished:
        training_state.is_training       = False
        training_state.training_complete = True

    return finished
