"""
evaluation.py — MazeMind RL
Fair comparison between a random baseline and the trained Q-Learning policy.

Both policies run on the same maze, start/goal, rewards, and step limit.
The Q-table is never modified during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent import QLearningAgent
from environment import MazeEnvironment


@dataclass
class EvaluationResult:
    """Aggregated statistics for one evaluated policy."""
    policy_name:        str
    n_episodes:         int
    n_successful:       int
    n_failed:           int
    success_rate:       float          # 0–100 %
    avg_steps_success:  Optional[float]
    avg_reward:         float
    avg_episode_length: float
    episode_rewards:    List[float]
    episode_steps:      List[int]
    episode_goals:      List[bool]
    final_path:         List[Tuple[int, int]]  # path from last episode

    def to_dict(self) -> Dict:
        return {
            "policy":               self.policy_name,
            "episodes":             self.n_episodes,
            "successful":           self.n_successful,
            "failed":               self.n_failed,
            "success_rate_pct":     round(self.success_rate, 2),
            "avg_steps_success":    round(self.avg_steps_success, 2) if self.avg_steps_success else None,
            "avg_reward":           round(self.avg_reward, 4),
            "avg_episode_length":   round(self.avg_episode_length, 2),
        }


def _run_random_episode(
    env: MazeEnvironment,
    rng: np.random.Generator,
) -> Tuple[float, int, bool, List[Tuple[int, int]]]:
    """One episode with a uniformly random policy."""
    env.reset()
    total_reward = 0.0
    reached_goal = False
    path = [env.agent_pos]

    for _ in range(env.max_steps):
        action = int(rng.integers(env.n_actions if hasattr(env, "n_actions") else 4))
        _, reward, done, info = env.step(action)
        total_reward += reward
        path.append(env.agent_pos)
        if done:
            reached_goal = info.get("reached_goal", False)
            break

    return total_reward, env.step_count, reached_goal, path


def _run_greedy_episode(
    env: MazeEnvironment,
    agent: QLearningAgent,
) -> Tuple[float, int, bool, List[Tuple[int, int]]]:
    """One episode with the greedy Q-policy (no exploration, no updates)."""
    state        = env.reset()
    total_reward = 0.0
    reached_goal = False
    path         = [env.agent_pos]

    for _ in range(env.max_steps):
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        path.append(env.agent_pos)
        state = next_state
        if done:
            reached_goal = info.get("reached_goal", False)
            break

    return total_reward, env.step_count, reached_goal, path


def evaluate_random(
    env: MazeEnvironment,
    n_episodes: int = 100,
    seed: Optional[int] = None,
) -> EvaluationResult:
    """Evaluate a uniformly random policy."""
    rng = np.random.default_rng(seed)
    rewards, steps, goals = [], [], []
    final_path: List[Tuple[int, int]] = []

    for _ in range(n_episodes):
        r, s, g, path = _run_random_episode(env, rng)
        rewards.append(r)
        steps.append(s)
        goals.append(g)
        final_path = path

    n_ok = sum(goals)
    good_steps = [steps[i] for i in range(n_episodes) if goals[i]]

    return EvaluationResult(
        policy_name        = "Random Baseline",
        n_episodes         = n_episodes,
        n_successful       = n_ok,
        n_failed           = n_episodes - n_ok,
        success_rate       = n_ok / n_episodes * 100.0,
        avg_steps_success  = float(np.mean(good_steps)) if good_steps else None,
        avg_reward         = float(np.mean(rewards)),
        avg_episode_length = float(np.mean(steps)),
        episode_rewards    = rewards,
        episode_steps      = steps,
        episode_goals      = goals,
        final_path         = final_path,
    )


def evaluate_agent(
    env: MazeEnvironment,
    agent: QLearningAgent,
    n_episodes: int = 100,
) -> EvaluationResult:
    """Evaluate the trained greedy policy (Q-table not modified)."""
    rewards, steps, goals = [], [], []
    final_path: List[Tuple[int, int]] = []

    for _ in range(n_episodes):
        r, s, g, path = _run_greedy_episode(env, agent)
        rewards.append(r)
        steps.append(s)
        goals.append(g)
        final_path = path

    n_ok = sum(goals)
    good_steps = [steps[i] for i in range(n_episodes) if goals[i]]

    return EvaluationResult(
        policy_name        = "Trained Q-Agent",
        n_episodes         = n_episodes,
        n_successful       = n_ok,
        n_failed           = n_episodes - n_ok,
        success_rate       = n_ok / n_episodes * 100.0,
        avg_steps_success  = float(np.mean(good_steps)) if good_steps else None,
        avg_reward         = float(np.mean(rewards)),
        avg_episode_length = float(np.mean(steps)),
        episode_rewards    = rewards,
        episode_steps      = steps,
        episode_goals      = goals,
        final_path         = final_path,
    )


def compare_results(
    random_res: EvaluationResult,
    agent_res:  EvaluationResult,
) -> Dict:
    """Compute relative improvement metrics between two evaluation results."""
    def pct_change(new, old):
        if old is None or old == 0:
            return None
        return (new - old) / abs(old) * 100.0

    return {
        "success_rate_improvement_pp": round(
            agent_res.success_rate - random_res.success_rate, 2
        ),
        "reward_improvement_pct": pct_change(
            agent_res.avg_reward, random_res.avg_reward
        ),
        "steps_improvement_pct": (
            pct_change(random_res.avg_steps_success, agent_res.avg_steps_success)
            if agent_res.avg_steps_success and random_res.avg_steps_success
            else None
        ),
    }
