"""
test_mazemind.py — MazeMind RL
Self-contained test suite. Run with: python3 test_mazemind.py
Tests environment, agent, training, and evaluation logic without Streamlit.
"""

import sys
import traceback
import numpy as np

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"{status}  {name}" + (f" — {detail}" if detail else ""))


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─── Imports ─────────────────────────────────────────────────────────────────
section("Import checks")
try:
    from environment import MazeEnvironment, UP, DOWN, LEFT, RIGHT, ACTION_DELTAS
    check("environment.py imports", True)
except Exception as e:
    check("environment.py imports", False, str(e))
    sys.exit(1)

try:
    from agent import QLearningAgent
    check("agent.py imports", True)
except Exception as e:
    check("agent.py imports", False, str(e))
    sys.exit(1)

try:
    from training import TrainingState, run_episode, train_batch
    check("training.py imports", True)
except Exception as e:
    check("training.py imports", False, str(e))

try:
    from evaluation import evaluate_random, evaluate_agent, compare_results
    check("evaluation.py imports", True)
except Exception as e:
    check("evaluation.py imports", False, str(e))


# ─── Environment tests ────────────────────────────────────────────────────────
section("Environment — core")

env = MazeEnvironment(rows=6, cols=6, seed=42)

check("n_states == 36", env.n_states == 36)
check("state_to_index(0,0) == 0", env.state_to_index((0, 0)) == 0)
check("state_to_index(5,5) == 35", env.state_to_index((5, 5)) == 35)
check("index_to_state(0) == (0,0)", env.index_to_state(0) == (0, 0))
check("index_to_state(35) == (5,5)", env.index_to_state(35) == (5, 5))
check("round-trip state/index", env.index_to_state(env.state_to_index((3, 4))) == (3, 4))

section("Environment — reset & step")

s0 = env.reset()
check("reset returns start index", s0 == env.state_to_index(env.start))
check("agent at start after reset", env.agent_pos == env.start)
check("done==False after reset", env.done == False)
check("step_count==0 after reset", env.step_count == 0)

section("Environment — wall collision")

# Move into the top wall from (0,0).
env.reset()
state_before = env.agent_pos
s1, r1, done1, info1 = env.step(UP)   # move Up from (0,0) — hits boundary
check("wall hit: stays in place", env.agent_pos == state_before)
check("wall hit: reward == reward_wall", r1 == env.reward_wall)
check("wall hit: not done", not done1)

section("Environment — obstacle collision")

# Check that obstacle cells are respected.
obs = list(env.obstacles)
if obs:
    or_, oc_ = obs[0]
    # Find an adjacent free cell.
    for a, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = or_ + dr, oc_ + dc
        adj = (nr, nc)
        if (
            0 <= nr < env.rows and 0 <= nc < env.cols
            and adj not in env.obstacles
            and adj != env.start and adj != env.goal
        ):
            env.reset()
            # Manually move agent to adj cell (by faking position).
            env.agent_pos = adj
            env.visited = [adj]
            # Try to move towards obstacle.
            rev = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
            move = {v: k for k, v in {UP: (-1,0), DOWN:(1,0), LEFT:(0,-1), RIGHT:(0,1)}.items()}
            direction = (or_ - nr, oc_ - nc)
            if direction in move:
                act = move[direction]
                _, r_obs, _, _ = env.step(act)
                check("obstacle collision: reward_wall", r_obs == env.reward_wall,
                      f"r={r_obs}")
            break

section("Environment — goal transition")

# Walk the agent to just before the goal and take the final step.
env2 = MazeEnvironment(rows=3, cols=3,
                        start=(0, 0), goal=(2, 2),
                        obstacles=[], max_steps=20, seed=0)
env2.reset()
env2.agent_pos = (2, 1)   # one step left of goal
env2.visited   = [(2, 1)]
ns_g, rg, done_g, info_g = env2.step(RIGHT)   # reach (2,2)
check("goal step: reward == reward_goal", rg == env2.reward_goal)
check("goal step: done == True", done_g)
check("goal step: info reached_goal", info_g.get("reached_goal", False))
check("goal step: next_state == goal index", ns_g == env2.state_to_index(env2.goal))

section("Environment — episode step limit")

env3 = MazeEnvironment(rows=3, cols=3, start=(0,0), goal=(2,2),
                        obstacles=[], max_steps=5, seed=1)
env3.reset()
done_early = False
for _ in range(6):
    _, _, d, _ = env3.step(UP)   # keep hitting wall — never reaches goal
    if d:
        done_early = True
        break
check("step limit terminates episode", done_early)

section("Environment — solvability")

check("default maze is solvable", env.is_solvable())
sp = env.shortest_path()
check("shortest_path returns a list", isinstance(sp, list) and len(sp) > 1)
check("shortest_path starts at start", sp[0] == env.start)
check("shortest_path ends at goal", sp[-1] == env.goal)


# ─── Agent tests ──────────────────────────────────────────────────────────────
section("Agent — initialisation")

agent = QLearningAgent(n_states=36, n_actions=4, seed=7)
check("Q-table shape (36,4)", agent.q_table.shape == (36, 4))
check("Q-table starts at zero", np.all(agent.q_table == 0))
check("epsilon starts at 1.0", agent.epsilon == 1.0)

section("Agent — Q-learning update (non-terminal)")

# Manual update.
agent2 = QLearningAgent(n_states=4, n_actions=4,
                         alpha=0.5, gamma=0.9, seed=0)
agent2.q_table[0, RIGHT] = 2.0
agent2.q_table[1]         = [4.0, 0, 0, 0]

rec = agent2.update(state=0, action=RIGHT, reward=5.0,
                     next_state=1, done=False)
# target = 5 + 0.9 * 4 = 8.6
# new_q  = 2 + 0.5 * (8.6 - 2) = 2 + 3.3 = 5.3
check("non-terminal target", abs(rec["target"] - 8.6) < 1e-9,
      f"target={rec['target']:.6f}")
check("non-terminal new_q", abs(rec["new_q"] - 5.3) < 1e-9,
      f"new_q={rec['new_q']:.6f}")
check("Q-table updated in place",
      abs(agent2.q_table[0, RIGHT] - 5.3) < 1e-9)

section("Agent — Q-learning update (terminal)")

agent3 = QLearningAgent(n_states=4, n_actions=4,
                         alpha=0.5, gamma=0.9, seed=0)
agent3.q_table[0, RIGHT] = 2.0
rec_t = agent3.update(state=0, action=RIGHT, reward=100.0,
                       next_state=0, done=True)
# target = 100 (no bootstrap)
# new_q  = 2 + 0.5 * (100 - 2) = 51
check("terminal target == reward", abs(rec_t["target"] - 100.0) < 1e-9,
      f"target={rec_t['target']:.6f}")
check("terminal new_q", abs(rec_t["new_q"] - 51.0) < 1e-9,
      f"new_q={rec_t['new_q']:.6f}")
check("terminal max_next_q == 0", rec_t["max_next_q"] == 0.0)

section("Agent — epsilon decay")

agent4 = QLearningAgent(n_states=4, epsilon=1.0, epsilon_min=0.05,
                         epsilon_decay=0.9)
agent4.decay_epsilon()
check("epsilon decays correctly", abs(agent4.epsilon - 0.9) < 1e-9,
      f"ε={agent4.epsilon:.6f}")
for _ in range(100):
    agent4.decay_epsilon()
check("epsilon floored at epsilon_min",
      agent4.epsilon >= agent4.epsilon_min - 1e-9)

section("Agent — greedy action (tie-breaking)")

agent5 = QLearningAgent(n_states=4, seed=0)
# All zeros — any action is valid.
a = agent5.select_action(0, greedy=True)
check("greedy action in valid range", 0 <= a < 4)

agent5.q_table[0, DOWN] = 1.0
a_best = agent5.select_action(0, greedy=True)
check("greedy picks highest Q-value", a_best == DOWN)

section("Agent — serialisation / deserialisation")

import json as _json
agent6 = QLearningAgent(n_states=9, n_actions=4, alpha=0.2, gamma=0.8, seed=3)
agent6.q_table[2, 1] = 3.14
j = agent6.to_json()
agent6b = QLearningAgent.from_json(j)
check("JSON round-trip preserves q_table",
      abs(agent6b.q_table[2, 1] - 3.14) < 1e-9)
check("JSON round-trip preserves alpha", agent6b.alpha == 0.2)
check("JSON round-trip preserves n_states", agent6b.n_states == 9)


# ─── Training tests ───────────────────────────────────────────────────────────
section("Training — run_episode")

env_t = MazeEnvironment(seed=0)
agent_t = QLearningAgent(n_states=env_t.n_states, seed=0)
ts = TrainingState()
ts.total_episodes = 5

q_before = agent_t.q_table.copy()
m = run_episode(env_t, agent_t, ts)
check("episode metrics returned", m is not None)
check("steps > 0", m.steps > 0)
check("total_reward is float", isinstance(m.total_reward, float))
check("Q-table changed after episode", not np.all(agent_t.q_table == q_before))

section("Training — train_batch (multiple episodes)")

env_b = MazeEnvironment(seed=1)
agent_b = QLearningAgent(n_states=env_b.n_states, seed=1)
ts2 = TrainingState()
ts2.total_episodes = 20
ts2.is_training = True
q0 = agent_b.q_table.copy()

done_flag = train_batch(env_b, agent_b, ts2, batch_size=10)
check("batch ran 10 episodes", ts2.current_episode == 10)
check("epsilon decayed after batch",
      agent_b.epsilon < 1.0, f"ε={agent_b.epsilon:.4f}")
check("Q-values changed in batch",
      not np.all(agent_b.q_table == q0))

# Run second batch.
ts2.is_training = True
ts2.is_paused   = False
done2 = train_batch(env_b, agent_b, ts2, batch_size=10)
check("second batch finishes training", done2)
check("all 20 episodes run", ts2.current_episode == 20)
check("training marked complete", ts2.training_complete)


# ─── Evaluation tests ─────────────────────────────────────────────────────────
section("Evaluation — random baseline")

env_ev = MazeEnvironment(seed=5)
res_r = evaluate_random(env_ev, n_episodes=50, seed=5)
check("random n_episodes == 50", res_r.n_episodes == 50)
check("random success_rate in [0,100]", 0 <= res_r.success_rate <= 100)
check("random n_successful + n_failed == n_episodes",
      res_r.n_successful + res_r.n_failed == 50)

section("Evaluation — trained agent (after some training)")

agent_ev = QLearningAgent(n_states=env_ev.n_states, seed=5)
ts_ev = TrainingState()
ts_ev.total_episodes = 100
ts_ev.is_training    = True
for _ in range(100):
    ts_ev.current_episode += 1
    m_ = run_episode(env_ev, agent_ev, ts_ev)
    ts_ev.metrics.append(m_)
    agent_ev.decay_epsilon()

res_a = evaluate_agent(env_ev, agent_ev, n_episodes=50)
check("agent n_episodes == 50", res_a.n_episodes == 50)
check("agent success_rate >= random (after training)",
      res_a.success_rate >= res_r.success_rate,
      f"agent={res_a.success_rate:.1f}% random={res_r.success_rate:.1f}%")

section("Evaluation — compare_results")

cmp = compare_results(res_r, res_a)
check("compare returns success_rate_improvement_pp",
      "success_rate_improvement_pp" in cmp)
check("improvement is numeric", isinstance(
      cmp["success_rate_improvement_pp"], (int, float)))


# ─── Visualization import test ────────────────────────────────────────────────
section("Visualization — imports and basic calls")

try:
    from visualization import (
        qtable_dataframe, render_maze,
        chart_episode_rewards, chart_success_rate,
        chart_steps_per_episode, chart_epsilon,
        qtable_heatmap, max_qtable_heatmap,
        comparison_bar_chart,
    )
    df_q = qtable_dataframe(agent_t.q_table, 6, 6)
    check("qtable_dataframe shape", df_q.shape == (36, 4))
    check("qtable_dataframe columns",
          list(df_q.columns) == ["Up ↑", "Down ↓", "Left ←", "Right →"])

    # render_maze returns a Figure.
    fig = render_maze(env_t)
    check("render_maze returns Figure",
          hasattr(fig, "data") and hasattr(fig, "layout"))

    # Heatmaps return Figures without keyword argument conflicts.
    fig_heat = qtable_heatmap(agent_t.q_table, 6, 6, 0)
    check("qtable_heatmap returns Figure", hasattr(fig_heat, "data"))

    fig_max = max_qtable_heatmap(agent_t.q_table, 6, 6)
    check("max_qtable_heatmap returns Figure", hasattr(fig_max, "data"))

    # Comparison bar chart returns Figure.
    fig_bar = comparison_bar_chart(res_r, res_a)
    check("comparison_bar_chart returns Figure", hasattr(fig_bar, "data"))

    # Charts don't crash with empty data.
    ts_empty = TrainingState()
    fig2 = chart_episode_rewards(ts_empty)
    check("chart_episode_rewards handles empty data",
          hasattr(fig2, "data"))

    fig3 = chart_steps_per_episode(ts_empty)
    check("chart_steps_per_episode handles empty data", hasattr(fig3, "data"))

    fig4 = chart_epsilon(ts_empty)
    check("chart_epsilon handles empty data", hasattr(fig4, "data"))

    check("visualization imports OK", True)
except Exception as e:
    check("visualization imports OK", False, str(e))
    traceback.print_exc()


# ─── Summary ─────────────────────────────────────────────────────────────────
section("Test Summary")

n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
print(f"\nTotal: {len(results)} tests | {n_pass} passed | {n_fail} failed\n")

if n_fail:
    print("Failed tests:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  {FAIL}  {name}  {detail}")
    sys.exit(1)
else:
    print("All tests passed! ✅")
