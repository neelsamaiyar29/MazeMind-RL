"""
app.py — MazeMind RL | Intelligent Navigation Lab
Main Streamlit application.

Architecture
────────────
All mutable state lives in st.session_state so it survives Streamlit reruns.
Training runs in *batches* (not a blocking while-loop) so the UI stays
responsive and pause/stop controls work correctly.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from agent import QLearningAgent
from environment import MazeEnvironment
from evaluation import compare_results, evaluate_agent, evaluate_random
from training import TrainingState, train_batch
from visualization import (
    chart_episode_rewards,
    chart_epsilon,
    chart_steps_per_episode,
    chart_success_rate,
    comparison_bar_chart,
    max_qtable_heatmap,
    qtable_dataframe,
    qtable_heatmap,
    render_maze,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MazeMind RL | Intelligent Navigation Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark-theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  body, .stApp { background-color: #0E1117; color: #FAFAFA; }

  /* ── KPI Cards ── */
  .kpi-card {
      background: linear-gradient(135deg, #1A1F2E 0%, #141922 100%);
      border: 1px solid #2A3040;
      border-radius: 10px;
      padding: 16px 18px;
      text-align: center;
  }
  .kpi-label { font-size: 0.72rem; color: #888; text-transform: uppercase;
               letter-spacing: 0.08em; margin-bottom: 4px; }
  .kpi-value { font-size: 1.6rem; font-weight: 700; color: #FFD700; }
  .kpi-sub   { font-size: 0.78rem; color: #555; margin-top: 3px; }

  /* ── Section headers ── */
  .section-header {
      font-size: 1.1rem; font-weight: 600; color: #00D4FF;
      border-bottom: 1px solid #2A3040; padding-bottom: 6px; margin-bottom: 12px;
  }

  /* ── Status badge ── */
  .status-badge {
      display: inline-block; border-radius: 12px; padding: 3px 12px;
      font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em;
  }
  .status-training { background: #1A3A1A; color: #00FF88; border: 1px solid #00FF88; }
  .status-paused   { background: #3A3A1A; color: #FFD700; border: 1px solid #FFD700; }
  .status-idle     { background: #1A2A3A; color: #00D4FF; border: 1px solid #00D4FF; }
  .status-done     { background: #2A1A3A; color: #CC88FF; border: 1px solid #CC88FF; }

  /* ── Legend ── */
  .legend-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; }
  .legend-item { display: flex; align-items: center; gap: 6px;
                 font-size: 0.82rem; color: #CCC; }
  .legend-dot { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }

  /* ── Update equation ── */
  .eq-box {
      background: #141922; border: 1px solid #2A3040; border-radius: 8px;
      padding: 14px 18px; font-family: monospace; font-size: 0.9rem; color: #FFD700;
  }
  .eq-terminal { color: #FF6B35; font-style: italic; font-size: 0.82rem; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] { background-color: #0A0D14; }

  /* ── Buttons ── */
  .stButton > button {
      border-radius: 6px; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_env(cfg: dict) -> MazeEnvironment:
    return MazeEnvironment(
        rows        = cfg["rows"],
        cols        = cfg["cols"],
        start       = tuple(cfg["start"]),
        goal        = tuple(cfg["goal"]),
        obstacles   = cfg["obstacles"],
        reward_goal = cfg["reward_goal"],
        reward_step = cfg["reward_step"],
        reward_wall = cfg["reward_wall"],
        max_steps   = cfg["max_steps"],
        seed        = cfg.get("seed"),
    )


def _init_agent(cfg: dict, n_states: int) -> QLearningAgent:
    return QLearningAgent(
        n_states      = n_states,
        n_actions     = 4,
        alpha         = cfg["alpha"],
        gamma         = cfg["gamma"],
        epsilon       = cfg["epsilon"],
        epsilon_min   = cfg["epsilon_min"],
        epsilon_decay = cfg["epsilon_decay"],
        seed          = cfg.get("seed"),
    )


DEFAULT_OBSTACLES = [
    [0, 3], [0, 4],
    [1, 1], [1, 2],
    [2, 4], [2, 5],
    [3, 1], [3, 2], [3, 3],
    [4, 3], [4, 4],
]

DEFAULT_CFG = {
    "rows": 6, "cols": 6,
    "start": [0, 0], "goal": [5, 5],
    "obstacles": DEFAULT_OBSTACLES,
    "reward_goal": 100.0, "reward_step": -1.0, "reward_wall": -5.0,
    "max_steps": 100, "seed": None,
    "alpha": 0.1, "gamma": 0.95,
    "epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": 0.995,
    "n_episodes": 1500,
    "eval_episodes": 100,
    "ma_window": 50,
    "batch_size": 25,
}


def init_session():
    ss = st.session_state
    if "cfg" not in ss:
        ss.cfg = dict(DEFAULT_CFG)
    if "env" not in ss:
        ss.env = _init_env(ss.cfg)
    if "agent" not in ss:
        ss.agent = _init_agent(ss.cfg, ss.env.n_states)
    if "ts" not in ss:
        ss.ts = TrainingState()
        ss.ts.total_episodes = ss.cfg["n_episodes"]
    if "eval_random" not in ss:
        ss.eval_random = None
    if "eval_agent" not in ss:
        ss.eval_agent = None
    if "demo_pos" not in ss:
        ss.demo_pos = None          # agent position for step-mode
    if "demo_state" not in ss:
        ss.demo_state = None
    if "demo_path" not in ss:
        ss.demo_path = []
    if "demo_reward" not in ss:
        ss.demo_reward = 0.0
    if "demo_done" not in ss:
        ss.demo_done = False
    if "demo_step" not in ss:
        ss.demo_step = 0
    if "last_action_info" not in ss:
        ss.last_action_info = None
    if "show_policy" not in ss:
        ss.show_policy = False
    if "anim_running" not in ss:
        ss.anim_running = False


init_session()
ss = st.session_state


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>"""


def fmt(v, decimals: int = 2, fallback: str = "—") -> str:
    if v is None:
        return fallback
    return f"{v:.{decimals}f}"


def status_badge() -> str:
    ts = ss.ts
    if ts.is_training and not ts.is_paused:
        return '<span class="status-badge status-training">⚡ Training</span>'
    if ts.is_paused:
        return '<span class="status-badge status-paused">⏸ Paused</span>'
    if ts.training_complete:
        return '<span class="status-badge status-done">✓ Complete</span>'
    return '<span class="status-badge status-idle">● Idle</span>'


def reset_demo():
    """Reset step-through mode state."""
    ss.env.reset()
    ss.demo_pos    = ss.env.agent_pos
    ss.demo_state  = ss.env.state_to_index(ss.env.agent_pos)
    ss.demo_path   = [ss.env.agent_pos]
    ss.demo_reward = 0.0
    ss.demo_done   = False
    ss.demo_step   = 0
    ss.last_action_info = None


def rebuild_env_agent():
    """Rebuild env and agent from current cfg (called after config changes)."""
    try:
        ss.env   = _init_env(ss.cfg)
        ss.agent = _init_agent(ss.cfg, ss.env.n_states)
        ss.ts    = TrainingState()
        ss.ts.total_episodes = ss.cfg["n_episodes"]
        ss.eval_random = None
        ss.eval_agent  = None
        reset_demo()
        return True, ""
    except ValueError as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Experiment Configuration")
    st.markdown("---")

    with st.expander("🗺️ Maze Settings", expanded=False):
        rows = st.slider("Grid Rows", 4, 10, ss.cfg["rows"])
        cols = st.slider("Grid Cols", 4, 10, ss.cfg["cols"])
        st.caption("Start / Goal (row, col)")
        sc1, sc2 = st.columns(2)
        with sc1:
            sr = st.number_input("Start Row", 0, rows-1, ss.cfg["start"][0])
            sc = st.number_input("Start Col", 0, cols-1, ss.cfg["start"][1])
        with sc2:
            gr = st.number_input("Goal Row", 0, rows-1, ss.cfg["goal"][0])
            gc = st.number_input("Goal Col", 0, cols-1, ss.cfg["goal"][1])
        st.caption("Obstacles (comma-separated row,col pairs per line)")
        obs_text_default = "\n".join(
            f"{r},{c}" for r, c in ss.cfg["obstacles"]
        )
        obs_text = st.text_area("Obstacles", obs_text_default, height=120)

        if st.button("Apply Maze Changes", type="primary"):
            new_obstacles = []
            valid_obs     = True
            for line in obs_text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split(",")
                    new_obstacles.append([int(parts[0]), int(parts[1])])
                except Exception:
                    st.error(f"Invalid obstacle format: '{line}'")
                    valid_obs = False
                    break
            if valid_obs:
                ss.cfg.update({
                    "rows": rows, "cols": cols,
                    "start": [sr, sc], "goal": [gr, gc],
                    "obstacles": new_obstacles,
                })
                ok, err = rebuild_env_agent()
                if ok:
                    if not ss.env.is_solvable():
                        st.warning("⚠️ Maze may be unsolvable — check obstacles!")
                    else:
                        st.success("✅ Maze updated!")
                else:
                    st.error(f"Configuration error: {err}")

    with st.expander("💰 Reward Settings", expanded=False):
        rg = st.number_input("Goal Reward",  value=ss.cfg["reward_goal"], step=10.0)
        rs = st.number_input("Step Penalty", value=ss.cfg["reward_step"], step=0.5)
        rw = st.number_input("Wall Penalty", value=ss.cfg["reward_wall"], step=1.0)
        ms = st.number_input("Max Steps/Episode", 20, 500, ss.cfg["max_steps"], step=10)
        if st.button("Apply Reward Changes"):
            ss.cfg.update({"reward_goal": rg, "reward_step": rs,
                           "reward_wall": rw, "max_steps": ms})
            ok, err = rebuild_env_agent()
            if ok:
                st.success("✅ Rewards updated — training reset.")
            else:
                st.error(err)

    with st.expander("🧬 Q-Learning Hyperparameters", expanded=True):
        alpha  = st.slider("Learning Rate (α)",   0.01, 1.0,  ss.cfg["alpha"],  0.01)
        gamma  = st.slider("Discount Factor (γ)", 0.1,  0.99, ss.cfg["gamma"],  0.01)
        eps    = st.slider("Initial Epsilon",     0.1,  1.0,  ss.cfg["epsilon"], 0.05)
        eps_mn = st.slider("Minimum Epsilon",     0.01, 0.5,  ss.cfg["epsilon_min"], 0.01)
        eps_dc = st.slider("Epsilon Decay",       0.9,  0.999, ss.cfg["epsilon_decay"], 0.001, format="%.3f")
        n_eps  = st.number_input("Training Episodes", 100, 10000, ss.cfg["n_episodes"], 100)
        batch  = st.number_input("Batch Size (eps/refresh)", 5, 100, ss.cfg["batch_size"], 5)
        seed   = st.number_input("Random Seed (-1 = off)", -1, 99999, -1 if ss.cfg["seed"] is None else ss.cfg["seed"])
        if st.button("Apply Hyperparameters"):
            ss.cfg.update({
                "alpha": alpha, "gamma": gamma,
                "epsilon": eps, "epsilon_min": eps_mn, "epsilon_decay": eps_dc,
                "n_episodes": n_eps, "batch_size": int(batch),
                "seed": None if seed == -1 else int(seed),
            })
            # Update agent hyperparams without wiping Q-table.
            ss.agent.alpha         = alpha
            ss.agent.gamma         = gamma
            ss.agent.epsilon_min   = eps_mn
            ss.agent.epsilon_decay = eps_dc
            ss.ts.total_episodes   = n_eps
            st.success("✅ Hyperparameters updated.")

    with st.expander("📊 Evaluation Settings", expanded=False):
        eval_eps = st.number_input("Evaluation Episodes", 10, 500,
                                   ss.cfg["eval_episodes"], 10)
        ma_win   = st.number_input("Moving-Avg Window", 10, 200,
                                   ss.cfg["ma_window"], 10)
        if st.button("Save Eval Settings"):
            ss.cfg["eval_episodes"] = int(eval_eps)
            ss.cfg["ma_window"]     = int(ma_win)
            st.success("✅ Saved.")

    st.markdown("---")
    st.markdown("**Legend**")
    st.markdown("""
    <div class="legend-row">
      <div class="legend-item"><div class="legend-dot" style="background:#00D4FF"></div>Start (S)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#00FF88"></div>Goal (G)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#8B1A1A"></div>Obstacle</div>
      <div class="legend-item"><div class="legend-dot" style="background:#FFD700"></div>Agent ★</div>
      <div class="legend-item"><div class="legend-dot" style="background:#2D4A6B"></div>Visited</div>
      <div class="legend-item"><div class="legend-dot" style="background:#FF6B35"></div>Eval Path</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center; color:#00D4FF; letter-spacing:0.03em;'>"
    "🧠 MazeMind RL</h1>"
    "<p style='text-align:center; color:#888; font-size:0.95rem; margin-top:-8px;'>"
    "Intelligent Navigation Lab — Q-Learning from scratch</p>",
    unsafe_allow_html=True,
)
st.markdown("""
> **How it works:** An agent explores a grid maze, collecting rewards and penalties.
> After each step it applies the Q-learning update rule to improve its policy.
> Over many episodes the agent learns the shortest safe path to the goal.
> All Q-values displayed are computed from real transitions — nothing is fabricated.
""")

col_status, col_ep, col_total = st.columns([2, 3, 1])
with col_status:
    st.markdown(f"**Status:** {status_badge()}", unsafe_allow_html=True)
with col_ep:
    denominator = max(ss.ts.total_episodes, 1)
    prog = min(1.0, max(0.0, float(ss.ts.current_episode / denominator)))
    disp_total = max(ss.ts.total_episodes, ss.ts.current_episode)
    st.progress(prog, text=f"Episode {ss.ts.current_episode} / {disp_total}")



# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📈 Key Performance Indicators</div>',
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k5, k6, k7, k8 = st.columns(4)

ts = ss.ts
with k1:
    st.markdown(kpi("Current Episode", str(ts.current_episode),
                    f"of {ts.total_episodes}"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi("Latest Reward", fmt(ts.latest_reward, 1)),
                unsafe_allow_html=True)
with k3:
    st.markdown(kpi("Moving Avg Reward", fmt(ts.moving_avg_reward, 1),
                    f"window={ss.cfg['ma_window']}"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi("Success Rate", f"{ts.success_rate:.1f}%",
                    f"{ts.n_successful} / {len(ts.metrics)} eps"),
                unsafe_allow_html=True)
with k5:
    st.markdown(kpi("Avg Steps (Success)", fmt(ts.avg_steps_success, 1)),
                unsafe_allow_html=True)
with k6:
    st.markdown(kpi("Current ε", fmt(ss.agent.epsilon, 4)),
                unsafe_allow_html=True)
with k7:
    st.markdown(kpi("Successful Episodes", str(ts.n_successful)),
                unsafe_allow_html=True)
with k8:
    q_max = float(np.max(ss.agent.q_table))
    st.markdown(kpi("Max Q-Value", fmt(q_max, 3)), unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🗺️ Maze Simulator",
    "📈 Training Progress",
    "🔢 Q-Table Explorer",
    "🔬 Learning Inspector",
    "🏆 Evaluation",
    "💾 Export & History",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Maze Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">🗺️ Live Maze Simulator</div>',
                unsafe_allow_html=True)

    col_maze, col_ctrl = st.columns([3, 2])

    with col_ctrl:
        st.markdown("#### Training Controls")

        c1, c2 = st.columns(2)
        with c1:
            train_btn = st.button("▶ Train Agent",   use_container_width=True, type="primary")
            stop_btn  = st.button("⏹ Stop",          use_container_width=True)
            reset_env = st.button("🔄 Reset Env",    use_container_width=True)
        with c2:
            pause_btn  = st.button("⏸ Pause",        use_container_width=True)
            resume_btn = st.button("▶ Resume",        use_container_width=True)
            reset_lrn  = st.button("🗑 Reset Learning", use_container_width=True)

        st.markdown("#### Step / Animate Controls")
        c3, c4 = st.columns(2)
        with c3:
            step_btn = st.button("👣 Step Once",    use_container_width=True)
            test_btn = st.button("🧪 Test Agent",   use_container_width=True)
        with c4:
            anim_btn = st.button("🎬 Play Anim",    use_container_width=True)
            ss.show_policy = st.checkbox("Show Policy Arrows", value=ss.show_policy)

        # ── Button handlers ────────────────────────────────────────────────
        if train_btn:
            if not ss.env.is_solvable():
                st.error("Maze is unsolvable — adjust obstacles.")
            else:
                ss.ts.is_training       = True
                ss.ts.is_paused         = False
                ss.ts.stop_requested    = False
                ss.ts.training_complete = False
                ss.ts.total_episodes    = ss.cfg["n_episodes"]

        if pause_btn and ss.ts.is_training:
            ss.ts.is_paused = True

        if resume_btn and ss.ts.is_paused:
            ss.ts.is_paused = False

        if stop_btn:
            ss.ts.stop_requested = True
            ss.ts.is_training    = False

        if reset_env:
            ss.env.reset()
            reset_demo()

        if reset_lrn:
            ss.agent.reset_qtable()
            ss.agent.epsilon = ss.cfg["epsilon"]
            ss.ts.reset()
            ss.ts.total_episodes = ss.cfg["n_episodes"]
            ss.eval_random = None
            ss.eval_agent  = None
            reset_demo()
            st.success("Learning reset — Q-table cleared.")

        if test_btn:
            reset_demo()
            # Run a full greedy episode and store path.
            state = ss.env.reset()
            ss.demo_pos    = ss.env.agent_pos
            ss.demo_state  = state
            ss.demo_path   = [ss.env.agent_pos]
            ss.demo_reward = 0.0
            ss.demo_done   = False
            for _ in range(ss.env.max_steps):
                a = ss.agent.select_action(state, greedy=True)
                ns, r, done, info = ss.env.step(a)
                ss.demo_reward += r
                ss.demo_path.append(ss.env.agent_pos)
                state = ns
                if done:
                    ss.demo_done = True
                    break
            ss.demo_pos = ss.env.agent_pos
            ss.ts.last_eval_path = list(ss.demo_path)

        if step_btn:
            if ss.demo_done or ss.demo_state is None:
                reset_demo()
            else:
                a = ss.agent.select_action(ss.demo_state, greedy=True)
                ns, r, done, info = ss.env.step(a)
                rec = ss.agent.update(ss.demo_state, a, r, ns, done)
                rec["pos"]      = info.get("prev_pos", ss.env.agent_pos)
                rec["next_pos"] = info.get("next_pos", ss.env.agent_pos)
                ss.ts.add_transition(rec)
                ss.demo_reward += r
                ss.demo_step   += 1
                ss.demo_path.append(ss.env.agent_pos)
                ss.demo_state  = ns
                ss.demo_pos    = ss.env.agent_pos
                ss.demo_done   = done
                ss.last_action_info = {
                    "action":  info["action"],
                    "reward":  r,
                    "cum_rew": ss.demo_reward,
                    "step":    ss.demo_step,
                    "done":    done,
                    "prev":    info.get("prev_pos"),
                    "next":    ss.env.agent_pos,
                }

        if anim_btn:
            reset_demo()
            state = ss.env.reset()
            ss.demo_path = [ss.env.agent_pos]
            # Animate greedy path — step through in a loop with sleep.
            anim_placeholder = st.empty()
            for _ in range(ss.env.max_steps):
                a = ss.agent.select_action(state, greedy=True)
                ns, r, done, info = ss.env.step(a)
                ss.demo_reward += r
                ss.demo_path.append(ss.env.agent_pos)
                state = ns
                fig = render_maze(
                    ss.env,
                    agent_pos=ss.env.agent_pos,
                    path=ss.demo_path,
                    eval_path=ss.ts.last_eval_path,
                    show_policy=ss.show_policy,
                    policy=ss.agent.get_greedy_policy() if ss.show_policy else None,
                )
                anim_placeholder.plotly_chart(fig, use_container_width=True, key=f"anim_{_}")
                time.sleep(0.25)
                if done:
                    break
            ss.demo_done = True
            ss.demo_pos  = ss.env.agent_pos

        # ── Step-info panel ────────────────────────────────────────────────
        if ss.last_action_info:
            ai = ss.last_action_info
            from environment import ACTION_NAMES
            st.markdown("#### Last Step Details")
            st.markdown(f"""
| Field | Value |
|-------|-------|
| Previous State | {ai['prev']} |
| Action | {ACTION_NAMES.get(ai['action'], ai['action'])} |
| Next State | {ai['next']} |
| Immediate Reward | **{ai['reward']:.2f}** |
| Cumulative Reward | {ai['cum_rew']:.2f} |
| Step # | {ai['step']} |
| Episode Done | {'✅ Yes' if ai['done'] else '⬜ No'} |
""")

        # ── Maze solvability ───────────────────────────────────────────────
        solvable = ss.env.is_solvable()
        if not solvable:
            st.error("⚠️ Current maze configuration is unsolvable!")
        else:
            sp = ss.env.shortest_path()
            if sp:
                st.info(f"Shortest path length: **{len(sp)-1} steps** (BFS optimal)")

    with col_maze:
        # Determine current agent position.
        agent_pos = ss.demo_pos if ss.demo_pos is not None else ss.env.agent_pos
        fig = render_maze(
            ss.env,
            agent_pos=agent_pos,
            path=ss.demo_path if ss.demo_path else None,
            eval_path=ss.ts.last_eval_path if ss.ts.last_eval_path else None,
            show_policy=ss.show_policy,
            policy=ss.agent.get_greedy_policy() if ss.show_policy else None,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Status for step-through mode.
        if ss.demo_done:
            if ss.env.agent_pos == ss.env.goal:
                st.success(f"🏆 Agent reached the goal! Total reward: {ss.demo_reward:.2f}")
            else:
                st.warning(f"Episode ended (step limit). Total reward: {ss.demo_reward:.2f}")

    # ── Training batch execution ───────────────────────────────────────────
    if ss.ts.is_training and not ss.ts.is_paused and not ss.ts.stop_requested:
        finished = train_batch(
            ss.env, ss.agent, ss.ts,
            batch_size=ss.cfg["batch_size"],
        )
        if not finished:
            time.sleep(0.05)
            st.rerun()
        else:
            ss.ts.is_training = False
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Training Progress
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">📈 Training Progress Dashboard</div>',
                unsafe_allow_html=True)

    win = ss.cfg["ma_window"]
    t2c1, t2c2 = st.columns(2)
    with t2c1:
        st.plotly_chart(chart_episode_rewards(ss.ts, win),
                        use_container_width=True)
        st.plotly_chart(chart_success_rate(ss.ts, win),
                        use_container_width=True)
    with t2c2:
        st.plotly_chart(chart_steps_per_episode(ss.ts),
                        use_container_width=True)
        st.plotly_chart(chart_epsilon(ss.ts),
                        use_container_width=True)

    if ss.ts.metrics:
        st.markdown("#### Training Summary Table (last 20 episodes)")
        rows_data = []
        for m in ss.ts.metrics[-20:]:
            rows_data.append({
                "Episode":       m.episode,
                "Total Reward":  round(m.total_reward, 2),
                "Steps":         m.steps,
                "Reached Goal":  "✅" if m.reached_goal else "❌",
                "Epsilon":       round(m.epsilon, 4),
            })
        st.dataframe(pd.DataFrame(rows_data), use_container_width=True,
                     hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Q-Table Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">🔢 Live Q-Table Explorer</div>',
                unsafe_allow_html=True)

    env  = ss.env
    rows_n, cols_n = env.rows, env.cols

    # ── Cell inspector ─────────────────────────────────────────────────────
    qc1, qc2 = st.columns([2, 3])
    with qc1:
        st.markdown("#### Inspect State")
        sel_r = st.number_input("Row", 0, rows_n - 1, 0, key="qr")
        sel_c = st.number_input("Col", 0, cols_n - 1, 0, key="qc")
        idx   = sel_r * cols_n + sel_c
        q_row = ss.agent.q_table[idx]
        best_a = int(np.argmax(q_row))
        st.markdown(f"""
**State ({sel_r},{sel_c}) → index {idx}**

| Action | Q-Value |
|--------|---------|
| ↑ Up | {q_row[0]:.4f} |
| ↓ Down | {q_row[1]:.4f} |
| ← Left | {q_row[2]:.4f} |
| → Right | {q_row[3]:.4f} |

**Best action:** `{['Up','Down','Left','Right'][best_a]}` ({q_row[best_a]:.4f})
""")

    with qc2:
        action_view = st.selectbox(
            "Heatmap — Select Action",
            ["Up ↑", "Down ↓", "Left ←", "Right →"],
            index=0,
        )
        action_idx_map = {"Up ↑": 0, "Down ↓": 1, "Left ←": 2, "Right →": 3}
        act_idx = action_idx_map[action_view]
        st.plotly_chart(
            qtable_heatmap(ss.agent.q_table, rows_n, cols_n, act_idx),
            use_container_width=True,
        )

    st.markdown("---")
    st.plotly_chart(
        max_qtable_heatmap(ss.agent.q_table, rows_n, cols_n),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### Full Q-Table (all states × actions)")
    st.caption("Highest Q-value per row is highlighted in training.")
    df_q = qtable_dataframe(ss.agent.q_table, rows_n, cols_n)

    # Highlight best action per row.
    def highlight_best(row):
        max_val = row.max()
        return ["background-color: #1A3A1A; color: #00FF88; font-weight:700"
                if v == max_val else "" for v in row]

    st.dataframe(
        df_q.style.apply(highlight_best, axis=1),
        use_container_width=True,
        height=min(600, rows_n * cols_n * 35 + 60),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Learning Process Inspector
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">🔬 Learning Process Inspector</div>',
                unsafe_allow_html=True)

    if not ss.ts.transition_history:
        st.info("No transitions recorded yet. Train the agent or use Step Once to record transitions.")
    else:
        history = ss.ts.transition_history
        n_hist  = len(history)

        sel_step = st.slider(
            "Select Transition",
            0, n_hist - 1, n_hist - 1,
            format="Transition %d",
        )
        t = history[sel_step]

        from environment import ACTION_NAMES as AN
        action_name = AN.get(t["action"], str(t["action"]))

        li_c1, li_c2 = st.columns(2)
        with li_c1:
            st.markdown("#### Transition Details")
            st.markdown(f"""
| Field | Value |
|-------|-------|
| State index | `{t['state']}` |
| Action | `{action_name}` |
| Immediate Reward | **{t['reward']:.4f}** |
| Next State index | `{t['next_state']}` |
| Episode Terminal | {'✅ Yes' if t['done'] else '⬜ No'} |
""")

        with li_c2:
            st.markdown("#### Q-Learning Update")
            if t["done"]:
                target_eq = f"r = {t['reward']:.4f}  (terminal — no future bootstrapping)"
                extra = '<div class="eq-terminal">Terminal state: max Q(s\',a\') = 0</div>'
            else:
                target_eq = f"{t['reward']:.4f} + {t['gamma']:.2f} × {t['max_next_q']:.4f} = {t['target']:.4f}"
                extra = ""

            st.markdown(f"""
<div class="eq-box">
<b>Update Equation:</b><br>
Q(s,a) ← Q(s,a) + α · [target − Q(s,a)]<br><br>

Old Q(s,a)    = <b>{t['old_q']:.6f}</b><br>
Reward (r)    = <b>{t['reward']:.4f}</b><br>
γ · max Q(s') = <b>{t['gamma']:.2f} × {t['max_next_q']:.4f} = {t['gamma']*t['max_next_q']:.4f}</b><br>
Target        = <b>{t['target']:.6f}</b><br>
TD Error      = <b>{t['td_error']:.6f}</b><br>
α             = <b>{t['alpha']}</b><br>
New Q(s,a)    = <b>{t['new_q']:.6f}</b><br>
{extra}
</div>
""", unsafe_allow_html=True)

        # Mini bar chart of Q-values before/after for this state.
        st.markdown("#### Q-values for this state (before → after)")
        q_after  = ss.agent.q_table[t["state"]].copy()
        q_before = q_after.copy()
        q_before[t["action"]] = t["old_q"]   # restore the old value

        action_labels = ["Up ↑", "Down ↓", "Left ←", "Right →"]
        compare_fig = go.Figure()
        compare_fig.add_trace(go.Bar(
            name="Before Update", x=action_labels, y=q_before.tolist(),
            marker_color="#4A90D9",
        ))
        compare_fig.add_trace(go.Bar(
            name="After Update", x=action_labels, y=q_after.tolist(),
            marker_color="#00FF88",
        ))
        _env_cols = ss.env.cols
        compare_fig.update_layout(
            barmode="group",
            paper_bgcolor="#0E1117", plot_bgcolor="#141922",
            font=dict(color="#CCC"),
            height=300,
            margin=dict(l=40, r=20, t=30, b=30),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=dict(text=f"State ({t['state']//_env_cols},{t['state']%_env_cols}) Q-values",
                       font=dict(color="#CCC", size=13)),
        )
        st.plotly_chart(compare_fig, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Performance Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">🏆 Performance Evaluation</div>',
                unsafe_allow_html=True)

    st.markdown("""
> **Methodology:** Both policies are evaluated over the same number of independent episodes,
> using identical maze, start/goal, reward rules, and step limit.
> The Q-table is **not** modified during evaluation.
> *Success rate* measures the fraction of episodes in which the agent reaches the goal —
> it is not a classification accuracy.
""")

    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        if st.button("🎲 Run Random Baseline", type="secondary", use_container_width=True):
            with st.spinner("Evaluating random policy…"):
                ss.eval_random = evaluate_random(
                    ss.env,
                    n_episodes=ss.cfg["eval_episodes"],
                    seed=ss.cfg.get("seed"),
                )
            st.success(f"Random baseline: {ss.eval_random.success_rate:.1f}% success rate.")
    with ev_col2:
        if st.button("🤖 Run Trained Agent", type="primary", use_container_width=True):
            with st.spinner("Evaluating trained policy…"):
                ss.eval_agent = evaluate_agent(
                    ss.env,
                    ss.agent,
                    n_episodes=ss.cfg["eval_episodes"],
                )
            ss.ts.last_eval_path = ss.eval_agent.final_path
            st.success(f"Trained agent: {ss.eval_agent.success_rate:.1f}% success rate.")

    if ss.eval_random is not None or ss.eval_agent is not None:
        st.markdown("---")
        res_cols = st.columns(2)
        for col, res, color in [
            (res_cols[0], ss.eval_random, "#4A90D9"),
            (res_cols[1], ss.eval_agent,  "#00FF88"),
        ]:
            if res is None:
                continue
            with col:
                st.markdown(f"#### {res.policy_name}")
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| Episodes | {res.n_episodes} |
| Successful | **{res.n_successful}** |
| Failed | {res.n_failed} |
| **Success Rate** | **{res.success_rate:.1f}%** |
| Avg Steps (success) | {fmt(res.avg_steps_success, 1)} |
| Avg Reward | {res.avg_reward:.4f} |
| Avg Episode Length | {res.avg_episode_length:.1f} |
""")

    if ss.eval_random is not None and ss.eval_agent is not None:
        st.markdown("---")
        st.plotly_chart(
            comparison_bar_chart(ss.eval_random, ss.eval_agent),
            use_container_width=True,
        )

        cmp = compare_results(ss.eval_random, ss.eval_agent)
        st.markdown("#### Relative Improvement")
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Success Rate (percentage-point improvement) | **+{cmp['success_rate_improvement_pp']:.1f} pp** |
| Avg Reward improvement | {fmt(cmp['reward_improvement_pct'], 1, 'N/A')}% |
| Avg Steps reduction (on successes) | {fmt(cmp['steps_improvement_pct'], 1, 'N/A')}% |
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Export & History
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">💾 Export & Experiment History</div>',
                unsafe_allow_html=True)

    ex_c1, ex_c2, ex_c3 = st.columns(3)

    # ── Training metrics CSV ───────────────────────────────────────────────
    with ex_c1:
        st.markdown("#### Training Metrics")
        if ss.ts.metrics:
            df_metrics = pd.DataFrame([{
                "episode":      m.episode,
                "total_reward": m.total_reward,
                "steps":        m.steps,
                "reached_goal": m.reached_goal,
                "epsilon":      m.epsilon,
                "timestamp":    m.timestamp,
            } for m in ss.ts.metrics])
            csv_m = df_metrics.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Metrics CSV",
                data=csv_m,
                file_name="mazemind_training_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"{len(ss.ts.metrics)} episodes recorded.")
        else:
            st.info("No training data yet.")

    # ── Q-table CSV ────────────────────────────────────────────────────────
    with ex_c2:
        st.markdown("#### Q-Table")
        df_q_export = qtable_dataframe(ss.agent.q_table, env.rows, env.cols)
        csv_q = df_q_export.to_csv().encode()
        st.download_button(
            "⬇ Download Q-Table CSV",
            data=csv_q,
            file_name="mazemind_qtable.csv",
            mime="text/csv",
            use_container_width=True,
        )
        # Save full agent JSON.
        agent_json = ss.agent.to_json()
        st.download_button(
            "⬇ Save Agent (JSON)",
            data=agent_json.encode(),
            file_name="mazemind_agent.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Evaluation CSV ─────────────────────────────────────────────────────
    with ex_c3:
        st.markdown("#### Evaluation Results")
        if ss.eval_random or ss.eval_agent:
            records = []
            if ss.eval_random:
                records.append(ss.eval_random.to_dict())
            if ss.eval_agent:
                records.append(ss.eval_agent.to_dict())
            eval_json = json.dumps(records, indent=2)
            st.download_button(
                "⬇ Download Eval JSON",
                data=eval_json.encode(),
                file_name="mazemind_evaluation.json",
                mime="application/json",
                use_container_width=True,
            )
            if ss.eval_random and ss.eval_agent:
                df_eval = pd.DataFrame(records)
                st.download_button(
                    "⬇ Download Eval CSV",
                    data=df_eval.to_csv(index=False).encode(),
                    file_name="mazemind_evaluation.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("Run evaluation first.")

    st.markdown("---")

    # ── Load agent ─────────────────────────────────────────────────────────
    st.markdown("#### Load Saved Agent")
    uploaded = st.file_uploader("Upload agent JSON", type=["json"])
    if uploaded:
        try:
            content = uploaded.read().decode()
            loaded_agent = QLearningAgent.from_json(content)
            # Validate compatibility.
            if loaded_agent.n_states != ss.env.n_states:
                st.error(
                    f"Incompatible agent: loaded n_states={loaded_agent.n_states}, "
                    f"current maze n_states={ss.env.n_states}. "
                    "Adjust maze size to match or use the original configuration."
                )
            else:
                ss.agent = loaded_agent
                st.success(
                    f"Agent loaded! Q-table shape {loaded_agent.q_table.shape}, "
                    f"ε={loaded_agent.epsilon:.4f}"
                )
        except Exception as e:
            st.error(f"Failed to load agent: {e}")

    st.markdown("---")

    # ── Experiment configuration summary ──────────────────────────────────
    st.markdown("#### Current Experiment Configuration")
    cfg_display = {
        **ss.env.config_dict(),
        "alpha":         ss.agent.alpha,
        "gamma":         ss.agent.gamma,
        "epsilon":       round(ss.agent.epsilon, 6),
        "epsilon_min":   ss.agent.epsilon_min,
        "epsilon_decay": ss.agent.epsilon_decay,
        "n_episodes":    ss.cfg["n_episodes"],
        "episodes_run":  ss.ts.current_episode,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.json(cfg_display)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#444; font-size:0.8rem;'>"
    "MazeMind RL · Q-Learning from scratch · Hackathon prototype · "
    "All Q-values computed from real agent–environment interactions"
    "</p>",
    unsafe_allow_html=True,
)
