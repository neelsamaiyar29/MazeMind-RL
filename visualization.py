"""
visualization.py — MazeMind RL
All Plotly/Matplotlib chart and maze-grid rendering helpers.

Keeping visuals separate from business logic keeps app.py readable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from environment import ACTION_NAMES, MazeEnvironment
from training import TrainingState

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "empty":    "#1E2A3A",
    "start":    "#00D4FF",
    "goal":     "#00FF88",
    "obstacle": "#8B1A1A",
    "agent":    "#FFD700",
    "visited":  "#2D4A6B",
    "path":     "#FF6B35",
    "bg":       "#0E1117",
    "card_bg":  "#1A1F2E",
    "text":     "#FFFFFF",
}

CELL_LABELS = {
    "empty":    "",
    "start":    "S",
    "goal":     "G",
    "obstacle": "■",
    "agent":    "★",
}

ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


# ── Maze grid ─────────────────────────────────────────────────────────────────

def render_maze(
    env: MazeEnvironment,
    agent_pos: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
    eval_path: Optional[List[Tuple[int, int]]] = None,
    show_policy: bool = False,
    policy: Optional[np.ndarray] = None,
    height: int = 420,
) -> go.Figure:
    """
    Render the maze as an annotated Plotly heatmap.

    Cell encoding (z-matrix):
      0 = empty  |  1 = visited  |  2 = path  |  3 = obstacle
      4 = start  |  5 = goal     |  6 = agent
    """
    rows, cols = env.rows, env.cols
    z = np.zeros((rows, cols), dtype=int)

    # Mark visited cells.
    if path:
        for pos in path[:-1]:
            if pos != env.start and pos != env.goal:
                z[pos[0]][pos[1]] = 1

    # Mark eval path.
    if eval_path:
        for pos in eval_path:
            if pos != env.start and pos != env.goal:
                if z[pos[0]][pos[1]] == 0:
                    z[pos[0]][pos[1]] = 2

    # Obstacles.
    for r, c in env.obstacles:
        z[r][c] = 3

    # Start / Goal.
    z[env.start[0]][env.start[1]] = 4
    z[env.goal[0]][env.goal[1]]   = 5

    # Agent.
    if agent_pos is not None:
        z[agent_pos[0]][agent_pos[1]] = 6

    # Colour scale: 7 levels.
    colorscale = [
        [0/6, COLORS["empty"]],
        [1/6, COLORS["visited"]],
        [2/6, COLORS["path"]],
        [3/6, COLORS["obstacle"]],
        [4/6, COLORS["start"]],
        [5/6, COLORS["goal"]],
        [6/6, COLORS["agent"]],
    ]

    # Text annotations.
    annotations_text = []
    for r in range(rows):
        row_text = []
        for c in range(cols):
            if (r, c) == env.goal:
                row_text.append("G")
            elif (r, c) == env.start:
                row_text.append("S")
            elif (r, c) in env.obstacles:
                row_text.append("■")
            elif agent_pos and (r, c) == agent_pos:
                row_text.append("★")
            elif show_policy and policy is not None and z[r][c] not in (3,):
                idx = r * cols + c
                row_text.append(ACTION_ARROWS.get(policy[idx], ""))
            else:
                row_text.append("")
        annotations_text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z,
        colorscale=colorscale,
        zmin=0, zmax=6,
        showscale=False,
        text=annotations_text,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        hovertemplate="Row %{y}, Col %{x}<extra></extra>",
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickmode="array",
            tickvals=list(range(cols)),
            ticktext=[str(c) for c in range(cols)],
            side="top",
            tickfont=dict(color="#888"),
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            tickmode="array",
            tickvals=list(range(rows)),
            ticktext=[str(r) for r in range(rows)],
            autorange="reversed",
            tickfont=dict(color="#888"),
        ),
    )
    return fig


# ── Training charts ────────────────────────────────────────────────────────────

def _base_layout(title: str) -> Dict:
    return dict(
        title=dict(text=title, font=dict(color="#CCC", size=14)),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor="#141922",
        font=dict(color="#CCC"),
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CCC")),
        xaxis=dict(gridcolor="#2A3040", color="#888"),
        yaxis=dict(gridcolor="#2A3040", color="#888"),
    )


def chart_episode_rewards(ts: TrainingState, window: int = 50) -> go.Figure:
    if not ts.metrics:
        return _empty_chart("Episode Rewards")
    episodes = [m.episode for m in ts.metrics]
    rewards  = [m.total_reward for m in ts.metrics]
    ma       = ts.rolling_mean_reward(window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards,
        mode="lines", name="Episode Reward",
        line=dict(color="#4A90D9", width=1),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=ma,
        mode="lines", name=f"{window}-ep Moving Avg",
        line=dict(color="#FFD700", width=2),
    ))
    fig.update_layout(**_base_layout("Episode Reward vs Episode"))
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Total Reward")
    return fig


def chart_steps_per_episode(ts: TrainingState) -> go.Figure:
    if not ts.metrics:
        return _empty_chart("Steps per Episode")
    episodes = [m.episode for m in ts.metrics]
    steps    = [m.steps for m in ts.metrics]
    fig = go.Figure(go.Scatter(
        x=episodes, y=steps,
        mode="lines", name="Steps",
        line=dict(color="#00D4FF", width=1),
    ))
    fig.update_layout(**_base_layout("Steps per Episode"))
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Steps")
    return fig


def chart_success_rate(ts: TrainingState, window: int = 50) -> go.Figure:
    if not ts.metrics:
        return _empty_chart("Success Rate")
    episodes = [m.episode for m in ts.metrics]
    sr       = ts.rolling_success_rate(window)
    fig = go.Figure(go.Scatter(
        x=episodes, y=sr,
        mode="lines", name="Success Rate (%)",
        line=dict(color="#00FF88", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.1)",
    ))
    fig.update_layout(**_base_layout(f"Goal-Reaching Rate (rolling {window} eps)"))
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Success Rate (%)", range=[0, 105])
    return fig


def chart_epsilon(ts: TrainingState) -> go.Figure:
    if not ts.metrics:
        return _empty_chart("Epsilon Decay")
    episodes = [m.episode for m in ts.metrics]
    epsilons = [m.epsilon for m in ts.metrics]
    fig = go.Figure(go.Scatter(
        x=episodes, y=epsilons,
        mode="lines", name="ε (Exploration Rate)",
        line=dict(color="#FF6B35", width=2),
    ))
    fig.update_layout(**_base_layout("Exploration Rate (ε) Decay"))
    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Epsilon", range=[0, 1.05])
    return fig


def _empty_chart(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No training data yet — start training to see this chart.",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(color="#666", size=13),
    )
    fig.update_layout(**_base_layout(title))
    return fig


# ── Q-table visualisations ─────────────────────────────────────────────────────

def qtable_dataframe(q_table: np.ndarray, rows: int, cols: int) -> pd.DataFrame:
    """Convert Q-table to a labelled DataFrame with state labels."""
    n = q_table.shape[0]
    labels = [f"({r},{c})" for r in range(rows) for c in range(cols)]
    df = pd.DataFrame(
        q_table,
        columns=["Up ↑", "Down ↓", "Left ←", "Right →"],
        index=labels[:n],
    )
    df.index.name = "State (row,col)"
    return df.round(4)


def qtable_heatmap(
    q_table: np.ndarray,
    rows: int,
    cols: int,
    action_idx: int = 0,
) -> go.Figure:
    """Heatmap of Q-values for a single action across the grid."""
    action_name = ["Up ↑", "Down ↓", "Left ←", "Right →"][action_idx]
    grid = q_table[:, action_idx].reshape(rows, cols)

    fig = go.Figure(go.Heatmap(
        z=grid,
        colorscale="RdYlGn",
        colorbar=dict(title="Q-value", tickfont=dict(color="#CCC")),
        hovertemplate="Row %{y}, Col %{x}<br>Q=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(f"Q-Values for Action: {action_name}"),
        height=380,
    )
    fig.update_xaxes(
        title="Col", tickmode="array", tickvals=list(range(cols)),
        gridcolor="#2A3040", color="#888",
    )
    fig.update_yaxes(
        title="Row", tickmode="array", tickvals=list(range(rows)),
        autorange="reversed", gridcolor="#2A3040", color="#888",
    )
    return fig


def max_qtable_heatmap(q_table: np.ndarray, rows: int, cols: int) -> go.Figure:
    """Heatmap of max Q-values (value function V) across the grid."""
    grid = np.max(q_table, axis=1).reshape(rows, cols)
    fig = go.Figure(go.Heatmap(
        z=grid,
        colorscale="Viridis",
        colorbar=dict(title="max Q", tickfont=dict(color="#CCC")),
        hovertemplate="Row %{y}, Col %{x}<br>V=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Value Function V(s) = max_a Q(s,a)"),
        height=380,
    )
    fig.update_xaxes(
        title="Col", tickmode="array", tickvals=list(range(cols)),
        gridcolor="#2A3040", color="#888",
    )
    fig.update_yaxes(
        title="Row", tickmode="array", tickvals=list(range(rows)),
        autorange="reversed", gridcolor="#2A3040", color="#888",
    )
    return fig


def comparison_bar_chart(random_result, agent_result) -> go.Figure:
    """Side-by-side bar chart comparing random vs trained policy."""
    categories = ["Success Rate (%)", "Avg Reward", "Avg Steps"]

    def safe(v, default=0):
        return v if v is not None else default

    random_vals = [
        round(random_result.success_rate, 1),
        round(random_result.avg_reward, 2),
        round(random_result.avg_episode_length, 1),
    ]
    agent_vals = [
        round(agent_result.success_rate, 1),
        round(agent_result.avg_reward, 2),
        round(agent_result.avg_episode_length, 1),
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Random Baseline",
        x=categories, y=random_vals,
        marker_color="#4A90D9",
    ))
    fig.add_trace(go.Bar(
        name="Trained Q-Agent",
        x=categories, y=agent_vals,
        marker_color="#00FF88",
    ))
    fig.update_layout(
        **_base_layout("Performance Comparison: Random vs Trained"),
        barmode="group",
        height=380,
    )
    return fig
