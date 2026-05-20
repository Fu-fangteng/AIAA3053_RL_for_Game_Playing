"""
make_demo_gif.py
Generate demo GIFs for the project page.

Produces:
  media/demo_win.gif     — a full winning game (greedy policy)
  media/demo_explore.gif — comparison: high-ε first move vs low-ε late game

Usage:
    python make_demo_gif.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import imageio.v2 as imageio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment.minesweeper_env import MinesweeperEnv
from algorithm.agent import SARSALambdaAgent
from algorithm.features import extract_features, _get_inference

# ── colour scheme (dark demo) ────────────────────────────────
BG        = "#060b14"
CARD      = "#111827"
HIDDEN    = "#1a2840"
HIDDEN_ED = "#2a3a50"
OPEN_BG   = "#0e1926"
ACCENT    = "#14b8a6"
MINE_RED  = "#ef4444"
NUM_COLS  = ["", "#3b82f6", "#22c55e", "#ef4444", "#1e3a8a",
             "#7f1d1d", "#0891b2", "#374151", "#6b7280"]


def load_agent(run_dir):
    data = np.load(Path(run_dir) / "final_model.npz")
    agent = SARSALambdaAgent()
    agent.w = data["w"]
    agent.epsilon = 0.0
    return agent


def draw_board(ax, env, chosen_action=None, mines_visible=False,
               title="", step=0, epsilon=None, highlight_inference=False):
    """Draw the board onto ax."""
    ax.set_facecolor(BG)
    grid = env._get_state()["map"]
    rows, cols = env.rows, env.cols

    # Build inference sets for highlighting
    state = env._get_state()
    mine_set, safe_set = _get_inference(state)

    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            x, y = c, rows - 1 - r  # flip y

            if v == -1:  # hidden
                col = HIDDEN
                edge = HIDDEN_ED
                lw = 0.6
                if highlight_inference:
                    if (r, c) in safe_set:
                        col = "#14423a"; edge = ACCENT; lw = 1.8
                    elif (r, c) in mine_set:
                        col = "#3b1a1a"; edge = "#ef4444"; lw = 1.8
                if chosen_action and chosen_action == (r, c):
                    col = "#0e3a35"; edge = ACCENT; lw = 2.5
                rect = mpatches.FancyBboxPatch(
                    (x + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02", linewidth=lw,
                    edgecolor=edge, facecolor=col)
                ax.add_patch(rect)
                # dot
                ax.plot(x + 0.5, y + 0.5, "o",
                        color="#ffffff22", ms=2.5, zorder=3)
                if chosen_action == (r, c):
                    # pulsing cursor mark
                    ax.plot(x + 0.5, y + 0.5, "*",
                            color=ACCENT, ms=10, zorder=5)

            elif v == 9:  # mine (detonated)
                rect = mpatches.FancyBboxPatch(
                    (x + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02", linewidth=1.5,
                    edgecolor=MINE_RED, facecolor="#3b1a1a")
                ax.add_patch(rect)
                ax.text(x + 0.5, y + 0.5, "💥",
                        ha="center", va="center", fontsize=9, zorder=4)

            elif v == 0:  # revealed 0
                rect = mpatches.FancyBboxPatch(
                    (x + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02", linewidth=0.3,
                    edgecolor="#14b8a615", facecolor=OPEN_BG)
                ax.add_patch(rect)

            else:  # revealed 1-8
                rect = mpatches.FancyBboxPatch(
                    (x + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02", linewidth=0.3,
                    edgecolor="#14b8a620", facecolor="#0c1e30")
                ax.add_patch(rect)
                ax.text(x + 0.5, y + 0.5, str(v),
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=NUM_COLS[v] if v < len(NUM_COLS) else "#e8ecf2",
                        zorder=4)

    # Mines visible (end of game)
    if mines_visible and not env._win:
        for r2 in range(rows):
            for c2 in range(cols):
                if env._mine_map[r2][c2] and not env._visible[r2][c2]:
                    x2, y2 = c2, rows - 1 - r2
                    rect = mpatches.FancyBboxPatch(
                        (x2 + 0.05, y2 + 0.05), 0.9, 0.9,
                        boxstyle="round,pad=0.02", linewidth=1,
                        edgecolor=MINE_RED + "80", facecolor="#3b1a1a80",
                        alpha=0.7)
                    ax.add_patch(rect)
                    ax.text(x2 + 0.5, y2 + 0.5, "💣",
                            ha="center", va="center", fontsize=8,
                            alpha=0.85, zorder=4)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    info = f"  Step {step}"
    if epsilon is not None:
        info += f"  |  ε = {epsilon:.3f}"
    ax.set_title(title + info,
                 color="#a1aab8", fontsize=9,
                 pad=4, loc="left",
                 fontfamily="monospace")


def make_win_gif(agent, seed=42, out_path="media/demo_win.gif"):
    """Record a full winning game as an animated GIF."""
    import random
    random.seed(seed + 999)
    np.random.seed(seed + 999)

    env = MinesweeperEnv(grid_size=(9, 9), num_mines=10)
    env.reset()

    # Re-seed until first move survives (avoid first-click death for clean demo)
    for attempt in range(200):
        state = env.reset()
        valid = env.get_valid_actions()
        action = agent.select_action(state, valid)
        state2, _, done, info = env.step(action)
        if not done:
            break

    frames = []
    step = 0
    max_steps = 120

    def capture(chosen=None, mines_vis=False, title="", n_dup=1):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        fig.patch.set_facecolor(BG)
        draw_board(ax, env, chosen_action=chosen,
                   mines_visible=mines_vis, title=title,
                   step=step, highlight_inference=True)
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(h, w, 4)[:, :, [1, 2, 3, 0]]  # ARGB→RGBA
        plt.close(fig)
        for _ in range(n_dup):
            frames.append(img)

    # First frame: board after first move
    capture(title="SARSA(λ) — greedy play  ", n_dup=3)

    while not state2["done"] and step < max_steps:
        step += 1
        state = state2
        valid = env.get_valid_actions()
        if not valid:
            break
        action = agent.select_action(state, valid)
        # Show cursor frame
        capture(chosen=action, title="SARSA(λ) — greedy play  ", n_dup=2)
        state2, reward, done, info = env.step(action)
        # Show reveal frame
        capture(title="SARSA(λ) — greedy play  ", n_dup=2)
        if done:
            break

    # Final frame
    win = info.get("win", False)
    title = "✓ WIN" if win else "✗ MINE HIT"
    capture(mines_vis=True,
            title=f"SARSA(λ) — {title}  ",
            n_dup=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=4, loop=0)
    print(f"[GIF] saved {out_path}  ({len(frames)} frames, {os.path.getsize(out_path)//1024} KB)")
    return win


def make_epsilon_compare_gif(agent, out_path="media/demo_epsilon.gif"):
    """
    Side-by-side comparison:
    left  = early training  (high ε, random-ish moves)
    right = late training   (low ε, greedy moves)
    """
    import random

    def play_game(eps, seed):
        random.seed(seed); np.random.seed(seed)
        env = MinesweeperEnv(grid_size=(9, 9), num_mines=10)
        a = SARSALambdaAgent()
        a.w = agent.w.copy()
        a.epsilon = eps
        steps = []
        state = env.reset()
        for _ in range(80):
            valid = env.get_valid_actions()
            if not valid: break
            act = a.select_action(state, valid)
            steps.append((env._get_state(), act, eps))
            state, _, done, _ = env.step(act)
            steps.append((env._get_state(), None, eps))
            if done: break
        return steps, env

    frames = []
    for seed in range(3):
        steps_high, env_h = play_game(0.9, seed)
        steps_low,  env_l = play_game(0.0, seed)

        n = max(len(steps_high), len(steps_low))
        for i in range(0, n, 2):
            sh = steps_high[i] if i < len(steps_high) else steps_high[-1]
            sl = steps_low[i]  if i < len(steps_low)  else steps_low[-1]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.5))
            fig.patch.set_facecolor(BG)
            fig.suptitle(
                "Exploration in Action:  ε=0.9 (early training)  vs  ε=0.0 (greedy policy)",
                color="#a1aab8", fontsize=9, fontfamily="monospace")

            # Temporarily swap env state for drawing (hacky but OK for demo)
            for ax, (st, act, eps) in [(ax1, sh), (ax2, sl)]:
                ax.set_facecolor(BG)
                draw_board.__wrapped__ = False  # not wrapped
                # Draw from state dict manually
                grid = st["map"]
                rows, cols = 9, 9
                state_obj = st
                mine_set, safe_set = _get_inference(state_obj)
                for r in range(rows):
                    for c in range(cols):
                        v = grid[r][c]
                        x, y = c, rows-1-r
                        if v == -1:
                            col = HIDDEN; edge = HIDDEN_ED; lw = 0.6
                            if (r, c) in safe_set: col="#14423a"; edge=ACCENT; lw=1.8
                            elif (r, c) in mine_set: col="#3b1a1a"; edge="#ef4444"; lw=1.8
                            if act and act == (r, c): col="#0e3a35"; edge=ACCENT; lw=2.5
                            rect = mpatches.FancyBboxPatch((x+.05,y+.05),.9,.9,
                                boxstyle="round,pad=0.02",lw=lw,ec=edge,fc=col)
                            ax.add_patch(rect)
                            ax.plot(x+.5,y+.5,"o",color="#ffffff22",ms=2.5,zorder=3)
                            if act and act==(r,c):
                                ax.plot(x+.5,y+.5,"*",color=ACCENT,ms=10,zorder=5)
                        elif v==9:
                            rect=mpatches.FancyBboxPatch((x+.05,y+.05),.9,.9,
                                boxstyle="round,pad=0.02",lw=1.5,ec=MINE_RED,fc="#3b1a1a")
                            ax.add_patch(rect)
                            ax.text(x+.5,y+.5,"💥",ha="center",va="center",fontsize=9,zorder=4)
                        elif v==0:
                            rect=mpatches.FancyBboxPatch((x+.05,y+.05),.9,.9,
                                boxstyle="round,pad=0.02",lw=0.3,ec="#14b8a615",fc=OPEN_BG)
                            ax.add_patch(rect)
                        else:
                            rect=mpatches.FancyBboxPatch((x+.05,y+.05),.9,.9,
                                boxstyle="round,pad=0.02",lw=0.3,ec="#14b8a620",fc="#0c1e30")
                            ax.add_patch(rect)
                            ax.text(x+.5,y+.5,str(v),ha="center",va="center",
                                    fontsize=11,fontweight="bold",
                                    color=NUM_COLS[v] if v<len(NUM_COLS) else "#e8ecf2",zorder=4)
                ax.set_xlim(0,cols); ax.set_ylim(0,rows)
                ax.set_aspect("equal"); ax.axis("off")
                label = f"ε = {eps:.1f}  {'(explore)' if eps>0.3 else '(exploit)'}"
                ax.set_title(label, color="#a1aab8", fontsize=9,
                             fontfamily="monospace", pad=4)

            fig.tight_layout(pad=0.6)
            fig.canvas.draw()
            w,h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_argb(),dtype=np.uint8)
            img = img.reshape(h,w,4)[:,:,[1,2,3,0]]
            plt.close(fig)
            frames.append(img)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=3, loop=0)
    print(f"[GIF] saved {out_path}  ({len(frames)} frames, {os.path.getsize(out_path)//1024} KB)")


if __name__ == "__main__":
    RUN = "training_evaluation/runs/20260422_080515_9x9_m10_main"
    print("[GIF] Loading model from", RUN)
    agent = load_agent(RUN)

    # Try multiple seeds to get a win
    win = False
    for seed in range(10):
        win = make_win_gif(agent, seed=seed,
                           out_path="media/demo_win.gif")
        if win:
            print(f"[GIF] Got a winning game on seed={seed}")
            break
    if not win:
        print("[GIF] Using best available game (no guaranteed win found in 10 seeds)")

    make_epsilon_compare_gif(agent, out_path="media/demo_epsilon.gif")
    print("[GIF] Done. Files in media/")
