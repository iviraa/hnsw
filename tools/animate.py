#!/usr/bin/env python3
# Animates one search over the graph dumped by trace. Usage: animate.py graph.json search.gif

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

BG = "#e4e4e4"
NODE = "#a8a8a8"
EDGE = "#dcdcdc"
GHOST = "#c6c6c6"
INK = "#111111"
LOOK = "#6f6f6f"
RED = "#b91c1c"
DPI = 150

src = sys.argv[1] if len(sys.argv) > 1 else "graph.json"
out = sys.argv[2] if len(sys.argv) > 2 else "search.gif"
g = json.load(open(src))

pts = g["points"]
levels = g["levels"]
query = g["query"]
steps = g["steps"]
edges = {layer["level"]: layer["edges"] for layer in g["layers"]}
members = {lv: [i for i, l in enumerate(levels) if l >= lv] for lv in edges}
order = sorted(edges, reverse=True)

plt.rcParams["font.family"] = "monospace"
fig = plt.figure(figsize=(8.0, 9.4), dpi=DPI, facecolor="white")
ax = fig.add_axes([0.06, 0.245, 0.88, 0.685])
key = fig.add_axes([0.06, 0.125, 0.88, 0.085])

for a in (ax, key):
    a.set_xticks([])
    a.set_yticks([])
ax.set_aspect("equal")
for s in ax.spines.values():
    s.set_color("#dddddd")
for s in key.spines.values():
    s.set_visible(False)

xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
pad = (max(xs) - min(xs)) * 0.05
LIM = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

heading = fig.text(0.06, 0.962, "", fontsize=15, color=INK)
stack = fig.text(0.06, 0.940, "", fontsize=10, color="#888888")
caption = fig.text(0.06, 0.088, "", fontsize=10.5, color="#444444", va="top", linespacing=1.8)

# A key, so a frame explains itself without a caption beside it.
KEY = [
    (0,  1, dict(s=150, c=INK), "search is here"),
    (26, 1, dict(s=34, c=INK), "opened already"),
    (52, 1, dict(s=80, facecolors="none", edgecolors=LOOK, linewidths=1.3),
     "measured, not opened"),
    (82, 1, dict(s=180, c=RED, marker="x", linewidths=2.4), "the query"),
    (0,  0, dict(s=110, facecolors="none", edgecolors=RED, linewidths=1.3, alpha=0.7),
     "held as a result"),
    (26, 0, dict(s=26, c=NODE), "node on this layer"),
    (52, 0, dict(s=14, c=BG), "node on a lower one"),
]
key.set_xlim(0, 100)
key.set_ylim(-0.6, 1.6)
for x, row, style, label in KEY:
    key.scatter([x + 1], [row], zorder=3, **style)
    key.text(x + 3.4, row, label, fontsize=9, color="#666666", va="center")

# Frames repeat steps so the eye can keep up, and pause on layer changes.
plan = []
for i, st in enumerate(steps):
    new_layer = i == 0 or st["level"] != steps[i - 1]["level"]
    plan += [i] * (6 if new_layer else 2)
plan += [len(steps) - 1] * 12

NOTE = {
    "start": "arrived at the node the layer above handed down",
    "hop": "moved to the neighbour closest to the query",
    "settle": "no neighbour is closer, so this layer is finished",
    "expand": "took the nearest unexplored candidate, opened its neighbours",
}


def draw(frame):
    i = plan[frame]
    st = steps[i]
    lv = st["level"]
    last = frame == len(plan) - 1
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(LIM[0], LIM[1])
    ax.set_ylim(LIM[2], LIM[3])

    ax.scatter(xs, ys, s=8, c=BG, zorder=1, linewidths=0)

    seg = [[pts[a], pts[b]] for a, b in edges[lv]]
    if seg:
        ax.add_collection(LineCollection(seg, colors=EDGE, linewidths=0.8, zorder=2))
    on = members[lv]
    ax.scatter([pts[j][0] for j in on], [pts[j][1] for j in on],
               s=18, c=NODE, zorder=3, linewidths=0)

    # Where the search went on the layers already passed.
    for earlier in steps[:i]:
        if earlier["level"] == lv:
            continue
        p = earlier["path"]
        if len(p) > 1:
            ax.plot([pts[j][0] for j in p], [pts[j][1] for j in p],
                    color=GHOST, lw=1.3, zorder=4)

    fresh = st["fresh"]
    if fresh and not last:
        ax.scatter([pts[j][0] for j in fresh], [pts[j][1] for j in fresh],
                   s=80, facecolors="none", edgecolors=LOOK, linewidths=1.3, zorder=5)
        for j in fresh:
            ax.plot([pts[st["at"]][0], pts[j][0]], [pts[st["at"]][1], pts[j][1]],
                    color=LOOK, lw=0.7, alpha=0.6, zorder=5)

    for j in st["result"]:
        ax.scatter(pts[j][0], pts[j][1], s=110, facecolors="none",
                   edgecolors=RED, linewidths=1.1, alpha=0.5, zorder=6)

    # Upper layers walk node to node. Layer 0 pops by distance, so only nodes are marked.
    path = st["path"]
    if lv > 0:
        ax.plot([pts[j][0] for j in path], [pts[j][1] for j in path],
                color=INK, lw=2.0, zorder=7)
    ax.scatter([pts[j][0] for j in path], [pts[j][1] for j in path],
               s=34, c=INK, zorder=8, linewidths=0)
    ax.scatter(pts[st["at"]][0], pts[st["at"]][1], s=150, c=INK,
               edgecolors="white", linewidths=1.8, zorder=9)

    if last:
        for j in g["answer"]:
            ax.scatter(pts[j][0], pts[j][1], s=110, c=RED, zorder=10, linewidths=0)

    ax.scatter(query[0], query[1], s=200, c=RED, marker="x", linewidths=2.6, zorder=11)

    stage = "greedy walk, one node at a time" if lv else f"beam search, keeps ef={g['ef']}"
    plural = "node" if len(on) == 1 else "nodes"
    heading.set_text(f"layer {lv}  ·  {len(on)} {plural}  ·  {stage}")
    stack.set_text("visiting layers top down:  " +
                   " ".join(f"[{l}]" if l == lv else f" {l} " for l in order))

    if last:
        touched = len(set().union(*[set(s["fresh"]) for s in steps]))
        opened = len({s["at"] for s in steps})
        hit = sum(1 for j in g["answer"] if j in g["exact"])
        caption.set_text(
            f"done in {len(steps)} steps. the {len(g['answer'])} filled red dots are what it "
            f"returned,\n{hit} of which are the true nearest. {touched} of {len(pts)} points were "
            f"measured\nagainst the query, and only {opened} of those were opened."
        )
    else:
        held = (f"holding the {len(st['result'])} closest found so far"
                if lv == 0 else "carrying one node, no results collected yet")
        caption.set_text(f"step {i + 1} of {len(steps)}: {NOTE[st['kind']]}\n{held}")


anim = FuncAnimation(fig, draw, frames=len(plan), interval=250)
anim.save(out, writer=PillowWriter(fps=4), dpi=DPI)
print(f"{out}: {len(plan)} frames from {len(steps)} steps")
