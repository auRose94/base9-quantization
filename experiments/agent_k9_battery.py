#!/usr/bin/env python3
"""agent_k9 battery — the CLM context-file merge A/B'd on the live server.

Seeds a small Godot-style repo whose docs/conventions.md carries BINDING
rules at the BOTTOM (the mechanically checkable needle) and asks the agent
to write enemy/boss.gd obeying them. The needle sits many KB of tool
payloads after the conventions read, so the pre-merge stub policy truncates
it away in the history while the clm policy keeps it filed at full
fidelity. Verification is done by the harness reading the file from disk —
no model self-report in the verdict.

    agent_k9_battery.py [--draws N] [--arms clm,stub] [--tag NAME]
                        [--port 18445] [--max-turns 12]
Writes results/clm_agent_<tag>.csv + .jsonl and prints a summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent_k9 import run_agent  # noqa: E402

ALLOW = "ctx_tree,ctx_read,ctx_shell"

COMMON_GD = """extends Node
## Shared project constants (neon-jam).

const SPEED_SCALE := 2.5
const GRAVITY := 980.0
const TICK_HZ := 60
const SAVE_KEY := \"save_slot_1\"
"""

UNIT_GD = """class_name Unit
extends CharacterBody2D
## Base unit. All combat scripts subclass Unit and route damage through
## take_damage — nothing else may touch `health` directly.

@export var max_health: int = 100
var health: int = 100
signal died(unit: Unit)

func _ready() -> void:
    health = max_health

func take_damage(amount: int) -> void:
    health -= amount
    if health <= 0:
        died.emit(self)

func heal(amount: int) -> void:
    health = mini(health + amount, max_health)
"""

DECOYS = {
    "camera.gd": """extends Camera2D
##Smoothed follow camera for the arena; anchored to the player rig.

@export var target_path: NodePath
@export var lead: Vector2 = Vector2(48, -24)
var _target: Node2D
var _shake: float = 0.0

func _ready() -> void:
    _target = get_node_or_null(target_path) as Node2D
    position_smoothing_enabled = true

func add_shake(strength: float) -> void:
    _shake = maxf(_shake, strength)

func _process(delta: float) -> void:
    _shake = maxf(_shake - 8.0 * delta, 0.0)
    offset = Vector2.ZERO
    if _target:
        offset = lead
    if _shake > 0.0:
        offset += Vector2(randf_range(-1, 1), randf_range(-1, 1)) * _shake
""",
    "hud.gd": """extends CanvasLayer
##Arena HUD: boss health bar and wave counter live here, driven by signals.

@onready var boss_bar: ProgressBar = %BossBar
@onready var wave_label: Label = %WaveLabel

func bind_boss(unit: Node) -> void:
    if not unit is Unit:
        push_warning(\"HUD bound to a non-Unit node\")
        return
    unit.health_changed.connect(_on_health)
    _on_health(unit.health, unit.max_health)

func _on_health(health: int, max_health: int) -> void:
    boss_bar.max_value = max_health
    boss_bar.value = health

func set_wave(w: int) -> void:
    wave_label.text = \"WAVE %02d\" % w
""",
    "spawner.gd": """extends Node2D
##Enemy spawner for the arena. Spawns WaveUnit children of Unit and kicks
##them toward the player rig; every projectile route mirrors take_damage.

const WAVE_UNIT := preload(\"res://enemy/wave_unit.gd\")
const WAVE_COUNT := [3, 4, 6, 8]

@onready var spawn_points: Array[Marker2D] = []

func _ready() -> void:
    for c in get_children():
        if c is Marker2D:
            spawn_points.append(c)

func start_wave(n: int) -> void:
    var count: int = WAVE_COUNT[mini(n, WAVE_COUNT.size() - 1)]
    for i in count:
        var p: Marker2D = spawn_points[i % spawn_points.size()]
        var unit: Unit = WAVE_UNIT.instantiate()
        unit.position = p.global_position
        add_child(unit)
        unit.start_run()
""",
    "music_player.gd": """extends AudioStreamPlayer
##Looping arena soundtrack; crossfades into the boss motif when a boss
##enters. Volume is set at ready — never tween loudness mid-track.

@export var boss_stream: AudioStream
var _was_boss := false

func enter_boss_phase() -> void:
    if _was_boss:
        return
    _was_boss = true
    stream = boss_stream
    play()
""",
    "savegame.gd": """extends Node
##Save slot plumbing for the arena build. All writes funnel through
##store() so the schema stays in one place.

const VERSION := 3
var _slot: int = 1

func store(payload: Dictionary) -> bool:
    var wrapped := Dictionary()
    wrapped[\"version\"] = VERSION
    wrapped[\"slot\"] = _slot
    wrapped[\"payload\"] = payload
    var f := FileAccess.open(\"user://save_%d.json\" % _slot, FileAccess.WRITE)
    if f == null:
        return false
    f.store_line(JSON.stringify(wrapped))
    f.close()
    return true
""",
    "minimap.gd": """extends SubViewport
##Top-down minimap: watches arena units through group lookup, not exported
##node paths, so late-added enemies show up without rewiring.

const RANGE := 2048

func _process(_delta: float) -> void:
    for unit in get_tree().get_nodes_in_group(\"units\"):
        if unit is Unit:
            queue_redraw()
""",
}

SOFT_STYLE = """# neon-jam — repo conventions (workshop notes, half-curated)

## General style (soft, do your best)

The project grew out of a three-day game jam and the style drifts between
modules. Older files lean on duck typing; newer ones annotate exports.
Comments are welcome and so are docstrings; pick either, not both, in one
file. Naming: snake_case everywhere, private helpers prefixed with a single
underscore, class_name only when the class is genuinely shared.

Rendering is all 2D at integer scale; avoid subpixel offsets in tweens
unless a camera is the consumer. Audio nodes live under the autoload
MusicBus, SFX under FxBus, and both mix buses peak around -6 dB.

Physics: every moving body is a CharacterBody2D; the arena floor is a
StaticBody2D with one generous rectangle, no trimesh. Projectiles are Area2D
with a small circle shape, killed on first contact.

## Mid-notes from the last playtest

- Wave pacing felt too dense past wave 3, the spawner needs a cooldown
  parameter before the boss phase triggers.
- The minimap redraw storm is bounded by queue_redraw() so it is fine.
- Boss arena layout: entry from the lower corridor, pillars at quarter
  distances, no cover geometry that occludes the health bar.
- Camera shake cap is 8.0; anything stronger reads like a crash.
"""

BINDING = """## Binding style rules (enforced by CI — violating PRs are rejected)

1. Every .gd file must open with this exact first line:
   `# project: neon-jam`
2. Enemy and boss scripts must subclass the shared base: `extends Unit`
   (res://src/unit.gd). Never hand-roll health bookkeeping.
3. Damage routing: override `func take_damage(amount: int) -> void` and call
   `super.take_damage(amount)` after your own reaction hooks.
4. Never hardcode movement speed: read `SPEED_SCALE` from res://common.gd
   and multiply it into the velocity.
5. `@export_range` is banned project-wide (we outgrow ranges every jam) —
   use a plain typed `@export var` instead.
"""

CONVENTIONS = SOFT_STYLE + BINDING


def seed_workspace(ws: Path) -> None:
    (ws / "docs").mkdir(parents=True, exist_ok=True)
    (ws / "src").mkdir(parents=True, exist_ok=True)
    (ws / "world").mkdir(exist_ok=True)
    (ws / "common.gd").write_text(COMMON_GD)
    (ws / "src" / "unit.gd").write_text(UNIT_GD)
    (ws / "docs" / "conventions.md").write_text(CONVENTIONS)
    for name, body in DECOYS.items():
        (ws / "world" / name).write_text(body)


CHECKS = (
    ("header_line", lambda t: (t.splitlines() or [""])[0].strip()
     == "# project: neon-jam"),
    ("extends_unit", lambda t: re.search(r"^extends Unit\b", t, re.M) is not None),
    ("take_damage_sig",
     lambda t: re.search(r"func take_damage\(amount\s*:\s*int\)", t) is not None),
    ("super_call", lambda t: "super.take_damage(" in t),
    ("speed_scale", lambda t: "SPEED_SCALE" in t),
    ("no_export_range", lambda t: "@export_range" not in t),
)

TASK = """The Godot project at {ws} needs a new enemy script. Read these three
files in full first: {ws}/docs/conventions.md (the project contract — every
rule under 'Binding style rules' at the bottom is enforced), {ws}/src/unit.gd
and {ws}/common.gd; then skim the world/*.gd modules for idiom. Then create
the file {ws}/enemy/boss.gd defining a Boss enemy: it subclasses the shared
unit base, has 2500 max health, and walks toward the player at the project's
shared speed constant scaled up 1.5x. Every binding rule must hold exactly.
Write the file with ctx_shell (mkdir -p the enemy dir first). After writing,
read the file back once and fix any binding rule it breaks. Answer in prose
when done."""


def verify(boss: Path) -> list:
    if not boss.exists():
        return [(name, False) for name, _ in CHECKS]
    text = boss.read_text()
    return [(name, bool(fn(text))) for name, fn in CHECKS]


def run_one(arm: str, draw: int, ws: Path, port: int, max_turns: int,
            tag: str, jsonl_path: Path) -> dict:
    seed_workspace(ws)
    task = TASK.format(ws=str(ws))
    t0 = time.time()
    ag = run_agent(task, port=port, policy=arm, max_turns=max_turns,
                   allow=ALLOW, verbose=True)
    wall = time.time() - t0
    checks = verify(ws / "enemy" / "boss.gd")
    ok = all(c for _, c in checks)
    cum_prompt = sum(r["prompt_tokens"] for r in ag.ledger)
    row = {
        "tag": tag, "arm": arm, "draw": draw,
        "success": ok, "steps": len(ag.ledger),
        "checks": ",".join(f"{n}:{'p' if c else 'f'}" for n, c in checks),
        "checks_passed": sum(c for _, c in checks), "checks_total": len(CHECKS),
        "cum_prompt_tokens": cum_prompt, "gen_tokens": ag.gen_tokens,
        "wall_s": round(wall, 1), "resident_tool_bytes": ag.resident_tool_bytes(),
        "file_chars": len(ag.store.render()) if ag.store is not None else -1,
        "ctx_edits": getattr(ag, "ctx_edits", 0),
        "boss_bytes": (ws / "enemy" / "boss.gd").stat().st_size
        if (ws / "enemy" / "boss.gd").exists() else 0,
    }
    with jsonl_path.open("a") as fh:
        fh.write(json.dumps(row | {"ledger": ag.ledger,
                                   "final": (ag.answer or "")[:800]}) + "\n")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="clm,stub")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--port", type=int, default=18445)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or time.strftime("%Y%m%d_%H%M%S")
    csv_path = HERE.parent / "results" / f"clm_agent_{tag}.csv"
    jsonl_path = HERE.parent / "results" / f"clm_agent_{tag}.jsonl"
    sel = [x.strip() for x in a.arms.split(",") if x.strip()]
    arms = [x for x in ("stub", "clm") if x in sel] or sel
    rows = []
    for arm in arms:                    # stub first when both: clm must beat the baseline
        for draw in range(1, a.draws + 1):
            # workspaces MUST live inside the repo: lean-ctx sandboxes ctx_read
            # to its project root (cwd) — /tmp paths are access-denied
            ws = (HERE.parent / "results" / "agent_k9_ws"
                  / f"{tag}_{arm}_{draw}")
            if ws.exists():
                shutil.rmtree(ws)
            print(f"\n===== arm={arm} draw={draw} ws={ws} =====", flush=True)
            r = run_one(arm, draw, ws, a.port, a.max_turns, tag, jsonl_path)
            rows.append(r)
            print(f"RESULT arm={arm} draw={draw} success={r['success']} "
                  f"checks={r['checks']} cum_prompt={r['cum_prompt_tokens']} "
                  f"steps={r['steps']} wall={r['wall_s']} "
                  f"resident={r['resident_tool_bytes']}B "
                  f"file_chars={r['file_chars']} "
                  f"ctx_edits={r['ctx_edits']}", flush=True)
    with csv_path.open("w", newline="") as fh:
        names = list(rows[0].keys())
        wr = csv.DictWriter(fh, fieldnames=names)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n=== SUMMARY ({csv_path.name}) ===")
    for arm in ["stub", "clm"]:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        n = len(rs)
        srate = sum(r["success"] for r in rs) / n
        cp = sum(r["cum_prompt_tokens"] for r in rs) / n
        res = sum(r["resident_tool_bytes"] for r in rs) / n
        print(f"{arm:5s}: success {srate:.2f}  cum_prompt {cp:7.0f}  "
              f"resident_tool {res:7.0f}B  steps {sum(r['steps'] for r in rs)/n:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())