#!/usr/bin/env python3
"""Measure how well a model writes Godot 4 GDScript — mechanically.

Three signals per task, all checkable without a human:

  parses    the reply compiles under the real Godot 4 parser (gdscript_verify)
  api       every required API name is present (CharacterBody2D, not KinematicBody2D)
  no_godot3 no forbidden Godot-3 name appears anywhere in the reply

`correct` = parses AND api AND no_godot3. That last pair is the point: the
pre-fine-tune 1.5B answers "move a player with the arrow keys" with
`extends KinematicBody2D` and `move()` — Godot 3 API that Godot 4 removed — and
asked for an FPS print it invented `GD.get_fps()`. Both are exactly what a
parser can catch and a perplexity number cannot.

Decoding is greedy (temperature 0) so before/after runs are comparable.

    python3 eval_gdscript.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --tag base
    python3 eval_gdscript.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --adapter out/qlora --tag tuned
    python3 eval_gdscript.py --model ... --k9 results/qwen_coder_1.5b_k63_embed99.k9
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch                                                      # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer      # noqa: E402

from gdscript_verify import GodotVerifier, OK                     # noqa: E402

SYSTEM = ("You are an expert Godot 4 game developer. You write correct, "
          "idiomatic GDScript 4 — never Godot 3 APIs. Reply with code only.")

# must / must_not are lowercase substring checks against the whole reply.
# Godot-3 markers, as regexes. Substring matching gets these wrong in both
# directions: "@onready var" contains "onready var", and ".connect(\"name\",
# callable)" is VALID Godot 4 (only the 3-arg ".connect(\"name\", self, \"m\")"
# form was removed). Both false positives cost us a correct answer.
G3 = [
    ("kinematic_body", r"\bKinematicBody(2D|3D)?\b"),
    ("spatial", r"\bSpatial\b"),
    ("get_world()", r"\bget_world\s*\(\s*\)"),
    ("yield(", r"\byield\s*\("),
    ("bare onready var", r"(?<!@)\bonready\s+var"),
    ("bare export(", r"(?<!@)\bexport\s*\("),
    ("connect 3-arg", r'\.connect\s*\(\s*"[^"]*"\s*,\s*[^,)]+\s*,'),
    ("instance()", r"\.instance\s*\(\s*\)"),
    ("change_scene(str)", r'\bchange_scene\s*\(\s*"'),
    ("GD.get_fps", r"\bGD\.get_fps\b"),
    ("OS.get_ticks", r"\bOS\.get_ticks\b"),
    ("PoolStringArray", r"\bPool(String|Byte|Int|Real|Vector)\w*Array\b"),
    ("rand_range(", r"\brand_range\s*\("),
]


def g3_hits(code: str) -> list:
    return [lab for lab, rx in G3 if re.search(rx, code)]


TASKS = [
    dict(id="player_move", must=["characterbody2d", "velocity", "move_and_slide"],
         must_not=["kinematicbody2d"],
         prompt="Write a complete Godot 4 script for a 2D player that moves on "
                "the horizontal axis with the ui_left/ui_right actions, uses "
                "gravity, and can jump on ui_accept."),
    dict(id="area_collision", must=["area2d", "signal"], must_not=[],
         prompt="Write a complete Godot 4 script for an Area2D pickup that "
                "emits a signal when a body enters it."),
    dict(id="timer_await", must=["create_timer", "await"], must_not=[],
         prompt="Write a complete Godot 4 script that waits 2 seconds using a "
                "scene tree timer and then prints hello, using await."),
    dict(id="instantiate", must=["instantiate", "add_child"], must_not=["instance()"],
         prompt="Write a complete Godot 4 script that loads a PackedScene from "
                "res://bullet.tscn, instantiates it and adds it as a child of "
                "the current node."),
    dict(id="tween", must=["create_tween"], must_not=[],
         prompt="Write a complete Godot 4 script that tweens its own scale to "
                "1.5 over 0.5 seconds using create_tween."),
    dict(id="export_var", must=["@export"], must_not=[],
         prompt="Write a complete Godot 4 script with an exported float named "
                "speed with a range hint from 0 to 100, defaulting to 50."),
    dict(id="file_write", must=["fileaccess"], must_not=[],
         prompt="Write a complete Godot 4 script that writes the text hello to "
                "user://save.txt using FileAccess."),
    dict(id="json_save", must=["json", "fileaccess"], must_not=[],
         prompt="Write a complete Godot 4 script that saves a dictionary "
                "{level: 3} to user://save.json as JSON and loads it back."),
    dict(id="randi", must=["randi_range"], must_not=["randi() %"],
         prompt="Write a complete Godot 4 script that prints a random integer "
                "between 1 and 6 inclusive."),
    dict(id="change_scene", must=["change_scene_to_file"], must_not=[],
         prompt="Write a complete Godot 4 script that changes the current scene "
                "to res://level2.tscn when the ui_accept action is pressed."),
    dict(id="enum_match", must=["enum", "match "], must_not=[],
         prompt="Write a complete Godot 4 script that declares an enum State "
                "with IDLE, RUNNING and JUMPING and uses match to print a "
                "message for the current state."),
    dict(id="callable_connect", must=["pressed.connect"],
         must_not=[],
         prompt="Write a complete Godot 4 script that connects a Button's "
                "pressed signal to a method using a Callable, and disconnects "
                "it when done."),
    dict(id="raycast2d", must=["direct_space_state", "intersect_ray"], must_not=[],
         prompt="Write a complete Godot 4 script that does a 2D raycast from "
                "its own position to a target position using "
                "PhysicsDirectSpaceState2D and prints what it hit."),
    dict(id="autoload_style", must=["extends node"], must_not=[],
         prompt="Write a complete Godot 4 script suitable for use as an autoload "
                "singleton that keeps a score and has add_score(amount) and "
                "reset() methods."),
    dict(id="resource_class", must=["class_name", "@export"], must_not=[],
         prompt="Write a complete Godot 4 script defining a custom Resource "
                "class named ItemData with exported fields name and price."),
    dict(id="audio_play", must=["audiostreamplayer", "play()"], must_not=[],
         prompt="Write a complete Godot 4 script that gets an AudioStreamPlayer "
                "child and plays it when a body enters its Area2D."),
    dict(id="fps_label", must=["engine"],
         must_not=[],
         prompt="Write a complete Godot 4 script that updates a Label child with "
                "the current frames per second every frame."),
    dict(id="pause_game", must=["get_tree()", "paused"], must_not=[],
         prompt="Write a complete Godot 4 script that pauses and unpauses the "
                "game tree when ui_cancel is pressed, keeping the node "
                "processing while paused."),
    dict(id="input_poll", must=["input.get_vector", "characterbody2d"],
         must_not=["get_action_strength"],
         prompt="Write a complete Godot 4 script for top-down 8-way movement "
                "using Input.get_vector with ui_left/ui_right/ui_up/ui_down."),
    dict(id="array_sum", must=["func"], must_not=[],
         prompt="Write a complete Godot 4 script with a function that takes an "
                "Array of floats and returns their sum, and prints the result "
                "for [1.0, 2.0, 3.5]."),
    dict(id="sprite_flip", must=["sprite2d", "flip_h"], must_not=[],
         prompt="Write a complete Godot 4 script that flips a Sprite2D child "
                "horizontally based on horizontal input direction."),
    dict(id="state_machine", must=["enum", "signal"], must_not=[],
         prompt="Write a complete Godot 4 script for a simple state machine "
                "node with a state_changed signal and a change_state(new_state) "
                "method, using an enum."),
    dict(id="camera_follow", must=["camera2d"], must_not=[],
         prompt="Write a complete Godot 4 script that makes a Camera2D child "
                "follow a target node smoothly in _process."),
    dict(id="dictionary_iter", must=[], must_not=[],
         prompt="Write a complete Godot 4 script that iterates a Dictionary of "
                "item names to counts and prints each key and value."),
]


def extract_code(reply: str) -> str:
    """First fenced block if present, else the whole reply."""
    m = re.search(r"```(?:gdscript|gd)?\s*\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).strip()


def gen(model, tok, device, prompt, max_new=640, sampling=None):
    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=prompt)]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        kw = dict(max_new_tokens=max_new, do_sample=False,
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if sampling:                       # multi-draw mode: sample, majority-vote
            kw.update(do_sample=True, temperature=sampling["temperature"],
                      top_p=sampling["top_p"], top_k=sampling["top_k"])
        out = model.generate(**enc, **kw)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter dir (peft)")
    ap.add_argument("--k9", help="load a K9 file into the model first")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--draws", type=int, default=1,
                    help="draws per task. 1 = the historical greedy decode "
                         "(backward compatible); N>1 = decoded at temperature/"
                         "top_p/top_k with majority voting per check, which "
                         "measures capability instead of one greedy knife-edge "
                         "(single greedy draws flipped ~5/24 tasks between "
                         "near-identical quantized models)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--runtime", action="store_true",
                    help="also instantiate+run each reply (slower, stronger)")
    ap.add_argument("--tasks", help="comma-separated task ids to run")
    ap.add_argument("--nf4", action="store_true",
                    help="load the base in 4-bit NF4 — for models too large for "
                         "a bf16 host (14B+). K9 load and NF4 are mutually "
                         "exclusive (the K9 path needs the bf16 host)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.k9 and a.nf4:
        print("--k9 and --nf4 are mutually exclusive: the K9 path replaces "
              "decoder tensors in a bf16 host")
        return 2
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.nf4:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            a.model, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True),
            low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation="sdpa").to(dev).eval()
    if a.k9:
        import k9
        t0 = time.time()
        with k9.K9File(a.k9) as f:
            names = set(f.names())
            for name, rec in f.iter_tensors():
                mod = (model.model.embed_tokens if name == "__embed__"
                       else model.lm_head if name == "__lm_head__"
                       else model.get_submodule(name))
                k9.load_into(mod, rec, rec["group"], device=dev, row_chunk=4096)
        print(f"loaded K9 {Path(a.k9).name} in {time.time()-t0:.0f}s", flush=True)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
        print(f"loaded adapter {a.adapter}", flush=True)

    tasks = TASKS
    if a.tasks:
        want = {t.strip() for t in a.tasks.split(",")}
        tasks = [t for t in TASKS if t["id"] in want]

    out = Path(a.out) if a.out else HERE / "out" / f"eval_{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    sampling = None if a.draws == 1 else dict(
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    rows = []
    agg = dict(tasks=0, parses=0, api=0, no_godot3=0, correct=0,
               draw_correct=0, total_draws=0)
    t0 = time.time()
    with GodotVerifier(workdir=str(HERE / "data" / "eval_proj")) as v, open(out, "w") as fo:
        for t in tasks:
            ds = []
            for _ in range(a.draws):
                reply = gen(model, tok, dev, t["prompt"], a.max_new, sampling)
                code = extract_code(reply)
                low = code.lower()
                r = v.check_runtime(code, name=t["id"]) if a.runtime else v.check_parse(code, t["id"])
                forb = g3_hits(code) + [x for x in t["must_not"] if x.lower() in low]
                ds.append(dict(status=r.status, chars=len(code),
                               parses=r.status == OK,
                               api=all(x.lower() in low for x in t["must"]),
                               no_godot3=not forb,
                               forbidden=forb,
                               missing=[x for x in t["must"] if x.lower() not in low],
                               errors=[e["message"][:110] for e in r.errors[:3]],
                               code=code))
            # majority per check (ties → False; odd --draws has no ties)
            checks = {k: sum(d[k] for d in ds) * 2 > len(ds)
                      for k in ("parses", "api", "no_godot3")}
            checks["correct"] = all(checks.values())
            agg["tasks"] += 1
            for k in ("parses", "api", "no_godot3", "correct"):
                agg[k] += checks[k]
            agg["draw_correct"] += sum(d["parses"] and d["api"] and d["no_godot3"]
                                       for d in ds)
            agg["total_draws"] += len(ds)
            best = next((d for d in ds if d["parses"] and d["api"] and d["no_godot3"]),
                        max(ds, key=lambda d: (d["parses"], d["api"])))
            row = dict(tag=a.tag, id=t["id"], status=best["status"],
                       chars=best["chars"], errors=best["errors"],
                       missing=best["missing"], forbidden=best["forbidden"],
                       code=best["code"], draws=a.draws,
                       draw_results=[{k: v for k, v in d.items() if k != "code"}
                                     for d in ds],
                       **checks)
            rows.append(row)
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            flag = "OK " if checks["correct"] else ("~  " if checks["parses"] else "BAD")
            print(f"  [{flag}] {t['id']:18s} {best['status']:12s}"
                  f" {sum(d['parses'] for d in ds)}/{a.draws} parse |"
                  f" {sum(d['api'] for d in ds)}/{a.draws} api", flush=True)

    n = max(agg["tasks"], 1)
    summary = dict(tag=a.tag, model=a.model, adapter=a.adapter, k9=a.k9,
                   draws=a.draws,
                   draw_correct_pct=round(100 * agg["draw_correct"]
                                          / max(agg["total_draws"], 1), 1),
                   seconds=round(time.time() - t0),
                   **{k: agg[k] for k in ("tasks", "parses", "api", "no_godot3", "correct")},
                   **{f"{k}_pct": round(100 * agg[k] / n, 1) for k in
                      ("parses", "api", "no_godot3", "correct")})
    print("\n" + json.dumps(summary, indent=2))
    Path(str(out).replace(".jsonl", ".summary.json")).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
