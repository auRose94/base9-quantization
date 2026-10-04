#!/usr/bin/env python3
"""Build an SFT corpus from the Godot-4 project dumps.

Pipeline, per project (one Godot import + one Godot scan):

    recon (write files + project.godot)  ->  godot --import  ->  godot_scan.gd
                                                                    |
                       per-file verdict: clean / context / broken  <-+

Two things make this worth the process cost. First, the scan validates every
`.gd` in a single Godot run: doing the same work with `--check-only --script`
per file costs a process launch each (~0.3 s), which is a minute per project.
(`load()` alone is NOT a validator — it returns a non-null resource for a script
that failed to compile; `reload()` returns ERR_PARSE_ERROR instead.)

Second, the verdict is three-way, not two-way. A file that fails only because it
references an **autoload singleton** (`UI`, `Enums`, `PS`) or a sibling class or
a `.tscn` the dump does not contain is still *valid Godot 4 code* — the failure
is a fact about our reconstruction, not about the code. Only files with real
syntax/API errors are dropped. Keeping the "context" files is most of the
corpus: on RDS-Game, 54/96 parse outright and nearly all 42 failures are
autoload references.

Usage:
    python3 build_sft.py --limit 40 --out out/trial.jsonl      # quick trial
    python3 build_sft.py --workers 12 --out out/godot4.jsonl   # full pass
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_corpus as bc                                    # noqa: E402
from gdscript_verify import _RE_SCRIPT                       # noqa: E402

GODOT = "/usr/bin/godot-mono"
WORK = HERE / "data" / "sft_work"
SCAN = "godot_scan.gd"

MIN_CHARS, MAX_CHARS = 200, 60_000
IMPORT_TIMEOUT, SCAN_TIMEOUT = 120, 90   # the scan is normally <2 s

# Failures that mean "our reconstruction is incomplete", not "this code is bad".
CONTEXT_MSG = re.compile(
    r'Identifier "[^"]+" not declared|'
    r'Could not find type "[^"]+"|'
    r'Could not find base class "[^"]+"|'
    r'Preload file "[^"]+" does not exist|'
    r'Cannot open file "[^"]+"|'
    r'Could not resolve external class member|'
    r'Could not resolve the class|'
    r'Cannot load|File not found'
)
# Failures that mean the code itself is wrong for Godot 4.
BROKEN_MSG = re.compile(
    r"keyword was removed in Godot 4|"
    r"Expected .* after|Expected .* instead|Expected statement|"
    r"Unexpected identifier|Unexpected .* in class body|"
    r"Too many arguments|Invalid argument|Invalid call|"
    r"Function .* not found|Static function .* not found|"
    r"Cannot use .* as|Cannot assign|Cannot pass|"
    r"Unterminated|Invalid escape|Invalid indentation|"
    r"shadows a native class|hides a native class|"
    r"doesn't match the parent|"
    r"is not a valid|"
    r"Parse Error: Expected|"
    r"Compile Error: "
)


def errors_by_file(stderr: str) -> dict:
    """Map res://path -> [message] from Godot's SCRIPT ERROR blocks."""
    out: dict = {}
    for m in _RE_SCRIPT.finditer(stderr):
        f = m["file"]
        if f.startswith("res://"):
            out.setdefault(f, []).append(m["msg"].strip())
    return out


def run(cmd, timeout):
    """Run Godot with a hard wall-clock cap, killing the whole process group on
    timeout. Some scraped projects contain a `@tool`/editor-plugin script that
    blocks forever under `--headless` (observed on the scan step), so a plain
    `subprocess.run(timeout=)` with a long budget stalls a worker for minutes."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, errors="replace", start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        p.communicate()
        raise
    return out, err


def process(project: bc.Project, keep_work: bool = False,
            exclude_addons: bool = False) -> dict:
    """Reconstruct, import, scan one project. Returns rows + counters."""
    name = re.sub(r"[^\w.-]", "_", project.name)
    work = WORK / name
    shutil.rmtree(work, ignore_errors=True)
    rows, counts = [], dict(projects=1, files=0, clean=0, context=0, broken=0,
                            too_small=0, too_big=0)
    try:
        work.mkdir(parents=True)
        (work / "project.godot").write_text(
            'config_version=5\n\n[application]\n'
            f'config/name="{project.name}"\n'
            'config/features=PackedStringArray("4.7")\n')
        for rel, (lang, code) in project.files.items():
            if lang not in ("gdscript", "shader", "cfg", "json"):
                continue
            dst = work / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(code)
            except OSError:
                pass
        shutil.copy(HERE / SCAN, work / SCAN)
        # --recovery-mode disables tool scripts / editor plugins / GDExtension
        # during the import pass, which is what makes scraped projects hang.
        run([GODOT, "--headless", "--path", str(work), "--recovery-mode",
             "--import"], IMPORT_TIMEOUT)
        out, err = run([GODOT, "--headless", "--path", str(work),
                        "--script", f"res://{SCAN}"], SCAN_TIMEOUT)
        perr = errors_by_file(err)
        for line in out.splitlines():
            if line.startswith("OK "):
                rel, clean = line[3:], True
            elif line.startswith("FAIL "):
                rel, clean = line.split(" ", 2)[2], False
            else:
                continue
            rel = rel.strip()
            if not rel.startswith("res://") or rel.endswith(SCAN):
                continue
            counts["files"] += 1
            msgs = perr.get(rel, [])
            if clean:
                verdict = "clean"
            elif msgs and all(CONTEXT_MSG.search(m) for m in msgs) \
                    and not any(BROKEN_MSG.search(m) for m in msgs):
                verdict = "context"
            elif not msgs:
                verdict = "broken"          # FAIL with no captured reason
            else:
                verdict = "broken"
            counts[verdict] += 1
            if verdict == "broken":
                continue
            code = project.files.get(rel[len("res://"):], (None, None))[1]
            if code is None:
                continue
            if exclude_addons and rel.startswith("res://addons/"):
                counts["addon_skipped"] = counts.get("addon_skipped", 0) + 1
                continue
            n = len(code)
            if n < MIN_CHARS:
                counts["too_small"] += 1
                continue
            if n > MAX_CHARS:
                counts["too_big"] += 1
                continue
            rows.append(dict(id=f"{project.name}::{rel[len('res://'):]}",
                             project=project.name, path=rel[len("res://"):],
                             verdict=verdict, chars=n, errors=msgs[:4], code=code))
    except subprocess.TimeoutExpired as e:
        counts["timeout"] = f"{type(e).__name__}: {e}"
    except Exception as e:                                        # noqa: BLE001
        counts["error"] = f"{type(e).__name__}: {e}"
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    return dict(project=project.name, rows=rows, counts=counts)


# ---------------------------------------------------------------- spec ------
_EXTENDS = re.compile(r"^extends\s+(.+?)\s*$", re.M)
_CLASSNAME = re.compile(r"^class_name\s+(\w+)", re.M)
_DOC = re.compile(r"^##\s?(.*)$", re.M)
_FUNC = re.compile(r"^\s*(?:static\s+)?func\s+(.+?):\s*$", re.M)
_SIGNAL = re.compile(r"^\s*signal\s+(\w+)", re.M)
_EXPORT = re.compile(r"^\s*@export\w*\s+var\s+(\w+)", re.M)


def spec(code: str, path: str) -> str:
    """A spec->code prompt derived from the file itself: base class, doc
    comments, and the interface it declares. The bodies remain the target.

    The wording deliberately mirrors how the model is *evaluated* ("a complete
    script", "no top-level statements", "start with extends"). An earlier
    version only said "write the file <path>", and the adapter duly produced
    fragments with top-level statements — right Godot 4 API, wrong shape.
    """
    lines = [f"Write the complete Godot 4 GDScript for `res://{path}` — a "
             "self-contained script with no top-level statements."]
    m = _EXTENDS.search(code)
    if m:
        lines.append(f"It extends `{m.group(1).strip()}`.")
    if (m := _CLASSNAME.search(code)):
        lines.append(f"Its class_name is `{m.group(1)}`.")
    doc = [d.strip() for d in _DOC.findall(code) if d.strip()][:6]
    if doc:
        lines.append("Documented purpose: " + " ".join(doc)[:400])
    sig = [s.strip() for s in _SIGNAL.findall(code)][:12]
    if sig:
        lines.append("Signals: " + ", ".join(sig))
    exp = [v.strip() for v in _EXPORT.findall(code)][:12]
    if exp:
        lines.append("Exported variables: " + ", ".join(exp))
    fn = [f.strip() for f in _FUNC.findall(code)][:24]
    if fn:
        lines.append("Methods to implement: " + "; ".join(fn))
    lines.append("Return only the file content, starting with `extends`. "
                 "Do not add example usage or top-level statements.")
    return "\n".join(lines)


SYSTEM = ("You are an expert Godot 4 game developer. You write correct, "
          "idiomatic GDScript 4 — never Godot 3 APIs. Reply with code only "
          "unless asked to explain.")


def to_sample(row: dict) -> dict:
    return dict(id=row["id"], project=row["project"], path=row["path"],
                verdict=row["verdict"], chars=row["chars"],
                messages=[dict(role="system", content=SYSTEM),
                          dict(role="user", content=spec(row["code"], row["path"])),
                          dict(role="assistant", content=row["code"])])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/godot4_sft.jsonl")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--format", choices=["instruct", "raw", "both"], default="instruct")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--exclude-addons", action="store_true",
                    help="drop addons/ (vendored third-party libraries)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep files whose content is byte-identical to an earlier one")
    ap.add_argument("--min-version", type=int, default=4)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    a = ap.parse_args()

    projects = [p for p in bc.load_all() if bc.major(p.version) == str(a.min_version)]
    if a.shuffle_seed:
        import random
        random.Random(a.shuffle_seed).shuffle(projects)
    if a.limit:
        projects = projects[:a.limit]
    out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{len(projects)} Godot-{a.min_version} projects | workers {a.workers}"
          f" | format {a.format}")
    t0 = time.time()
    tot = dict(projects=0, files=0, clean=0, context=0, broken=0,
               too_small=0, too_big=0, kept=0, chars=0, dupes=0, addon_skipped=0,
               timeouts=0)
    fail = 0
    seen: set = set()          # exact-content dedup across the whole corpus:
                               # the same addon ships in dozens of project dumps
    with open(out, "w") as fo, ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, p, a.keep_work, a.exclude_addons): p.name
                for p in projects}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:                                # noqa: BLE001
                fail += 1
                print(f"  ! {futs[f]}: {type(e).__name__}: {e}", flush=True)
                continue
            c = r["counts"]
            for k in ("files", "clean", "context", "broken", "too_small",
                      "too_big", "addon_skipped"):
                tot[k] += c.get(k, 0)
            tot["projects"] += 1
            if "error" in c or "timeout" in c:
                fail += 1
                tot["timeouts"] += 1 if "timeout" in c else 0
                print(f"  ! {r['project']}: {c.get('error') or c.get('timeout')}",
                      flush=True)
            for row in r["rows"]:
                if not a.no_dedup:
                    h = hashlib.sha1(row["code"].encode()).digest()
                    if h in seen:
                        tot["dupes"] += 1
                        continue
                    seen.add(h)
                if a.format == "instruct":
                    rec = to_sample(row)
                elif a.format == "raw":
                    rec = dict(id=row["id"], project=row["project"],
                               path=row["path"], verdict=row["verdict"],
                               chars=row["chars"], code=row["code"])
                else:
                    rec = dict(to_sample(row), code=row["code"],
                               spec=spec(row["code"], row["path"]))
                fo.write(json.dumps(rec) + "\n")
                tot["kept"] += 1
                tot["chars"] += row["chars"]
            if i % 25 == 0 or i == len(projects):
                dt = time.time() - t0
                print(f"  [{i}/{len(projects)}] {dt:5.0f}s "
                      f"({dt/i:.2f}s/proj) | kept {tot['kept']:,} | "
                      f"clean {tot['clean']:,} ctx {tot['context']:,} "
                      f"dup {tot['dupes']:,} | err {fail}", flush=True)

    dt = time.time() - t0
    tot["seconds"] = round(dt)
    tot["out"] = str(out)
    tot["model_tokens_est"] = int(tot["chars"] / 3.85)
    tot["extrapolated_full_pass_s"] = int(dt / max(tot["projects"], 1) * 3231)
    print(json.dumps(tot, indent=2))
    (out.with_suffix(".stats.json")).write_text(json.dumps(tot, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
