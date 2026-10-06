#!/usr/bin/env python3
"""K9-in-KoboldCpp gate: K9 sentinel GGUF vs materialized Q8_0, same binary.

KoboldCpp vendors the whole llama.cpp loader, so with the K9 port applied
(branch `k9`, patching the vendored ggml + llama-model-loader exactly like
the llama.cpp fork does) a K9 GGUF must decode into the same Q8_0/F16
weights. The materialized reference file (materialize_k9_gguf.py) is the
byte-exact stock-loader counterpart. With deterministic greedy generation
(temperature 0, top_k 1, fixed seed), the two servers must produce
token-identical output; any bit-level difference in the decoded weights
shows up as a greedy-token divergence.

Runs each model in its own koboldcpp server process (CPU build), sequentially.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def http_post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_ready(port: int, deadline: float) -> None:
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/extra/version", timeout=2) as r:
                info = json.loads(r.read().decode())
                # /api/extra/version returns {"result": "KoboldCpp", "version": ..., "llm": true}
                if info.get("llm") is True:
                    return
        except Exception as e:
            print(f"  waiting ({e.__class__.__name__}): port {port}", flush=True)
            time.sleep(2.0)
    raise SystemExit("koboldcpp did not come up")


def start_kcpp(kcpp_dir: Path, model: Path, port: int, threads: int, ctx: int,
               env_base: dict, name: str = "") -> subprocess.Popen:
    env = dict(env_base)
    env.pop("DISPLAY", None)  # suppress webbrowser/xdg-open in LaunchWebbrowser
    model = str(model.resolve())  # resolve before the cwd switch below
    log_path = Path(f"/tmp/kcpp_arm_{name}_server.log")
    log_path.unlink(missing_ok=True)
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [sys.executable, str(kcpp_dir / "koboldcpp.py"),
         "--model", str(model),
         "--usecpu", "--quiet",
         "--threads", str(threads), "--blasthreads", str(threads),
         "--host", "127.0.0.1", "--port", str(port),
         "--contextsize", str(ctx)],
        cwd=str(kcpp_dir), env=env,
        stdout=log, stderr=subprocess.STDOUT)
    proc.log_file = log
    try:
        wait_ready(port, time.time() + 600)
    except SystemExit:
        proc.terminate()
        raise
    except Exception as e:
        proc.terminate()
        raise e
    return proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcpp-dir", type=Path, default=Path("/home/rose/Work/koboldcpp"))
    ap.add_argument("--k9-gguf", type=lambda p: Path(p).resolve(), required=True)
    ap.add_argument("--ref-gguf", type=lambda p: Path(p).resolve(), required=True,
                    help="materialized-from-K9 q8_0 GGUF (same weights, stock loader)")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-tokens", type=int, default=32)
    args = ap.parse_args()

    env_base = os.environ.copy()
    prompts = [
        "The capital city of",
        "Once upon a time, the",
        "Godot 4 has a method",
        "Quantization works by",
        "def add(a, b): return",
        "The three primary colors",
    ]
    arms = [("k9", args.k9_gguf), ("stock", args.ref_gguf)]
    out = {}

    for name, model in arms:
        print(f"=== arm '{name}': {model.name} on port {args.port}", flush=True)
        proc = start_kcpp(args.kcpp_dir, model, args.port, args.threads, args.ctx,
                          env_base, name=name)
        try:
            gens = []
            for p in prompts:
                resp = http_post(f"http://127.0.0.1:{args.port}/api/v1/generate", {
                    "prompt": p, "max_length": args.n_tokens,
                    "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                    "rep_pen": 1.0, "seed": 42, "trim_stop": False,
                })
                gens.append(resp["results"][0]["text"])
            out[name] = gens
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"    sample: {gens[0]!r}")

    exact = out["k9"] == out["stock"]
    for p, a, b in zip(prompts, out["k9"], out["stock"]):
        mark = "==" if a == b else "!="
        print(f"  [{mark}] {p!r}: k9={a!r} | stock={b!r}")
    print("\nGATE:", "PASS" if exact else "FAIL")
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()