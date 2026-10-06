#!/usr/bin/env python3
"""Measure how well a model writes self-contained C++ — mechanically.

The C++ sibling of eval_gdscript.py, with one upgrade the toolchain allows:
tasks marked `run` are judged by actually building + executing and comparing
stdout (execution-verified), not just by parsing.

Signals per task:
  parses    g++ -fsyntax-only succeeds
  builds    g++ (full build) succeeds
  output    (run-tasks only) the executed stdout matches `expect`
  api       every required identifier appears
  no-bad    no forbidden pattern (pythonisms, rust-isms, C-isms) appears

correct = parses AND builds AND (output if run) AND api AND no-bad.
Decoding: draws-1 = greedy (comparable to the GDScript ladder protocol);
draws-N = sampled at temperature/top_p/top_k with majority voting per check.

    python3 eval_cpp.py --model Qwen/Qwen2.5-Coder-7B-Instruct --tag cpp_base
    python3 eval_cpp.py --model out/merged_mix --draws 3 --tag cpp_mix
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
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch                                                            # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer            # noqa: E402

from cpp_verify import CppVerifier, OK, RUNTIME_ERROR                   # noqa: E402

SYSTEM = ("You are an expert C++ developer. You write correct, self-contained "
          "modern C++ (C++17) that compiles cleanly with g++ alone. Reply with "
          "code only.")

BAD = [
    ("python def", r"\bdef\s+\w+\s*\("),
    ("python import", r"^\s*import\s+\w+", re.M),
    ("rust fn", r"^\s*fn\s+\w+", re.M),
    ("rust let", r"\blet\s+mut\s+"),
    ("python async", r"\basync\s+def\b"),
    ("printf in modern", r"\bprintf\s*\("),           # modern-style tasks
    ("malloc in modern", r"\bmalloc\s*\(|\bfree\s*\("),
    ("TODO", r"\bTODO\b"),
]


def bad_hits(code: str) -> list:
    return [lab for lab, rx, *rest in [(x[0], x[1]) for x in BAD]
            if re.search(rx, code)]


TASKS = [
    dict(id="stdin_sum", run=True, expect="10",
         must=["while", "cin"], must_not=["fscanf"],
         prompt="Write a complete C++ program that reads integers from standard "
                "input until EOF and prints their sum. Input: 7 3 on one line "
                "produces 10."),
    dict(id="vec_sort_desc", run=True, expect="9 4 2 1",
         must=["vector", "sort"], must_not=[],
         prompt="Write a complete C++ program that puts {4, 9, 1, 2} into a "
                "std::vector, sorts it in descending order with std::sort and "
                "a lambda, then prints the elements space-separated."),
    dict(id="map_word_count", run=True, expect="apple 3 banana 1",
         must=["unordered_map", "getline"], must_not=[],
         prompt="Write a complete C++ program that reads one line of "
                "space-separated words and prints each distinct word with its "
                "count, words in input order of first appearance."),
    dict(id="raii_buffer", run=True, expect="held released",
         must=["class", "destructor", "cout"], must_not=[],
         prompt="Write a complete C++ program defining an RAII class Buffer "
                "whose constructor prints held and whose destructor prints "
                "released; create one in main and let it go out of scope."),
    dict(id="atomic_counter", run=True, expect="2000",
         must=["atomic", "thread"], must_not=[],
         prompt="Write a complete C++ program that spawns two threads, each "
                "incrementing a std::atomic<int> 1000 times, joins them, and "
                "prints the final value."),
    dict(id="fib_memo", run=True, expect="55",
         must=["map", "fib"], must_not=[],
         prompt="Write a complete C++ program with a memoized recursive fib "
                "function (use std::map<long long, long long>) and print "
                "fib(10)."),
    dict(id="template_max", run=True, expect="42",
         must=["template", "typename"], must_not=[],
         prompt="Write a complete C++ program with a variadic-template-free "
                "function template my_max(a, b) and print my_max(40, 42)."),
    dict(id="exception_class", run=True, expect="caught: bad config",
         must=["runtime_error", "try", "catch"], must_not=[],
         prompt="Write a complete C++ program defining ConfigError : "
                "std::runtime_error, throwing it with message \"bad config\" "
                "from main's try block, catching it and printing caught: "
                "<what()> on one line."),
    dict(id="optional_variant", run=True, expect="has 7 none",
         must=["optional", "variant", "holds_alternative"], must_not=[],
         prompt="Write a complete C++17 program using std::optional holding "
                "7 (print has 7), then a std::variant<int, std::string> "
                "holding int; print none if the variant does not hold "
                "std::string."),
    dict(id="mutex_queue", run=True, expect="drained 0",
         must=["queue", "mutex", "lock_guard"], must_not=[],
         prompt="Write a complete C++ program: a thread-safe queue of ints "
                "guarded by a std::mutex + std::lock_guard: push 5 values, "
                "pop-drain them, print drained 0."),
    dict(id="string_split", run=True, expect="a/b/c",
         must=["getline"], must_not=["strtok"],
         prompt="Write a complete C++ program that splits \"a,b,c\" on commas "
                "using std::getline with a std::istringstream and prints the "
                "tokens joined with '/'."),
    dict(id="unique_ptr_factory", run=True, expect="Widget 3",
         must=["unique_ptr", "make_unique"], must_not=["new ", "delete"],
         prompt="Write a complete C++ program with a Widget class (name, id), "
                "a make_widget factory returning std::unique_ptr<Widget> "
                "(std::make_unique only), and print Widget <id> for id 3."),
    dict(id="op_stream", run=True, expect="[Gear 4]",
         must=["operator<<", "friend"], must_not=[],
         prompt="Write a complete C++ program defining a Gear struct with an "
                "int id and a friend operator<< that prints [Gear <id>], then "
                "print one with id 4."),
    dict(id="static_poly", run=False,
         must=["template", "virtual"], must_not=[],
         prompt="Write a complete C++17 header-style class Shape with a "
                "virtual double area() const, a Circle : Shape overriding it, "
                "and explain in one comment line where static polymorphism "
                "would beat the virtual call."),
    dict(id="move_semantics", run=False,
         must=["&&", "noexcept", "swap"], must_not=[],
         prompt="Write a complete C++17 class LargeBuffer with a "
                "noexcept move constructor implemented via member-wise swap, "
                "and a comment stating why moving is cheaper than copying."),
    dict(id="const_correct", run=False,
         must=["const", "constexpr"], must_not=[],
         prompt="Write a complete C++17 header-style set of declarations "
                "showing constexpr array size, a const member function, and a "
                "const reference parameter — compile-clean declarations."),
]

_REXPECT_TOL = 2      # trailing newline tolerance


def extract_code(reply: str) -> str:
    m = re.search(r"```(?:cpp|c\+\+|cc|cxx)?\s*\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).strip()


def gen(model, tok, device, prompt, max_new=640, sampling=None):
    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=prompt)]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        kw = dict(max_new_tokens=max_new, do_sample=False,
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if sampling:
            kw.update(do_sample=True, temperature=sampling["temperature"],
                      top_p=sampling["top_p"], top_k=sampling["top_k"])
        out = model.generate(**enc, **kw)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def judge(v: CppVerifier, t: dict, code: str) -> tuple[dict, str]:
    low = code.lower()
    checks = dict(parses=False, builds=False, output=None,
                  api=all(x.lower() in low for x in t["must"]),
                  no_bad=not bad_hits(code))
    r = v.check_runtime(code) if t.get("run") else v.check_parse(code)
    if t.get("run"):
        parses_status = r.status in (OK, RUNTIME_ERROR)   # run implies parsed
        checks["parses"] = parses_status
        checks["builds"] = r.status != RUNTIME_ERROR
        out = (r.stdout or "").strip()
        want = t["expect"].lower()
        got = " ".join(out.split())
        checks["output"] = want in got.lower()
    else:
        checks["parses"] = r.status == OK
        checks["builds"] = r.status == OK
    hard = [k for k in t.get("must_not", []) if k.lower() in low]
    checks["no_bad"] = checks["no_bad"] and not hard
    correct = all(v for k, v in checks.items() if v is not None)
    return checks, (code if not t.get("run") else f"{code}\n/* stdout: {(r.stdout or '')[:120]} */")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--k9")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out")
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--tasks")
    ap.add_argument("--nf4", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
        with k9.K9File(a.k9) as f:
            for name, rec in f.iter_tensors():
                mod = (model.model.embed_tokens if name == "__embed__"
                       else model.lm_head if name == "__lm_head__"
                       else model.get_submodule(name))
                k9.load_into(mod, rec, rec["group"], device=dev, row_chunk=4096)
        print(f"loaded K9 {Path(a.k9).name}", flush=True)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()

    tasks = TASKS
    if a.tasks:
        want = {t.strip() for t in a.tasks.split(",")}
        tasks = [t for t in TASKS if t["id"] in want]

    sampling = None if a.draws == 1 else dict(
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    out = Path(a.out) if a.out else HERE / "out" / f"eval_{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    agg = dict(tasks=0, parses=0, builds=0, output=0, api=0, no_bad=0, correct=0,
               draw_correct=0, total_draws=0)
    t0 = time.time()
    with CppVerifier() as v, open(out, "w") as fo:
        for t in tasks:
            ds = []
            for _ in range(a.draws):
                reply = gen(model, tok, dev, t["prompt"], a.max_new, sampling)
                code = extract_code(reply)
                checks, kept_code = judge(v, t, code)
                checks["code"] = kept_code
                ds.append(checks)
            keys = [k for k in ("parses", "builds", "output", "api", "no_bad")
                    if ds[0].get(k) is not None]
            majority = {k: sum(d[k] for d in ds) * 2 > len(ds) for k in keys}
            corrects = [all(d[k] for k in keys if d.get(k) is not None) for d in ds]
            checks = {**majority, "correct": all(majority.values())}
            agg["tasks"] += 1
            for k in ("parses", "builds", "output", "api", "no_bad", "correct"):
                agg[k] += 1 if checks.get(k) else 0
            agg["draw_correct"] += sum(corrects)
            agg["total_draws"] += len(ds)
            fo.write(json.dumps(dict(tag=a.tag, id=t["id"], draws=a.draws,
                                     draw_results=[{k: v for k, v in d.items()
                                                    if k != "code"} for d in ds],
                                     code=ds[0]["code"], **checks)) + "\n")
            fo.flush()
            flag = "OK " if checks["correct"] else "~  "
            print(f"  [{flag}] {t['id']:18s} runs={sum(d['builds'] for d in ds)}"
                  f"/{a.draws} out={sum(bool(d.get('output')) for d in ds)}"
                  f"/{a.draws}", flush=True)

    n = max(agg["tasks"], 1)
    summary = dict(tag=a.tag, model=a.model, adapter=a.adapter, draws=a.draws,
                   draw_correct_pct=round(100 * agg["draw_correct"]
                                          / max(agg["total_draws"], 1), 1),
                   seconds=round(time.time() - t0),
                   **{k: agg[k] for k in ("tasks", "parses", "builds", "output",
                                          "api", "no_bad", "correct")},
                   **{f"{k}_pct": round(100 * agg[k] / n, 1) for k in
                      ("parses", "builds", "output", "api", "no_bad", "correct")})
    print("\n" + json.dumps(summary, indent=2))
    Path(str(out).replace(".jsonl", ".summary.json")).write_text(
        json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())