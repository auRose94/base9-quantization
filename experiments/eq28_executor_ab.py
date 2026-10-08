#!/usr/bin/env python3
"""eq28 — RQ13 c-x: the codec executor — the P44-vs-P50 A/B (docs/09).

Registered (docs/09, verbatim):
  P50 (specialist executor). The dedicated companion reaches >= 95% exact
      renorm-digit accuracy zero-shot on an UNSEEN frequency table — P44's
      threshold with a dedicated model; the contrast pair (P44 generalist vs
      P50 specialist) is the registered specialization experiment.

Identical question, executor differs only in identity:
  * the task — one base-9 rANS push+renorm step. Given the state x as 13
    base-9 digits (most significant first, 9^12 <= x < 9^13) and the pushed
    symbol's table frequency f as 4 base-9 digits, output the emitted
    renormalization digits (0-4 of them, most-significant-first):
        while x >= f * 9^9:  emit x % 9;  x //= 9
    The metric is the EXACT step: the emitted digit COUNT and all its digits
    correct. Diagnostics (count-only, per-digit, parse failures) are also
    reported, and the next state is recorded as a secondary, ungated column.
  * P44 — the generalist: a locally served Qwen2.5-Coder-7B-Instruct
    (llama-server, greedy, 8 in-context examples drawn from the TRAIN
    tables), answering the same prompts;
  * P50 — the specialist: a tiny transformer (2 layers, d=96) over the same
    18 input tokens, trained on 200k steps from the train tables.

Zero-shot discipline: the test cases come from UNSEEN tables, in two
regimes — (a) the registered bar: unseen tables with the training
f-value range; (b) strict: unseen tables AND a disjoint f-value range
(never-seen frequencies). Both executors see the identical 300-case A/B
subset; the specialist's full-test numbers are also reported.

CPU/GPU: the specialist trains on CUDA if present; the generalist runs on the
local llama-server (f16 Qwen7B, partial offload if the card is tight).
"""
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

RES = L.RESULTS
M9 = 6561                    # 9^4
TH9 = 9 ** 9                 # f * 9^9 is the renorm threshold scale
L9 = 9 ** 12
ALPHA = 63
N_TRAIN_TABLES, N_TEST_TABLES = 24, 8
N_TRAIN_CASES = 200_000
N_TEST_CASES = 20_000
N_AB = 300                   # the shared evaluator subset
TRAIN_POOL = (100, 800)      # per-symbol frequency ranges
TEST_POOL = (100, 800)       # regime (a): unseen tables, same range
STRICT_POOL = (801, 1500)    # regime (b): disjoint range
LLAMA_SERVER = "/mnt/matrix/Work/llama.cpp/build-cuda/bin/llama-server"
B9R = "/mnt/matrix/Work/base9-quantization/results"
LLAMA_MODEL = os.path.join(B9R, "ref_qwen7b_f16.gguf")
PORT = 8093
N_SHOT = 8


# ------------------------------------------------------------------ data ----
def quantize(p, M=M9):
    """Largest-remainder allocation to sum exactly M."""
    p = np.asarray(p, dtype=np.float64)
    f = np.maximum(1, np.floor(p * M).astype(np.int64))
    rem = M - int(f.sum())
    if rem > 0:
        frac = p * M - np.floor(p * M)
        order = np.argsort(-frac, kind="stable")
        for j in range(rem):
            f[order[j % len(order)]] += 1
    while rem < 0:
        j = int(np.argmax(f))
        if f[j] > 1:
            f[j] -= 1
            rem += 1
        else:
            break
    return f


def make_tables(n, pool, rng):
    """n 63-ary frequency tables: per-symbol values drawn realistically from
    `pool` (the artifact class's per-symbol counts are log-normal-ish)."""
    lo, hi = pool
    out = []
    for _ in range(n):
        mu = math.log(lo) + (math.log(hi) - math.log(lo)) * rng.uniform(0.45, 0.8)
        v = np.exp(rng.normal(mu, 0.55, ALPHA))
        f = np.clip(np.round(v), lo, hi).astype(np.int64)
        out.append(f)
    return out


def digitize(x, n_digits=13):
    d = []
    for _ in range(n_digits):
        d.append(int(x % 9))
        x //= 9
    return d[::-1]


def make_cases(tables, n, rng):
    """(x digits, f digits, emit count, emit digits, next-x digits)."""
    xs, fs, cs, ds, ns = [], [], [], [], []
    picks = rng.integers(0, len(tables), n)
    syms = rng.integers(0, ALPHA, n)
    for i in range(n):
        f = int(tables[picks[i]][syms[i]])
        x = L9 + int(rng.integers(0, L9))
        y, k, digs = x, 0, []
        while y >= f * TH9 and k < 8:
            digs.append(int(y % 9))
            y //= 9
            k += 1
        digs = digs[::-1]                      # most-significant first
        xs.append(digitize(x))
        fs.append(digitize(f, 4))
        cs.append(k)
        ds.append(digs + [0] * (4 - k))
        ns.append(digitize(y))
    return (np.asarray(xs), np.asarray(fs), np.asarray(cs),
            np.asarray(ds), np.asarray(ns))


# ------------------------------------------------------------ specialist ----
class Executor(nn.Module):
    """Tiny transformer over 18 digit tokens (13 x-digits, SEP, 4 f-digits);
    a [CLS] query predicts the emit count and up to 4 digits non-autoregressively."""
    def __init__(self, d=96, nhead=4, nlayers=2, ffn=256):
        super().__init__()
        self.tok = nn.Embedding(2 * 10, d)          # value + type
        self.pos = nn.Embedding(20, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, nhead, ffn, dropout=0.0,
                                           batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head_c = nn.Linear(d, 5)
        self.head_d = nn.Linear(d, 36)

    def forward(self, x, f):
        B = x.shape[0]
        xt = self.tok(x + 0)                         # type 0 = x digit
        ft = self.tok(f + 10)                        # type 1 = f digit
        sep = self.tok(torch.full((B, 1), 19, device=x.device))
        h = torch.cat([self.cls.expand(B, -1, -1), xt, sep, ft], 1)
        h = h + self.pos(torch.arange(h.shape[1], device=x.device))[None]
        h = self.enc(h)
        c = h[:, 0]
        return self.head_c(c), self.head_d(c).reshape(B, 4, 9)


def spec_eval(net, X, F, C, D, dev):
    net.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(dev)
        fb = torch.from_numpy(F).to(dev)
        lc, ld = net(xb, fb)
        pc = lc.argmax(-1).cpu().numpy()
        pd = ld.argmax(-1).cpu().numpy()
    exact = ((pc == C) & np.all(
        (pd == D) | (np.arange(4)[None, :] >= pc[:, None]), axis=1))
    cnt_ok = pc == C
    dig_ok = []
    for i in range(len(C)):
        k = int(C[i])
        m = min(k, 4)
        dig_ok.append(bool(m > 0 and np.all(pd[i, :m] == D[i, :m])))
    return dict(exact=float(exact.mean()), count_acc=float(cnt_ok.mean()),
                digits_acc=float(np.mean(dig_ok)))


def train_specialist(Xtr, Ftr, Ctr, Dtr, seed=0, steps=6000, bs=256):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    net = Executor().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, len(Ctr), bs)
        xb = torch.from_numpy(Xtr[idx]).to(dev)
        fb = torch.from_numpy(Ftr[idx]).to(dev)
        cb = torch.from_numpy(Ctr[idx]).to(dev)
        db = torch.from_numpy(Dtr[idx]).to(dev)
        lc, ld = net(xb, fb)
        loss = nn.functional.cross_entropy(lc, cb)
        pos = torch.arange(4, device=dev)[None, :]
        mask = (pos < cb[:, None]).float()
        ld9 = ld.reshape(-1, 9)
        loss = loss + (nn.functional.cross_entropy(
            ld9, db.reshape(-1), reduction="none").reshape(db.shape)
            * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"[eq28]   train step {step} loss {float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)")
    return net, dev


# ------------------------------------------------------------ generalist ----
def start_server():
    log = open("/tmp/eq28_llama.log", "wb")
    for ngl in (99, 26, 20):
        p = subprocess.Popen(
            [LLAMA_SERVER, "-m", LLAMA_MODEL, "-ngl", str(ngl), "-c", "1536",
             "--host", "127.0.0.1", "--port", str(PORT), "--temp", "0",
             "-np", "1", "-lv", "2"],
            stdout=log, stderr=log)
        up = False
        for _ in range(240):
            time.sleep(1)
            if p.poll() is not None:
                break
            try:
                with socket.create_connection(("127.0.0.1", PORT), 1):
                    up = True
                    break
            except OSError:
                continue
        if up:
            # the port binds before the model finishes loading: wait for health
            import urllib.request
            for _ in range(180):
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                        if json.loads(r.read().decode()).get("status") == "ok":
                            print(f"[eq28] llama-server up and healthy "
                                  f"(-ngl {ngl})")
                            return p
                except Exception:
                    pass
                time.sleep(1)
            print("[eq28] health never became ok")
        print(f"[eq28] llama-server attempt -ngl {ngl}: rc={p.returncode}")
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)
        log.flush()
        with open("/tmp/eq28_llama.log", "rb") as fh:
            tail = fh.read().decode(errors="replace").splitlines()[-6:]
        print("[eq28]   log tail: " + " | ".join(tail))
    raise RuntimeError("llama-server failed to start (see /tmp/eq28_llama.log)")


def ask(prompt, n_predict=12):
    import urllib.request
    body = json.dumps({"prompt": prompt, "temperature": 0.0,
                       "n_predict": n_predict, "top_k": 1,
                       "stop": ["\n", "Q:", "x ="]}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion", data=body,
        headers={"Content-Type": "application/json"})
    for att in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode())["content"]
        except Exception as e:
            if att == 5:
                raise
            time.sleep(3)


def fmt_digits(d):
    return " ".join(str(int(v)) for v in d)


RULE = ("You execute one renormalization step of a base-9 rANS encoder.\n"
        "The state x is a 13-digit base-9 number (most significant digit "
        "first), always with 9^12 <= x < 9^13. A symbol with table frequency "
        "f is pushed; the step emits digits by repeating:\n"
        "    while x >= f * 9^9:  emit (x mod 9);  x = x // 9\n"
        "List the emitted digits most-significant-first. Answer with only the "
        "digits separated by spaces, or the word none.\n\n")


def build_prompt(shot_idx, shots_data, X, F, q):
    Xs, Fs, Cs, Ds = shots_data
    p = RULE
    for i in shot_idx:
        p += (f"x = {fmt_digits(Xs[i])}, f = {fmt_digits(Fs[i])}\n"
              f"emitted: {fmt_digits(Ds[i][:int(Cs[i])]) if Cs[i] else 'none'}\n\n")
    p += (f"x = {fmt_digits(X[q])}, f = {fmt_digits(F[q])}\n"
          f"emitted:")
    return p


def parse_answer(txt):
    t = txt.strip().lower()
    if not t:
        return None
    if "none" in t.split()[:1] or t.startswith("none"):
        return []
    ds = [int(v) for v in re.findall(r"[0-8]", t)][:4]
    return ds


def gen_eval(X, F, C, D, shots, shots_data):
    ok_exact = ok_cnt = ok_dig = 0
    parsed_fail = 0
    examples = []
    for i in range(len(C)):
        txt = ask(build_prompt(shots, shots_data, X, F, i))
        got = parse_answer(txt)
        if got is None:
            parsed_fail += 1
            got = [-1]
        k = int(C[i])
        truth = [int(v) for v in D[i][:k]]
        m = min(k, 4)
        ex = (got == truth)
        ok_exact += ex
        ok_cnt += (len(got) == k)
        ok_dig += (m > 0 and len(got) >= m and got[:m] == truth[:m])
        if i < 8:
            examples.append(dict(q=i, truth=truth, got=got, raw=txt[:60]))
    n = len(C)
    return dict(exact=ok_exact / n, count_acc=ok_cnt / n,
                digits_acc=ok_dig / n, parse_failures=parsed_fail,
                n=n, examples=examples)


# ------------------------------------------------------------------ main ----
def main():
    t0 = time.time()
    rng = np.random.default_rng(7)
    tr_tables = make_tables(N_TRAIN_TABLES, TRAIN_POOL, rng)
    te_tables = make_tables(N_TEST_TABLES, TEST_POOL, rng)      # unseen tables
    st_tables = make_tables(N_TEST_TABLES, STRICT_POOL, rng)    # disjoint f range

    Xtr, Ftr, Ctr, Dtr, _ = make_cases(tr_tables, N_TRAIN_CASES, rng)
    Xte, Fte, Cte, Dte, _ = make_cases(te_tables, N_TEST_CASES, rng)
    Xst, Fst, Cst, Dst, _ = make_cases(st_tables, N_TEST_CASES, rng)
    print(f"[eq28] data: train {len(Ctr)} | unseen-table test {len(Cte)} |"
          f" strict disjoint-f test {len(Cst)} | count dist train "
          f"{np.bincount(Ctr, minlength=5).tolist()} "
          f"({time.time()-t0:.0f}s)")

    # ---- P50: the specialist
    net, dev = train_specialist(Xtr, Ftr, Ctr, Dtr)
    s_eval = {reg: spec_eval(net, X, F, C, D, dev)
              for reg, (X, F, C, D) in
              (("unseen_tables", (Xte, Fte, Cte, Dte)),
               ("strict_disjoint_f", (Xst, Fst, Cst, Dst)),
               ("train", (Xtr[:N_TEST_CASES], Ftr[:N_TEST_CASES],
                          Ctr[:N_TEST_CASES], Dtr[:N_TEST_CASES])))}
    print(f"[eq28] P50 specialist: unseen-table exact "
          f"{100*s_eval['unseen_tables']['exact']:.2f}% | strict "
          f"{100*s_eval['strict_disjoint_f']['exact']:.2f}% | train "
          f"{100*s_eval['train']['exact']:.2f}% ({time.time()-t0:.0f}s)")

    # ---- P44: the generalist (same cases, identical information)
    ab = rng.choice(len(Cte), N_AB, replace=False)
    shots = rng.choice(N_TRAIN_CASES, N_SHOT, replace=False)
    proc = start_server()
    try:
        g_eval = {}
        shots_data = (Xtr, Ftr, Ctr, Dtr)
        got_a = gen_eval(Xte[ab], Fte[ab], Cte[ab], Dte[ab], shots, shots_data)
        g_eval["unseen_tables_AB300"] = got_a
        print(f"[eq28] P44 generalist unseen_tables_AB300: exact "
              f"{100*got_a['exact']:.1f}% | count {100*got_a['count_acc']:.1f}%"
              f" | digits {100*got_a['digits_acc']:.1f}% | parse fails "
              f"{got_a['parse_failures']} ({time.time()-t0:.0f}s)")
        stab = ab[:120]
        got_s = gen_eval(Xst[stab], Fst[stab], Cst[stab], Dst[stab],
                         shots, shots_data)
        g_eval["strict_disjoint_f_AB120"] = got_s
        print(f"[eq28] P44 generalist strict_disjoint_f_AB120: exact "
              f"{100*got_s['exact']:.1f}% | count {100*got_s['count_acc']:.1f}%"
              f" | digits {100*got_s['digits_acc']:.1f}% | parse fails "
              f"{got_s['parse_failures']} ({time.time()-t0:.0f}s)")
        s_ab = {reg: spec_eval(net, X[idx], F[idx], C[idx], D[idx], dev)
                for reg, (X, F, C, D), idx in (
                    ("unseen_tables_AB300", (Xte, Fte, Cte, Dte), ab),
                    ("strict_disjoint_f_AB120", (Xst, Fst, Cst, Dst), stab))}
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    p50 = dict(prediction="P50",
               exact_unseen_tables=round(s_eval["unseen_tables"]["exact"], 4),
               exact_strict_disjoint_f=round(
                   s_eval["strict_disjoint_f"]["exact"], 4),
               exact_train=round(s_eval["train"]["exact"], 4),
               count_acc=round(s_eval["unseen_tables"]["count_acc"], 4),
               digits_acc=round(s_eval["unseen_tables"]["digits_acc"], 4),
               bar=0.95,
               verdict="PASS" if s_eval["unseen_tables"]["exact"] >= 0.95
               else "FAIL")
    p44 = dict(executor="Qwen2.5-Coder-7B-Instruct f16 (local llama-server,"
               " greedy, 8 in-context examples)",
               n_ab=N_AB, **{k: {kk: vv for kk, vv in v.items()
                                 if kk != "examples"} for k, v in g_eval.items()})
    contrast = dict(
        note="identical question, identical 300 A/B cases, executor differs "
             "only in identity: the generalist gets 8 in-context examples from"
             " the train tables; the specialist is trained on them",
        ab_exact_generalist=round(g_eval["unseen_tables_AB300"]["exact"], 4),
        ab_exact_specialist=round(s_ab["unseen_tables_AB300"]["exact"], 4),
        ab_exact_generalist_strict=round(
            g_eval["strict_disjoint_f_AB120"]["exact"], 4),
        ab_exact_specialist_strict=round(
            s_ab["strict_disjoint_f_AB120"]["exact"], 4),
        generalist_examples=g_eval["unseen_tables_AB300"]["examples"])

    out = dict(n_train_cases=N_TRAIN_CASES, n_test_cases=N_TEST_CASES,
               tables=dict(train=N_TRAIN_TABLES, unseen_test=N_TEST_TABLES,
                           strict_test=N_TEST_TABLES,
                           train_pool=TRAIN_POOL, test_pool=TEST_POOL,
                           strict_pool=STRICT_POOL),
               P50=p50, P44=p44, contrast=contrast)
    with open(os.path.join(RES, "eq28_executor_ab.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    md = ["# eq28 — RQ13 c-x: the codec executor A/B (docs/09 P44 vs P50)\n",
          "One task: a base-9 rANS push+renorm step — given x (13 base-9"
          " digits) and the symbol's frequency f (4 base-9 digits), emit the"
          " renorm digits (0-4, MSD-first). Metric = the EXACT step (count +"
          " all digits). Identical cases for both executors.\n",
          f"- **P50 ({p50['verdict']})** (bar {p50['bar']}): specialist exact"
          f" = {100*p50['exact_unseen_tables']:.2f}% on UNSEEN tables,"
          f" {100*p50['exact_strict_disjoint_f']:.2f}% on unseen tables with a"
          f" DISJOINT f-range, {100*p50['exact_train']:.2f}% on train"
          f" (count {100*p50['count_acc']:.2f}%, digits"
          f" {100*p50['digits_acc']:.2f}%).",
          "",
          f"- **P44 (generalist)**: Qwen2.5-Coder-7B-Instruct f16, greedy,"
          f" 8 in-context examples from the train tables, same {N_AB} cases:"
          f" exact {100*p44['unseen_tables_AB300']['exact']:.1f}% (count"
          f" {100*p44['unseen_tables_AB300']['count_acc']:.1f}%, digits"
          f" {100*p44['unseen_tables_AB300']['digits_acc']:.1f}%, parse fails"
          f" {p44['unseen_tables_AB300']['parse_failures']}); strict-regime"
          f" {100*p44['strict_disjoint_f_AB120']['exact']:.1f}%.",
          "",
          f"- **The contrast (same A/B cases)**: generalist"
          f" {100*contrast['ab_exact_generalist']:.1f}% vs specialist"
          f" {100*contrast['ab_exact_specialist']:.1f}% (unseen tables);"
          f" strict regime {100*contrast['ab_exact_generalist_strict']:.1f}% vs"
          f" {100*contrast['ab_exact_specialist_strict']:.1f}%.",
          "",
          "Sample A/B answers (generalist):", ""]
    for e in contrast["generalist_examples"][:6]:
        md.append(f"- truth `{e['truth']}` -> got `{e['got']}`"
                  f" | raw `{e['raw']}`")
    with open(os.path.join(RES, "eq28_executor_ab.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"[eq28] P50 {p50['verdict']} | contrast "
          f"{100*contrast['ab_exact_generalist']:.1f}% vs"
          f" {100*contrast['ab_exact_specialist']:.1f}%"
          f" ({time.time()-t0:.0f}s). artifacts: eq28_executor_ab.json/.md")


if __name__ == "__main__":
    main()