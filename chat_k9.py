#!/usr/bin/env python3
"""Chat with the K9-quantized Qwen2.5-Coder models — and with the bf16 baseline.

Loads one base model, caches its pristine bf16 weights on the CPU, and streams
K9 files into the same weights on demand. Because every K9 file in `results/`
covers *all* decoder tensors (196 body + embed [+ lm_head]), switching between
two K9 variants needs no restore at all — only switching back to bf16 does.

    python3 chat_k9.py                     # 1.5B: bf16 baseline vs k63/embed99
    python3 chat_k9.py --preset 7b         # 7B: bf16, k9, k15, k9+GPTQ, k15+GPTQ
    python3 chat_k9.py --preset 7b-tuned   # 7B Godot-4 tune: bf16, merged, k9, k15
    python3 chat_k9.py --preset 7b --no-serve    # same thing in the terminal
    python3 chat_k9.py --k9 results/x.k9 --base Qwen/...   # explicit

Opens a local web chat (stdlib http.server — no gradio/flask/network needed).
The variant picker is the point: same prompts, same weights' shape, different
codec settings, so the quantization effect is directly visible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "experiments"))

os.environ.setdefault("HF_HOME", str(HERE / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # noqa: E402

import k9  # noqa: E402

PRESETS = {
    "1.5b": dict(
        base="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        variants={"bf16 (unquantized)": None,
                  "k63 + embed99": "results/qwen_coder_1.5b_k63_embed99.k9"},
    ),
    "7b": dict(
        base="Qwen/Qwen2.5-Coder-7B-Instruct",
        variants={"bf16 (unquantized)": None,
                  "k9 + embed99": "results/qwen7b_k9_embed99.k9",
                  "k15 + embed99": "results/qwen7b_k15_embed99.k9",
                  "k9 + GPTQ": "results/qwen7b_gptq_k9_embed99.k9",
                  "k15 + GPTQ": "results/qwen7b_gptq_k15_embed99.k9"},
    ),
    "7b-tuned": dict(
        base="Qwen/Qwen2.5-Coder-7B-Instruct",
        variants={
            "bf16 (unquantized)": None,
            "tuned bf16 (merged)": "phase2/out/merged_7b",
            "tuned k9 + embed99": "results/qwen7b_tuned_k9_embed99.k9",
            "tuned k15 + embed99": "results/qwen7b_tuned_k15_embed99.k9",
            "tuned k63 + embed99": "results/qwen7b_tuned_k63_embed99.k9",
        },
    ),
}
DEFAULTS = dict(temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=512)
MAX_CTX = 30000


# --------------------------------------------------------------------- info --
def describe(path: Path, n_params: int) -> dict:
    """Codec summary straight from the container's directory."""
    with k9.K9File(str(path)) as f:
        ks: dict = {}
        for m in f.meta:
            ks[m["k"]] = ks.get(m["k"], 0) + 1
    size = path.stat().st_size
    body_k = max(ks, key=lambda k: ks[k])
    return dict(bytes=size, mb=round(size / 1e6, 1),
                bits_per_param=round(8 * size / n_params, 3),
                tensors=sum(ks.values()),
                k_hist={str(k): v for k, v in sorted(ks.items())},
                summary=f"{size/1e6:.0f} MB · {8*size/n_params:.2f} b/param · "
                        f"body k={body_k} (g64) · " +
                        ", ".join(f"{v} tensors at k={k}" for k, v in sorted(ks.items())))


def describe_path(p: Path, n_params: int) -> dict:
    """Codec summary for a variant: a K9 container, or a merged model dir."""
    if p.is_dir():
        tot = sum(f.stat().st_size for f in p.glob("*.safetensors"))
        return dict(bytes=tot, mb=round(tot / 1e6, 1),
                    bits_per_param=round(8 * tot / n_params, 3), tensors=None,
                    k_hist={},
                    summary=f"merged bf16 · {tot/1e6:.0f} MB · "
                            f"{8*tot/n_params:.2f} b/param (fine-tuned)")
    return describe(p, n_params)


class Engine:
    """One base model in VRAM; K9 files stream into it; bf16 restores from a
    CPU cache of the pristine weights."""

    def __init__(self, base: str, variants: dict, device: str, row_chunk: int = 4096,
                 cache_pristine: bool = True, verbose: bool = True):
        self.device = device
        self.row_chunk = row_chunk
        self.lock = threading.Lock()          # one GPU generation at a time
        self.variants = variants
        self.base_id = base
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation="sdpa").to(device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.info = {name: (None if p is None
                            else describe_path(Path(p), self.n_params))
                     for name, p in variants.items()}
        self.pristine = {}
        if cache_pristine and None in variants.values() \
                and any(p is not None for p in variants.values()):
            # Only worth caching if a K9 variant exists to switch away from.
            if verbose:
                print(f"caching pristine bf16 weights on CPU "
                      f"({self.n_params*2/1e9:.1f} GB)...", flush=True)
            with torch.no_grad():
                for _, mod in self._weights():
                    self.pristine[self._key(mod)] = \
                        mod.weight.data.detach().to("cpu", copy=True)
        self.active = self._variant_with_none() or next(iter(variants))
        if verbose:
            print(f"model ready in {time.time()-t0:.0f}s | {self.n_params/1e9:.2f} B params"
                  f" | VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB"
                  if device == "cuda" else f"model ready in {time.time()-t0:.0f}s", flush=True)

    # -- module mapping: K9 names are module paths, plus __embed__/__lm_head__ --
    def _weights(self):
        yield "__embed__", self.model.model.embed_tokens
        if not self.model.config.tie_word_embeddings:
            yield "__lm_head__", self.model.lm_head

    @staticmethod
    def _key(mod) -> str:
        return "weight_" + str(id(mod))

    def _load_bf16(self):
        with torch.no_grad():
            for _, mod in self._weights():
                master = self.pristine[self._key(mod)]
                W = mod.weight.data
                for a in range(0, W.shape[0], 1024):
                    b = min(W.shape[0], a + 1024)
                    W[a:b].copy_(master[a:b].to(self.device, non_blocking=True))
        self._free()

    def _load_k9(self, path: Path, progress=None):
        with k9.K9File(str(path)) as f:
            names = set(f.names())
            for i, (name, rec) in enumerate(f.iter_tensors()):
                if name == "__embed__":
                    mod = self.model.model.embed_tokens
                elif name == "__lm_head__":
                    mod = self.model.lm_head
                else:
                    mod = self.model.get_submodule(name)
                k9.load_into(mod, rec, rec["group"], device=self.device,
                             row_chunk=self.row_chunk)
                del rec
                if progress and i % 24 == 0:
                    progress(name, i, len(names))
        self._free()

    @staticmethod
    def _free():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_dir(self, path: Path):
        """Load weights from a saved HF model dir (e.g. a QLoRA adapter merged
        into bf16 by merge_adapter.py). State-dict keys are module paths, so the
        same get_submodule mapping as the K9 path works."""
        from safetensors.torch import load_file
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors in {path}")
        sd = {}
        for f in files:
            sd.update(load_file(str(f), device="cpu"))
        with torch.no_grad():
            for key, tensor in sd.items():
                if key.endswith(".weight") or key.endswith(".bias"):
                    parent, attr = key.rsplit(".", 1)
                else:
                    continue
                try:
                    mod = self.model.get_submodule(parent)
                except AttributeError:
                    continue
                cur = getattr(mod, attr, None)
                if cur is None or tuple(cur.shape) != tuple(tensor.shape):
                    continue
                for a in range(0, tensor.shape[0], 1024):
                    b = min(tensor.shape[0], a + 1024)
                    cur[a:b].copy_(tensor[a:b].to(self.device, non_blocking=True))
                del tensor
        self._free()

    def _variant_with_none(self):
        return next((n for n, p in self.variants.items() if p is None), None)

    def use(self, name: str, progress=None) -> float:
        if name not in self.variants:
            raise KeyError(f"unknown variant {name!r}")
        with self.lock:
            t0 = time.time()
            p = self.variants[name]
            if p is None:
                self._load_bf16()
            elif Path(p).is_dir():
                self._load_dir(Path(p))
            else:
                self._load_k9(Path(p), progress)
            self.active = name
            return time.time() - t0

    # -- generation -----------------------------------------------------------
    def _trim(self, msgs: list) -> list:
        """Drop the oldest turns until the rendered prompt fits MAX_CTX. Keeps a
        leading system message and never drops the final user turn."""
        def total(ms):
            return sum(len(self.tok.apply_chat_template([m], tokenize=True,
                                                        add_generation_prompt=False))
                       for m in ms)
        while len(msgs) > 1 and total(msgs) > MAX_CTX:
            i = 1 if msgs[0]["role"] == "system" else 0
            if i >= len(msgs) - 1:
                break
            msgs = msgs[:i] + msgs[i + 1:]
        return msgs

    @torch.no_grad()
    def stream(self, msgs: list, **kw):
        """Yield (delta_text, None) while generating, then (None, stats)."""
        gen = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
        msgs = self._trim(list(msgs))
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True,
                                        skip_special_tokens=True)
        greedy = gen["temperature"] in (0, 0.0)
        kwg = dict(**enc, streamer=streamer, max_new_tokens=gen["max_new_tokens"])
        if greedy:
            kwg.update(do_sample=False)
        else:
            kwg.update(do_sample=True, temperature=gen["temperature"],
                       top_p=gen["top_p"], top_k=gen["top_k"])
        with self.lock:
            th = threading.Thread(target=self.model.generate, kwargs=kwg, daemon=True)
            t0 = time.time()
            th.start()
            n = 0
            for piece in streamer:
                n += 1
                yield piece, None
            th.join()
            dt = time.time() - t0
        yield None, dict(prompt_tokens=int(enc["input_ids"].shape[1]),
                         reply_tokens=n, seconds=round(dt, 2),
                         tok_per_s=round(n / dt, 1) if dt else 0.0,
                         variant=self.active)


# -------------------------------------------------------------------- web ----
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>K9 chat</title><style>
:root{--bg:#0f1115;--panel:#171a21;--line:#272b34;--fg:#e6e8ee;--dim:#8b93a3;
--accent:#6ea8fe;--user:#1e2a3d;--bot:#1a1d24}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
background:var(--bg);color:var(--fg);display:flex;flex-direction:column;height:100vh}
header{display:flex;gap:14px;align-items:center;padding:10px 16px;background:var(--panel);
border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.3px}
select,button,input,textarea{font:inherit;color:var(--fg);background:#12151b;
border:1px solid var(--line);border-radius:7px;padding:6px 9px}
button{cursor:pointer}button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:#08101f;border-color:var(--accent);font-weight:600}
.spacer{flex:1}
#stats{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
#statbar{padding:6px 16px;background:#12151b;border-bottom:1px solid var(--line);
color:var(--dim);font-size:12px;display:flex;gap:16px;flex-wrap:wrap}
#log{flex:1;overflow-y:auto;padding:18px 16px;display:flex;flex-direction:column;gap:14px}
.row{display:flex;gap:10px;max-width:900px;width:100%}
.row.me{align-self:flex-end;flex-direction:row-reverse}
.who{width:56px;flex:none;color:var(--dim);font-size:11px;padding-top:5px;text-align:right}
.row.bot .who{text-align:left}
.bub{background:var(--bot);border:1px solid var(--line);border-radius:11px;padding:10px 13px;
white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;flex:1}
.row.me .bub{background:var(--user)}
.bub pre{background:#0b0d12;border:1px solid var(--line);border-radius:8px;padding:10px;
overflow-x:auto;margin:8px 0}
.bub code{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.bub p{margin:0 0 8px}.bub p:last-child{margin:0}
.meta{color:var(--dim);font-size:11px;margin-top:7px}
footer{padding:12px 16px;background:var(--panel);border-top:1px solid var(--line);
display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;resize:none;height:52px;max-height:200px}
details{position:relative}summary{list-style:none;cursor:pointer;color:var(--dim);padding:6px 9px;
border:1px solid var(--line);border-radius:7px;font-size:12px}
details>div{position:absolute;bottom:44px;right:0;background:var(--panel);border:1px solid var(--line);
border-radius:9px;padding:12px;width:250px;display:flex;flex-direction:column;gap:9px}
details label{display:flex;justify-content:space-between;gap:8px;color:var(--dim);font-size:12px}
details input[type=range]{width:120px}
</style></head><body>
<header>
  <h1>K9 chat</h1>
  <select id="variant"></select>
  <span id="stats"></span>
  <span class="spacer"></span>
  <button id="clear">Clear</button>
</header>
<div id="statbar"></div>
<div id="log"></div>
<footer>
  <textarea id="in" placeholder="Ask for GDScript, or anything else…  (Enter to send, Shift+Enter for newline)"></textarea>
  <details><summary>params</summary><div>
    <label>temperature <input id="temp" type="range" min="0" max="1.5" step="0.05" value="0.7"><span id="tempv">0.7</span></label>
    <label>top_p <input id="topp" type="range" min="0.1" max="1" step="0.05" value="0.8"><span id="toppv">0.8</span></label>
    <label>max tokens <input id="maxt" type="range" min="64" max="2048" step="64" value="512"><span id="maxtv">512</span></label>
    <label style="color:var(--fg)"><input id="greedy" type="checkbox"> greedy (temp 0)</label>
  </div></details>
  <button class="primary" id="send">Send</button>
</footer>
<script>
const log=document.getElementById('log'), inp=document.getElementById('in');
const sel=document.getElementById('variant'), stats=document.getElementById('stats');
const statbar=document.getElementById('statbar');
let msgs=[], info={}, busy=false;
const fmt=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
  .replace(/```(\w*)\n([\s\S]*?)```/g,(m,l,c)=>`<pre><code>${c.replace(/\n$/,'')}</code></pre>`)
  .replace(/`([^`\n]+)`/g,'<code>$1</code>');
function add(role,text,meta){const r=document.createElement('div');r.className='row '+(role==='user'?'me':'bot');
  r.innerHTML=`<div class="who">${role==='user'?'you':'model'}</div><div class="bub">${fmt(text)}${meta?`<div class="meta">${meta}</div>`:''}</div>`;
  log.appendChild(r);log.scrollTop=log.scrollHeight;return r.querySelector('.bub');}
function setStat(o){stats.textContent = o.active ? `${o.active} · ${o.tok_per_s} tok/s · ${o.reply_tokens} tok` : '';
  statbar.textContent = o.summary || '';}
fetch('/state').then(r=>r.json()).then(s=>{
  info=s.info;
  // s.variants is a LIST of names; s.info is the name -> codec summary mapping.
  for(const n of s.variants){const o=document.createElement('option');
    o.value=n;o.textContent=n;sel.appendChild(o);}
  // set selection on the <select>, not per-option: an option's `selected` set
  // before/while it is appended gets dropped and the first option wins.
  sel.value=s.active;
  setStat({active:s.active,tok_per_s:0,reply_tokens:0,summary:info[s.active]?.summary||''});
  sel.disabled=false;});
sel.disabled=true;
sel.onchange=async()=>{const v=sel.value;stats.textContent='loading '+v+' …';statbar.textContent='';
  sel.disabled=true;
  const r=await fetch('/variant',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:v})}).then(r=>r.json());
  sel.disabled=false;
  if(r.error){stats.textContent='ERROR: '+r.error;sel.value=r.active;}else{
    info=r.info; stats.textContent='';msgs=[];log.innerHTML='';
    setStat({active:v,tok_per_s:0,reply_tokens:0,summary:info[v]?.summary||''});
    add('bot',`— switched to ${v} in ${r.seconds.toFixed(0)}s (history cleared) —`);}};
async function send(){
  const text=inp.value.trim(); if(!text||busy) return; busy=true; inp.value=''; sel.disabled=true;
  add('user',text); msgs.push({role:'user',content:text});
  const bub=add('bot',''); const g=document.getElementById('greedy').checked;
  const body={messages:msgs, temperature:g?0:parseFloat(temp.value), top_p:parseFloat(topp.value),
    max_new_tokens:parseInt(maxt.value)};
  let acc='';
  try{
    const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    const rd=res.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){const {value,done}=await rd.read(); if(done)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\n\n'); buf=parts.pop();
      for(const p of parts){ if(!p.startsWith('data: '))continue;
        const ev=JSON.parse(p.slice(6));
        if(ev.error){bub.innerHTML=fmt('**error:** '+ev.error);}
        else if(ev.t!==undefined){acc+=ev.t;bub.innerHTML=fmt(acc);log.scrollTop=log.scrollHeight;}
        else if(ev.done){msgs.push({role:'assistant',content:acc});
          bub.innerHTML=fmt(acc)+`<div class="meta">${ev.stats.reply_tokens} tok · ${ev.stats.tok_per_s} tok/s · ${ev.stats.seconds}s</div>`;
          // ev.stats carries `variant`; setStat reads `active`.
          setStat({active:ev.stats.variant,tok_per_s:ev.stats.tok_per_s,
                   reply_tokens:ev.stats.reply_tokens,
                   summary:info[ev.stats.variant]?.summary||''});}
      }}
  }catch(e){bub.innerHTML=fmt('**error:** '+e);}
  busy=false; sel.disabled=false; inp.focus();
}
document.getElementById('send').onclick=send;
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
document.getElementById('clear').onclick=()=>{msgs=[];log.innerHTML='';};
for(const id of ['temp','topp','maxt'])
  document.getElementById(id).oninput=e=>document.getElementById(id+'v').textContent=e.target.value;
inp.focus();
</script></body></html>"""


def make_handler(eng: Engine, state: dict):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                       # keep the console readable
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._json(200, dict(active=eng.active, variants=list(eng.variants),
                                     base=eng.base_id, n_params=eng.n_params,
                                     info=eng.info))
            else:
                self._json(404, dict(error="not found"))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:                                    # noqa: BLE001
                return self._json(400, dict(error=f"bad json: {e}"))
            if self.path == "/variant":
                name = req.get("name")
                try:
                    print(f"  switching → {name}", flush=True)

                    def prog(nm, i, tot):
                        state["prog"] = f"{nm} ({i}/{tot})"
                    secs = eng.use(name, progress=prog)
                except Exception as e:                                # noqa: BLE001
                    return self._json(200, dict(error=str(e), active=eng.active))
                print(f"  → {name} ready in {secs:.0f}s", flush=True)
                self._json(200, dict(active=name, seconds=secs, info=eng.info))
            elif self.path == "/chat":
                msgs = req.get("messages") or []
                if not msgs:
                    return self._json(400, dict(error="no messages"))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                # No Content-Length and no chunking: the browser's reader ends
                # only when the socket closes, so make sure it does.
                self.close_connection = True
                try:
                    for piece, stats in eng.stream(
                            msgs, temperature=req.get("temperature"),
                            top_p=req.get("top_p"),
                            max_new_tokens=req.get("max_new_tokens")):
                        ev = dict(done=True, stats=stats) if piece is None else dict(t=piece)
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:                                # noqa: BLE001
                    try:
                        self.wfile.write(f"data: {json.dumps(dict(error=str(e)))}\n\n".encode())
                        self.wfile.flush()
                    except Exception:
                        pass
            else:
                self._json(404, dict(error="not found"))

    return H


def serve(eng: Engine, port: int, host: str = "127.0.0.1"):
    from http.server import ThreadingHTTPServer
    state: dict = {}
    srv = ThreadingHTTPServer((host, port), make_handler(eng, state))
    url = f"http://{host}:{port}"
    print(f"\n  chat ready → {url}\n  (ctrl-c to stop)\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


# -------------------------------------------------------------------- repl ---
def repl(eng: Engine):
    print("\nvariant:", eng.active, "| /use <name>, /reset, /q\n")
    msgs: list = []
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/q", "/quit", "/exit"):
            break
        if line == "/reset":
            msgs = []
            print("(history cleared)")
            continue
        if line.startswith("/use "):
            name = line[5:].strip()
            cands = [n for n in eng.variants if name.lower() in n.lower()]
            if len(cands) != 1:
                print(f"  match {cands or 'nothing'} — pick one")
                continue
            t = eng.use(cands[0])
            msgs = []
            print(f"(switched to {cands[0]} in {t:.0f}s, history cleared)")
            continue
        if line == "/use" or line == "/variants":
            for n in eng.variants:
                print(f"  {'*' if n == eng.active else ' '} {n}")
            continue
        msgs.append({"role": "user", "content": line})
        print("model> ", end="", flush=True)
        acc = ""
        stats = None
        for piece, st in eng.stream(msgs):
            if piece is None:
                stats = st
            else:
                acc += piece
                print(piece, end="", flush=True)
        msgs.append({"role": "assistant", "content": acc})
        if stats:
            print(f"\n  [{stats['reply_tokens']} tok · {stats['tok_per_s']} tok/s"
                  f" · {stats['seconds']}s · {eng.active}]\n")


# -------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="1.5b")
    ap.add_argument("--base", help="HF model id (overrides --preset)")
    ap.add_argument("--k9", action="append", default=[], metavar="PATH",
                    help="add a K9 variant (repeatable); label defaults to the filename")
    ap.add_argument("--tuned", action="append", default=[], metavar="DIR",
                    help="add a merged/adapted model dir as a variant (repeatable)")
    ap.add_argument("--variant", help="variant to activate at startup")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--row-chunk", type=int, default=4096)
    ap.add_argument("--no-serve", action="store_true", help="terminal REPL instead of the web UI")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()

    pre = PRESETS[a.preset]
    base = a.base or pre["base"]
    # An explicit --base means the preset's K9 files (which belong to a
    # different base) must not be carried over. Preset paths are anchored to
    # the repo root no matter which directory the server is launched from.
    variants = {"bf16 (unquantized)": None} if a.base else {
        n: None if p is None else str(HERE / p)
        for n, p in pre["variants"].items()}
    for p in a.k9:
        name = Path(p).name.replace(".k9", "")
        variants[name] = str(Path(p) if Path(p).is_absolute() else HERE / p)
    for d in a.tuned:
        dd = Path(d) if Path(d).is_absolute() else HERE / d
        variants[f"fine-tuned ({dd.name})"] = str(dd)
    # drop variants whose file is missing
    for n, p in list(variants.items()):
        if p is not None and not Path(p).exists():
            print(f"  ! skipping {n}: {p} not found")
            del variants[n]

    print(f"base: {base}")
    for n, p in variants.items():
        print(f"  variant {n}: {p or 'bf16 (from HF weights)'}")
    eng = Engine(base, variants, a.device, row_chunk=a.row_chunk)
    # Default to a quantized variant: that is the artifact under test. Falling
    # back to bf16 is only for a --base-only run.
    start = a.variant or next((n for n, p in variants.items() if p is not None), None)
    if start:
        print(f"loading {start} ...", flush=True)
        print(f"  ready in {eng.use(start):.0f}s", flush=True)
    if not a.no_warmup:
        print("warming up (first token is always slow)...", flush=True)
        t0 = time.time()
        for _ in eng.stream([{"role": "user", "content": "hi"}], max_new_tokens=1):
            pass
        print(f"  warm in {time.time()-t0:.1f}s", flush=True)

    if a.no_serve:
        repl(eng)
    else:
        serve(eng, a.port, a.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
