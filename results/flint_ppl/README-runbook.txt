# Flint K9 build + A/B — run book (2026-10-27)

Machine: A9 Mega (Strix-Halo gfx1151 iGPU, 96 GiB pool; 30 GB sys RAM; 32 threads).
Ollama sits on :11434 (ornith-1.5:122b — before serving: `curl -s localhost:11434/api/generate -d '{"model":"ornith-1.5:122b","keep_alive":0,"prompt":""}'` to release ~77 GB; she reloads on next use).

Artifacts (this run):
- ref:   /home/rose/flint_ref/Flint-BF16.gguf  (ggml-org, sha256 bddf310fc9f33ad394674a6d94f1ba3fe4d83c0ad57bb69f72cc0d443146ff2e)
- K9:    /mnt/Matrix/Work/base9-quantization/results/flint_k63_e99_v2.gguf  (~28.7 GB)
- iq4:   .../flint_iq4nl.gguf (~21.6 GB, k99 heads stay exact)
- baselines: ggml-org Flint-Q8_0.gguf (8a257b63…), Flint-Q4_K_M.gguf (c3c4d93f…)

Serve (plain process, port 18460):
```
SRV=/home/rose/Work/llama.cpp/build-hip-a9/bin/llama-server
nohup $SRV -m <gguf> --port 18460 -ngl 99 -c 8192 --no-mmap \
  --chat-template-file /home/rose/flint_ref/chat_template.jinja \
  -t 6 --threads-batch 16 > results/flint_srv_<tag>.log 2>&1 &
```
Kill per arm: `pkill -f "llama-server.*18460"`.

Battery: experiments/ab_glm45air.py --port 18460 --tag flint_<arm>  (same GDScript battery).
PPL: build-hip-a9/bin/llama-perplexity -m <gguf> -f results/wiki2_raw_test.txt -ngl 99 -c 2048 -t 16 → parse "Final estimate: PPL =".

Build sequence:
1. pass1: `glm_k9_build.py pass1 --ref Flint-BF16.gguf --spill results/flint_k9_spill` (RAM-safe streaming; est 30-60 min)
2. pass2 (no --mini) → K9 GGUF; delete spill when done.
3. gates: cmd_gate on token_embd (k99), an attn, an expert tensor (materialize dst both q8_0 + iq4_nl).
4. materialize --dst iq4_nl → stock GGUF; gguf_audit.
5. A/B: ppl ×3 models + battery ×3.
6. Write results/flint_k63_smoke.md style report + commit. NO publish without rose.