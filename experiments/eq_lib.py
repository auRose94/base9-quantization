#!/usr/bin/env python3
"""Shared library for the weights-as-equations pilot on stories260K.

Model: Llama-2-style decoder as exported by llm.c (llama2.c conventions):
  dim=64, 5 layers, 8 heads, 4 KV heads (GQA), SwiGLU FFN hidden=172,
  RMSNorm (eps 1e-5), RoPE theta=1e4 (interleaved pairs), seq len 512,
  vocab 512 (SentencePiece tok512 tied to nothing; untied embeddings).
Everything is written from scratch against the checkpoint's state-dict keys so
the whole pipeline is explicit: this repo does not import anyone's GPT.

Weights are manipulated as float64 on CPU (analysis) and float32 on GPU (eval,
synthesis). Bit accounting for "equation" representations is raw and
conservative (no entropy coding credited): every kept coefficient costs
b bits + its basis index (log2 of the tensor's entry count) + 16-bit scale.
"""
import math
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- model load

def unique_parameters(sd):
    """Dedup weight-tied views (llama2.c ties tok_embeddings to output.weight)."""
    seen, out = set(), {}
    for k, v in sd.items():
        if id(v) not in seen:
            seen.add(id(v))
            out[k] = v
    return out


def load_state_dict(path=None):
    """-> (state_dict dict[str, Tensor fp32], model_args dict, ckpt dict).

    llama2.c ties tok_embeddings.weight to output.weight; the checkpoint stores
    two identical copies and the trained model is tied, so alias to one tensor.
    """
    path = path or os.path.join(DATA, "stories260K.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v.float() for k, v in ck["model"].items()}
    sd["tok_embeddings.weight"] = sd["output.weight"]
    return sd, ck["model_args"], ck


def to_device(sd, device):
    return {k: v.to(device) for k, v in sd.items()}


# ---------------------------------------------------------------- forward

def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos):
    """RoPE on (B, T, H, D) with llama2.c interleaved pairs; pos: (T,)."""
    D = x.shape[-1]
    inv = 10000.0 ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    ang = pos.to(x.device).float()[:, None] * inv[None, :]  # (T, D/2)
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def forward(sd, idx, ma):
    """idx: (B, T) int64 on device -> logits (B, T, V) fp32."""
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    h = sd["tok_embeddings.weight"][idx]  # (B, T, dim)
    pos = torch.arange(T, device=idx.device)
    mask = torch.full((T, T), float("-inf"), device=idx.device).triu(1)
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hn = rmsnorm(h, sd[p + "attention_norm.weight"])
        q = (hn @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k = (hn @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v = (hn @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q = rope(q, pos)
        k = rope(k, pos)
        rep = nh // nkv
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
        att = (q @ k.transpose(-2, -1)) / hd**0.5 + mask
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, dim)
        h = h + (o @ sd[p + "attention.wo.weight"].T)
        hn = rmsnorm(h, sd[p + "ffn_norm.weight"])
        w1out = hn @ sd[p + "feed_forward.w1.weight"].T
        ff = (w1out * torch.sigmoid(w1out)
              * (hn @ sd[p + "feed_forward.w3.weight"].T))
        h = h + (ff @ sd[p + "feed_forward.w2.weight"].T)
    h = rmsnorm(h, sd["norm.weight"])
    return h @ sd["output.weight"].T


# ---------------------------------------------------------------- tokenizer

def get_sp():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(DATA, "tok512.model"))
    return sp


def valid_tokens(n_tokens):
    """Token stream from TinyStories-valid.txt, llama2.c training prep:
    BOS-prefixed stories, NO EOS, concatenated (BOS is the story delimiter)."""
    cache = os.path.join(DATA, "valid_tokens.npy")
    if os.path.exists(cache):
        tok = np.load(cache)
    else:
        sp = get_sp()
        bos = sp.bos_id()
        txt = open(os.path.join(DATA, "TinyStories-valid.txt"), encoding="utf-8").read()
        docs = [d.strip() for d in re_end_of_doc().split(txt) if d.strip()]
        ids = []
        for d in docs:
            ids += [bos] + sp.encode(d)
        tok = np.array(ids, dtype=np.int32)
        np.save(cache, tok)
    return tok[:n_tokens]


def re_end_of_doc():
    import re
    return re.compile(r"<END_OF_DOC_TAG>|_{8,}")


# ---------------------------------------------------------------- eval

@torch.no_grad()
def eval_loss(sd, ma, tokens, device, n_windows=384, block=None, batch=16):
    """Paired-window val loss: next-token NLL over disjoint 512-token windows,
    first position of each window dropped. Returns (mean, sem, per-window)."""
    block = block or ma["max_seq_len"]
    need = n_windows * block
    assert len(tokens) >= need, f"need {need} tokens, have {len(tokens)}"
    win = torch.from_numpy(tokens[:need].astype(np.int64)).view(n_windows, block)
    losses = []
    for i in range(0, n_windows, batch):
        b = win[i:i + batch].to(device)
        lg = forward(sd, b, ma)[:, :-1]
        tg = b[:, 1:]
        nll = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
        losses.append(nll.mean(-1).cpu())
    losses = torch.cat(losses)
    return losses.mean().item(), (losses.std(unbiased=True) / len(losses)**0.5).item(), losses


@torch.no_grad()
def sample_story(sd, ma, device, n_new=320, temp=0.8, topk=50, seed=0):
    """Generate from prompt 'Once upon a time,' starting with BOS."""
    torch.manual_seed(seed)
    sp = get_sp()
    ids = [sp.bos_id()] + sp.encode("Once upon a time,")
    for _ in range(n_new):
        ctx = ids[-ma["max_seq_len"]:]
        logits = forward(sd, torch.tensor([ctx], device=device), ma)[0, -1]
        logits = logits.float() / temp
        v, ix = torch.topk(logits, min(topk, logits.shape[-1]))
        idx = ix[torch.multinomial(torch.softmax(v, -1), 1)]
        ids.append(int(idx))
    return sp.decode([i for i in ids if i not in (0,)])[len("Once upon a time,"):].strip()


# ---------------------------------------------------------------- DCT machinery

def dct_matrix(N):
    """Orthonormal DCT-II basis (N x N). forward = C @ x, inverse = C.T @ x."""
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64)
    C = torch.cos(math.pi / N * (n[None, :] + 0.5) * k[:, None]) * math.sqrt(2.0 / N)
    C[0] /= math.sqrt(2.0)
    return C


def dct2(W):
    Cn, Cm = dct_matrix(W.shape[0]), dct_matrix(W.shape[1])
    return Cn @ W @ Cm.T


def idct2(X):
    Cn, Cm = dct_matrix(X.shape[0]), dct_matrix(X.shape[1])
    return Cn.T @ X @ Cm


def quantize_sym(arr, b):
    """Symmetric uniform quantization to b bits (int levels in [-2^(b-1)+1, 2^(b-1)-1],
    level 0 exact). Returns (levels int32, scale float64) such that arr ≈ levels*scale."""
    m = float(arr.abs().max() if isinstance(arr, torch.Tensor) else np.abs(arr).max())
    qmax = 2 ** (b - 1) - 1
    scale = m / qmax if m > 0 else 1.0
    if isinstance(arr, torch.Tensor):
        q = torch.round(arr / scale).clamp(-qmax, qmax).to(torch.int32)
    else:
        q = np.rint(np.asarray(arr) / scale).clip(-qmax, qmax).astype(np.int32)
    return q, scale


def dequant(q, scale):
    return q.double() * scale if isinstance(q, torch.Tensor) else np.asarray(q, dtype=np.float64) * scale


def rel_err(W, Wh):
    return float((torch.as_tensor(Wh) - torch.as_tensor(W)).norm() / torch.as_tensor(W).norm())


# ---------------------------------------------------------------- bits accounting

def tensor_bits_schemeB(n, m, k, b):
    """Magnitude-top-K scheme: k kept coefficients, b bits each + index + scale header."""
    return k * (b + math.ceil(math.log2(n * m))) + 32


def tensor_bits_schemeA(kx, ky, b, original_shape):
    """Index-free low-pass rectangle: kx*ky coefficients, b bits each + scale header."""
    return kx * ky * b + 32