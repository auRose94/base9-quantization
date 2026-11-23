#!/usr/bin/env python3
"""Archive the f16 reference GGUF to a private HF repo, verify by sha256,
then (only on exact match) free the local copy."""
import hashlib, pathlib, sys
from huggingface_hub import HfApi
from hf_upload_glm45air import token

LOCAL = pathlib.Path("/home/rose/glm45air_ref_f16.gguf")
REPO = "auRose94/glm45air-f16-ref-gguf"
LOG = "/home/rose/glm_f16_ref_archive.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as fh:
        fh.write(msg + "\n")

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

api = HfApi(token=token())
try:
    api.repo_info(REPO, repo_type="model")
except Exception:
    api.create_repo(REPO, repo_type="model", private=True)
    log(f"created private repo {REPO}")
api.upload_file(path_or_fileobj="# f16 reference GGUF for GLM-4.5-Air ninths builds\n"
                "Converted from zai-org/GLM-4.5-Air safetensors (fork branch k9,\n"
                "convert_hf_to_llama.py). Reference for byte-exact gates; not needed\n"
                "for inference. Private archive.\n".encode(),
                path_in_repo="README.md", repo_id=REPO, repo_type="model")
log(f"uploading {LOCAL} ({LOCAL.stat().st_size} bytes) ...")
api.upload_file(path_or_fileobj=str(LOCAL), path_in_repo=LOCAL.name,
                repo_id=REPO, repo_type="model")
info = api.model_info(REPO, files_metadata=True)
remote = next((s for s in info.siblings if s.rfilename == LOCAL.name), None)
if remote is None:
    log("VERIFY FAIL: file missing on hub — local copy kept"); sys.exit(1)
log(f"remote size={remote.size} blobs={getattr(remote, 'blob_id', None) is not None}")
local_sha = sha256_of(LOCAL)
remote_sha = getattr(remote, "lfs", None) and remote.lfs.sha256
log(f"sha256 local={local_sha[:16]}... remote={remote_sha[:16] if remote_sha else None}...")
if remote_sha and remote_sha == local_sha and remote.size == LOCAL.stat().st_size:
    LOCAL.unlink()
    log("VERIFIED — local f16 ref deleted (206 GiB freed)")
else:
    log("VERIFY MISMATCH — local copy KEPT")
