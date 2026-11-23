#!/usr/bin/env python3
"""Generic HuggingFace uploader for the ninths artifacts.

Usage:
  hf_upload_glm45air.py FILE REPO_ID [--name IN_REPO_NAME] [--private]

Reads the HF token from $HF_TOKEN or ~/.cache/huggingface/token (strips NUL
padding — the cached file on this box is pre-padded with broken zeros).
"""
import argparse
import os
import pathlib
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def token() -> str:
    tok = os.environ.get("HF_TOKEN", "").replace("\x00", "").strip()
    if tok:
        return tok
    tf = pathlib.Path.home() / ".cache" / "huggingface" / "token"
    if tf.exists():
        tok = tf.read_text().replace("\x00", "").strip()
        if tok.startswith("hf_"):
            return tok
    raise SystemExit(
        "No HF token: export HF_TOKEN (write scope) or fix "
        "~/.cache/huggingface/token — the cached copy has NUL padding.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("repo")
    ap.add_argument("--name", default=None)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    src = pathlib.Path(a.file).resolve()
    if not src.is_file():
        raise SystemExit(f"missing file: {src}")
    name = a.name or src.name
    api = HfApi(token=token())
    try:
        api.repo_info(a.repo, repo_type="model")
        print(f"repo exists: {a.repo}")
    except RepositoryNotFoundError:
        api.create_repo(a.repo, repo_type="model", private=a.private)
        print(f"created repo: {a.repo} (private={bool(a.private)})")
    print(f"uploading {src} ({src.stat().st_size} bytes) -> {a.repo}/{name}")
    api.upload_file(path_or_fileobj=str(src), path_in_repo=name,
                    repo_id=a.repo, repo_type="model")
    print("done:", f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
