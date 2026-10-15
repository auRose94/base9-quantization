"""Unblock torch/transformers imports on machines with a mismatched torchaudio.

This box has torch built for CUDA 13.4 but torchaudio for 13.3; torchaudio
raises a version-mismatch RuntimeError at import, which transformers turns
into a fatal model-import failure ("Could not import module
'Qwen2ForCausalLM'"). Nothing on this repo's transformers path touches
audio, so the optional torchaudio import is replaced by a stub with a real
package spec (transformers' availability checks call find_spec on it).

Usage — call BEFORE importing torch/transformers:

    from audio_shim import apply_torchaudio_shim
    apply_torchaudio_shim()
    import torch                      # noqa: E402
"""
import importlib.machinery
import sys
import types


def apply_torchaudio_shim() -> None:
    try:
        import torchaudio                          # noqa: F401
        return
    except Exception:
        pass
    stub = types.ModuleType("torchaudio")
    # a bare ModuleType has __spec__ = None, and transformers'
    # _is_package_available() raises ValueError on exactly that — give the
    # stub a real package spec so availability checks simply proceed.
    stub.__spec__ = importlib.machinery.ModuleSpec(
        "torchaudio", None, is_package=True)

    def _need(*a, **k):                            # noqa: ARG001
        raise RuntimeError("torchaudio is stubbed (CUDA version mismatch)")

    stub.load = _need
    functional = types.ModuleType("torchaudio.functional")
    functional.resample = _need
    stub.functional = functional
    sys.modules["torchaudio"] = stub