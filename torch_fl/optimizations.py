"""
Optional runtime optimization that fuses HuggingFace's ``*RMSNorm.forward``
into a single ``torch.ops.flagos.rms_norm`` kernel launch.

Background (see ``docs/trace_optimization_plan_2026-07-06.md`` §2.2(b)):
HuggingFace's ``*RMSNorm.forward`` is a hand-written chain of 5+ separate
aten launches::

    hidden_states = hidden_states.to(torch.float32)          # _to_copy #1
    variance = hidden_states.pow(2).mean(-1, keepdim=True)   # pow + mean
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)       # _to_copy #2 + mul

On the launch-overhead-bound MetaX v2 backend (~3us CPU per launch), shaving
5 launches per RMSNorm call (~2.7k calls / 64-token Qwen3 decode) directly
buys throughput. The fused kernel reduces in fp32 and keeps input/output in
the model dtype — numerically equivalent to the HF fp32 path.

HF RMSNorm is a hand-written forward (not an aten op), so it cannot be
intercepted through the dispatcher. This module patches the modelling
classes' ``forward`` to call ``torch.ops.flagos.rms_norm`` directly. Opt-in:
set ``FLAGOS_RMSNORM_FUSED=1`` before importing ``torch_fl`` to install an
import hook that patches the classes as they are loaded, or call
:func:`apply_fused_rmsnorm` explicitly after importing transformers.

Requires the flagos C++ extension providing ``flagos::rms_norm``
(see ``csrc/aten/rmsnorm.h`` and ``backends/metax_v2/rmsnorm_v2.cu``).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import threading
from typing import Dict, List, Tuple

import torch

__all__ = [
    "apply_fused_rmsnorm",
    "restore_rmsnorm",
    "is_rmsnorm_fused_active",
    "FLAGOS_RMSNORM_FUSED_ENV",
]

FLAGOS_RMSNORM_FUSED_ENV = "FLAGOS_RMSNORM_FUSED"

# (module path, class name) -> nothing. Order does not matter; the import hook
# keys off the module path and patches every class listed for that module.
_TARGETS: Dict[str, Tuple[str, ...]] = {
    "transformers.models.qwen3.modeling_qwen3": ("Qwen3RMSNorm",),
    "transformers.models.qwen2.modeling_qwen2": ("Qwen2RMSNorm",),
    "transformers.models.llama.modeling_llama": ("LlamaRMSNorm",),
    "transformers.models.mistral.modeling_mistral": ("MistralRMSNorm",),
    "transformers.models.mixtral.modeling_mixtral": ("MixtralRMSNorm",),
    "transformers.models.gemma2.modeling_gemma2": ("Gemma2RMSNorm",),
    "transformers.models.gemma.modeling_gemma": ("GemmaRMSNorm",),
    "transformers.models.starcoder2.modeling_starcoder2": ("Starcoder2RMSNorm",),
}

# cls qualified-name -> original forward. Kept so the patch is reversible and
# idempotent (we can tell whether a class is already patched by identity).
_ORIG_FORWARDS: Dict[str, object] = {}

_lock = threading.Lock()
_hook_installed = False


def _fused_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """RMSNorm forward that runs the whole normalize as one fused kernel.

    Replaces HF's pow/mean/add/rsqrt/mul chain (5+ launches) with a single
    ``torch.ops.flagos.rms_norm`` launch. The kernel reduces in fp32 and keeps
    input/output in the model dtype — numerically equivalent to the HF fp32
    path, but with far fewer launches on the launch-overhead-bound MetaX v2
    backend. See docs/trace_optimization_plan_2026-07-06.md §2.2(b).
    """
    return torch.ops.flagos.rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _is_fused_forward(fn: object) -> bool:
    return getattr(fn, "__name__", None) == "_fused_rmsnorm_forward"


def _install_on_class(cls: type, qualname: str) -> bool:
    """Install the fused forward on a single class. Returns True if newly
    patched."""
    if _is_fused_forward(cls.__dict__.get("forward")):
        return False
    # Only patch classes whose forward looks like the HF RMSNorm pattern: it
    # must call ``.to(torch.float32)`` on the input. This guards against
    # silently overriding a customised RMSNorm.
    try:
        import inspect

        src = inspect.getsource(cls.forward)
    except Exception:
        src = ""
    if "float32" not in src and "to(input_dtype)" not in src:
        return False
    _ORIG_FORWARDS[qualname] = cls.forward
    cls.forward = _fused_rmsnorm_forward  # type: ignore[assignment]
    return True


def _patch_module(mod: object) -> List[str]:
    """Patch every target RMSNorm class defined in *mod*. Returns the list of
    qualified names that were patched (newly, not previously)."""
    mod_name = getattr(mod, "__name__", "")
    cls_names = _TARGETS.get(mod_name)
    if not cls_names:
        return []
    patched: List[str] = []
    for cls_name in cls_names:
        cls = getattr(mod, cls_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        qualname = f"{mod_name}.{cls_name}"
        with _lock:
            did = _install_on_class(cls, qualname)
        if did:
            patched.append(qualname)
    return patched


def apply_fused_rmsnorm() -> List[str]:
    """Patch every already-imported HF ``*RMSNorm`` class to run the whole
    normalize as a single ``torch.ops.flagos.rms_norm`` launch.

    Safe to call before or after transformers is imported; classes that are
    not yet on ``sys.modules`` are skipped (use the ``FLAGOS_RMSNORM_FUSED=1``
    import hook to catch them lazily). Idempotent: calling twice is a no-op.

    Returns the list of qualified class names that were *newly* patched.
    """
    patched: List[str] = []
    for mod_name in _TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        patched.extend(_patch_module(mod))
    return patched


def restore_rmsnorm() -> List[str]:
    """Undo :func:`apply_fused_rmsnorm` / the import hook. Returns the list of
    qualified class names that were restored."""
    restored: List[str] = []
    with _lock:
        for qualname, orig in list(_ORIG_FORWARDS.items()):
            mod_name, _, cls_name = qualname.rpartition(".")
            # qualname is "transformers.models.<pkg>.modeling_<pkg>.<Cls>"
            # so rpartition(".") splits off the class name correctly even though
            # the module path itself contains dots.
            mod = sys.modules.get(mod_name)
            cls = getattr(mod, cls_name, None) if mod is not None else None
            if cls is not None and _is_fused_forward(cls.__dict__.get("forward")):
                cls.forward = orig  # type: ignore[assignment]
                restored.append(qualname)
        _ORIG_FORWARDS.clear()
    return restored


def is_rmsnorm_fused_active() -> bool:
    """True if at least one RMSNorm class is currently patched with the fused
    single-kernel forward."""
    with _lock:
        return bool(_ORIG_FORWARDS)


class _RMSNormImportHook(importlib.abc.MetaPathFinder):
    """Meta path finder that patches ``*RMSNorm`` classes right after their
    modelling module is exec'd by the normal import machinery.

    Installed only when ``FLAGOS_RMSNORM_FUSED=1`` so the default behaviour is
    unchanged. The finder returns ``None`` (does not take over loading) for
    every module; for target modules it wraps the resolved loader's
    ``exec_module`` to call :func:`_patch_module` afterwards.
    """

    def find_spec(self, fullname, path, target=None):  # noqa: D401, ARG002
        if fullname not in _TARGETS:
            return None
        # Resolve the real spec using the remaining finders (avoid recursion).
        real_spec = self._find_real_spec(fullname, path, target)
        if real_spec is None or real_spec.loader is None:
            return None
        real_loader = real_spec.loader
        if _is_fused_forward(getattr(real_loader, "__wrapped_exec__", None)):
            # Already wrapped by us — leave the spec untouched.
            return None

        class _PatchingLoader(importlib.abc.Loader):
            __wrapped_exec__ = _fused_rmsnorm_forward  # marker, identity-checked

            def create_module(self, spec):  # noqa: D401, ARG002
                if hasattr(real_loader, "create_module"):
                    return real_loader.create_module(spec)
                return None

            def exec_module(self, module):
                real_loader.exec_module(module)
                try:
                    _patch_module(module)
                except Exception:
                    # Never let the patch break model import.
                    pass

        real_spec.loader = _PatchingLoader()
        return real_spec

    @staticmethod
    def _find_real_spec(fullname, path, target):
        for finder in sys.meta_path:
            if isinstance(finder, _RMSNormImportHook):
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            try:
                spec = find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None:
                return spec
        # Fall back to importlib.util.find_spec (handles PathFinder etc.).
        try:
            return importlib.util.find_spec(fullname)
        except (ImportError, ValueError):
            return None


def _install_import_hook() -> None:
    """Install the meta path finder (once). Gated by env var at import time."""
    global _hook_installed
    with _lock:
        if _hook_installed:
            return
        # Prepend so we run before PathFinder wraps loaders, but only matter for
        # the target module names.
        sys.meta_path.insert(0, _RMSNormImportHook())
        _hook_installed = True


def _maybe_auto_install() -> None:
    if os.environ.get(FLAGOS_RMSNORM_FUSED_ENV, "0") == "1":
        _install_import_hook()
        # Also patch anything already loaded (in case transformers was imported
        # before torch_fl — uncommon on MACA, but harmless to check).
        apply_fused_rmsnorm()


_maybe_auto_install()
