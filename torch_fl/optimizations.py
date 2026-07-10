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
    "apply_fused_rope",
    "restore_rope",
    "is_rope_fused_active",
    "FLAGOS_ROPE_FUSED_ENV",
    "apply_fused_softmax",
    "restore_softmax",
    "is_softmax_fused_active",
    "FLAGOS_SOFTMAX_FUSED_ENV",
]

FLAGOS_RMSNORM_FUSED_ENV = "FLAGOS_RMSNORM_FUSED"
FLAGOS_ROPE_FUSED_ENV = "FLAGOS_ROPE_FUSED"
FLAGOS_SOFTMAX_FUSED_ENV = "FLAGOS_SOFTMAX_FUSED"

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


# ===========================================================================
# Fused RoPE (flagos::rope) — module-level function patch
# ===========================================================================
#
# apply_rotary_pos_emb is a *module-level function* (not a class method):
# Qwen3Attention.forward calls it by its global name within the modelling
# module. Therefore the patch target is the module attribute itself
# (``setattr(module, "apply_rotary_pos_emb", _fused_rope)``), not a class
# ``forward`` like RMSNorm.
#
# HF chain per layer (12–14 launches):
#   cos.unsqueeze(d); sin.unsqueeze(d)            # 2× unsqueeze (view)
#   rotate_half(q): slice×2 + neg + cat           # 4 launches
#   rotate_half(k): slice×2 + neg + cat           # 4 launches
#   q*cos + rotate_half(q)*sin                    # mul×2 + add
#   k*cos + rotate_half(k)*sin                    # mul×2 + add
# Fused: a single ``torch.ops.flagos.rope`` launch (q and k each one kernel
# internally). 28 layers × 64 tokens → ~330 launches removed.

# (module path, function name) -> nothing. RoPE patches the module attribute.
_ROPE_TARGETS: Dict[str, str] = {
    "transformers.models.qwen3.modeling_qwen3": "apply_rotary_pos_emb",
    "transformers.models.qwen2.modeling_qwen2": "apply_rotary_pos_emb",
    "transformers.models.llama.modeling_llama": "apply_rotary_pos_emb",
    "transformers.models.mistral.modeling_mistral": "apply_rotary_pos_emb",
    "transformers.models.mixtral.modeling_mixtral": "apply_rotary_pos_emb",
    "transformers.models.gemma.modeling_gemma": "apply_rotary_pos_emb",
    "transformers.models.gemma2.modeling_gemma2": "apply_rotary_pos_emb",
    "transformers.models.starcoder2.modeling_starcoder2": "apply_rotary_pos_emb",
}

# qualname -> original function. Kept so the patch is reversible and
# idempotent (identity check tells whether already patched).
_ROPE_ORIG: Dict[str, object] = {}

_ROPE_HOOK_INSTALLED = False


def _fused_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Fused rotary position embedding — single ``torch.ops.flagos.rope``
    launch replacing HF's unsqueeze + rotate_half + mul + add chain (12–14
    launches). Signature matches HF ``apply_rotary_pos_emb`` (including the
    unused ``position_ids`` arg). The kernel computes in fp32 internally and
    keeps q/k in the model dtype — numerically equivalent to the HF fp32 path.
    """
    q_embed, k_embed = torch.ops.flagos.rope(q, k, cos, sin, unsqueeze_dim)
    return q_embed, k_embed


def _is_fused_rope(fn: object) -> bool:
    return getattr(fn, "__name__", None) == "_fused_rope"


def _install_rope_on_module(mod, mod_name: str) -> List[str]:
    """Patch ``apply_rotary_pos_emb`` on *mod* with the fused version. Returns
    the list of qualified names newly patched."""
    fn_name = _ROPE_TARGETS.get(mod_name)
    if fn_name is None:
        return []
    current = getattr(mod, fn_name, None)
    qualname = f"{mod_name}.{fn_name}"
    if _is_fused_rope(current):
        return []
    # Guard: only patch if the target looks like HF's implementation
    # (contains ``rotate_half`` and ``unsqueeze``), avoiding custom impls.
    try:
        import inspect

        src = inspect.getsource(current)
    except Exception:
        src = ""
    if "rotate_half" not in src or "unsqueeze" not in src:
        return []
    with _lock:
        _ROPE_ORIG[qualname] = current
    setattr(mod, fn_name, _fused_rope)
    return [qualname]


def apply_fused_rope() -> List[str]:
    """Patch every already-imported HF ``apply_rotary_pos_emb`` module-level
    function to run the single fused ``torch.ops.flagos.rope`` launch.

    Safe to call before or after transformers is imported; modules not yet on
    ``sys.modules`` are skipped (use the ``FLAGOS_ROPE_FUSED=1`` import hook
    to catch them lazily). Idempotent.

    Returns the list of qualified names that were *newly* patched.
    """
    patched: List[str] = []
    for mod_name in _ROPE_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        patched.extend(_install_rope_on_module(mod, mod_name))
    return patched


def restore_rope() -> List[str]:
    """Undo :func:`apply_fused_rope` / the import hook. Returns the list of
    qualified names that were restored."""
    restored: List[str] = []
    with _lock:
        for qualname, orig in list(_ROPE_ORIG.items()):
            mod_name, _, fn_name = qualname.rpartition(".")
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            current = getattr(mod, fn_name, None)
            if _is_fused_rope(current):
                setattr(mod, fn_name, orig)
                restored.append(qualname)
        _ROPE_ORIG.clear()
    return restored


def is_rope_fused_active() -> bool:
    """True if at least one ``apply_rotary_pos_emb`` is currently patched with
    the fused single-kernel version."""
    with _lock:
        return bool(_ROPE_ORIG)


def _install_rope_import_hook() -> None:
    """Install the unified fusion meta path finder (once). Gated by env var.

    The original per-fusion RoPE finder (_RoPEImportHook) has been replaced by
    the single _FusionImportHook defined further below (it applies all three
    patches). This name is kept as a back-compat alias target.
    """
    _install_fusion_import_hook()


def _maybe_auto_install_rope() -> None:
    if os.environ.get(FLAGOS_ROPE_FUSED_ENV, "0") == "1":
        _install_rope_import_hook()
        apply_fused_rope()


# NOTE: the actual env-gated install + apply for ALL fusions happens once at the
# bottom of this module in _maybe_auto_install(), after the unified import hook
# is defined. Do not call _maybe_auto_install_rope() here — _install_*_import_hook
# is an alias for _install_fusion_import_hook which is defined further below.


# ===========================================================================
# Fused masked softmax (flagos::masked_softmax) — module-level function patch
# ===========================================================================
#
# HuggingFace eager_attention_forward builds the attention scores then runs a
# softmax sub-chain:
#   attn_weights = torch.matmul(q, k^T) * scaling          # mul scalar (1+)
#   if mask: attn_weights = attn_weights + attention_mask   # add broadcast (1+)
#   attn_weights = nn.functional.softmax(..., dim=-1,
#                       dtype=torch.float32).to(query.dtype)# softmax + _to_copy
# Each of those is 1+ launches; nn.functional.softmax alone is max/sub/exp/sum/
# div internally. This patch replaces the whole sub-chain with a single
# ``torch.ops.flagos.masked_softmax`` launch that fuses scale + mask + stable
# softmax (fp32 reduction) + dtype cast. Covers both "softmax op fusion" (the
# softmax reduction) and "attention op fusion" (the surrounding scale/mask/cast
# chain), which are adjacent in eager_attention_forward — hence implemented as
# one kernel.

# (module path, function name) -> nothing. Softmax patches the module attribute.
_SOFTMAX_TARGETS: Dict[str, str] = {
    "transformers.models.qwen3.modeling_qwen3": "eager_attention_forward",
    "transformers.models.qwen2.modeling_qwen2": "eager_attention_forward",
    "transformers.models.llama.modeling_llama": "eager_attention_forward",
    "transformers.models.mistral.modeling_mistral": "eager_attention_forward",
    "transformers.models.mixtral.modeling_mixtral": "eager_attention_forward",
    "transformers.models.gemma.modeling_gemma": "eager_attention_forward",
    "transformers.models.gemma2.modeling_gemma2": "eager_attention_forward",
    "transformers.models.starcoder2.modeling_starcoder2": "eager_attention_forward",
}

# qualname -> (original function, repeat_kv reference). Kept so the patch is
# reversible and idempotent (identity check tells whether already patched).
_SOFTMAX_ORIG: Dict[str, object] = {}

_SOFTMAX_HOOK_INSTALLED = False


def _make_fused_eager_attention_forward(repeat_kv, torch_mod):
    """Build a patched ``eager_attention_forward`` that runs the scale+mask+
    softmax+cast sub-chain as a single ``torch.ops.flagos.masked_softmax``
    launch. Captures ``repeat_kv`` / ``torch`` from the modelling module's
    namespace at patch time so the wrapper is self-contained.
    """

    def _fused_eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        # Raw bmm output — scaling is applied inside the fused kernel.
        attn_weights = torch_mod.matmul(query, key_states.transpose(2, 3))
        attn_weights = torch_mod.ops.flagos.masked_softmax(
            attn_weights, attention_mask, scaling
        )
        attn_weights = torch_mod.nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = torch_mod.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return _fused_eager_attention_forward


def _is_fused_softmax(fn: object) -> bool:
    return getattr(fn, "__name__", None) == "_fused_eager_attention_forward"


def _install_softmax_on_module(mod, mod_name: str) -> List[str]:
    """Patch ``eager_attention_forward`` on *mod* with the fused version.
    Returns the list of qualified names newly patched."""
    fn_name = _SOFTMAX_TARGETS.get(mod_name)
    if fn_name is None:
        return []
    current = getattr(mod, fn_name, None)
    qualname = f"{mod_name}.{fn_name}"
    if _is_fused_softmax(current):
        return []
    # Guard: only patch if the target looks like HF's eager attention
    # (references softmax + scaling), avoiding custom impls.
    try:
        import inspect

        src = inspect.getsource(current)
    except Exception:
        src = ""
    if "softmax" not in src or "scaling" not in src:
        return []
    # Capture helpers the wrapper needs from the module's namespace.
    repeat_kv = getattr(mod, "repeat_kv", None)
    torch_mod = getattr(mod, "torch", None)
    if repeat_kv is None or torch_mod is None:
        # Fallbacks: resolve from the function's own globals (HF modules import
        # these at module scope).
        fn_globals = getattr(current, "__globals__", {})
        repeat_kv = repeat_kv or fn_globals.get("repeat_kv")
        torch_mod = torch_mod or fn_globals.get("torch")
    if repeat_kv is None or torch_mod is None:
        return []
    with _lock:
        _SOFTMAX_ORIG[qualname] = current
    setattr(mod, fn_name, _make_fused_eager_attention_forward(repeat_kv, torch_mod))
    return [qualname]


def apply_fused_softmax() -> List[str]:
    """Patch every already-imported HF ``eager_attention_forward`` module-level
    function to run the single fused ``torch.ops.flagos.masked_softmax`` launch.

    Safe to call before or after transformers is imported; modules not yet on
    ``sys.modules`` are skipped (use the ``FLAGOS_SOFTMAX_FUSED=1`` import hook
    to catch them lazily). Idempotent.

    Returns the list of qualified names that were *newly* patched.
    """
    patched: List[str] = []
    for mod_name in _SOFTMAX_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        patched.extend(_install_softmax_on_module(mod, mod_name))
    return patched


def restore_softmax() -> List[str]:
    """Undo :func:`apply_fused_softmax` / the import hook."""
    restored: List[str] = []
    with _lock:
        for qualname, orig in list(_SOFTMAX_ORIG.items()):
            mod_name, _, fn_name = qualname.rpartition(".")
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            current = getattr(mod, fn_name, None)
            if _is_fused_softmax(current):
                setattr(mod, fn_name, orig)
                restored.append(qualname)
        _SOFTMAX_ORIG.clear()
    return restored


def is_softmax_fused_active() -> bool:
    """True if at least one ``eager_attention_forward`` is currently patched
    with the fused single-kernel version."""
    with _lock:
        return bool(_SOFTMAX_ORIG)


def _apply_all_fused_patches(module) -> None:
    """Apply every enabled fused patch (RMSNorm / RoPE / softmax) to *module*.

    Used by all three import hooks' ``exec_module`` so that whichever hook wraps
    a given modelling module's loader, all three patches (gated by their
    respective env vars) get applied. Never raises — a failed patch must not
    break model import.
    """
    try:
        mod_name = getattr(module, "__name__", "")
        if os.environ.get(FLAGOS_RMSNORM_FUSED_ENV, "0") == "1":
            _patch_module(module)
        if os.environ.get(FLAGOS_ROPE_FUSED_ENV, "0") == "1":
            _install_rope_on_module(module, mod_name)
        if os.environ.get(FLAGOS_SOFTMAX_FUSED_ENV, "0") == "1":
            _install_softmax_on_module(module, mod_name)
    except Exception:
        # Never let the patch break model import.
        pass


def _maybe_auto_install_softmax() -> None:
    if os.environ.get(FLAGOS_SOFTMAX_FUSED_ENV, "0") == "1":
        _install_softmax_import_hook()
        apply_fused_softmax()


# (install is deferred to the unified _maybe_auto_install() at the bottom.)


# ===========================================================================
# Single unified import hook for ALL fused patches (RMSNorm / RoPE / softmax)
# ===========================================================================
#
# Earlier versions had three cooperating meta path finders (one per fusion).
# With three finders in sys.meta_path, each one's _find_real_spec could call
# the others' find_spec, and for shared target modules the mutual re-entry
# recursed indefinitely, deadlocking the import of modelling_qwen3 when all
# three env vars were set. Replacing them with ONE finder that targets the
# union of all patch targets and applies every enabled patch in a single
# exec_module wrap eliminates the recursion entirely.

# Union of every modelling module that any fusion targets (they all target the
# same set of HF modelling modules, but take the union defensively).
_ALL_FUSION_TARGETS = frozenset(
    list(_TARGETS.keys()) + list(_ROPE_TARGETS.keys()) + list(_SOFTMAX_TARGETS.keys())
)

_FUSION_HOOK_INSTALLED = False


class _FusionImportHook(importlib.abc.MetaPathFinder):
    """Single meta path finder that wraps the loader of every target HF
    modelling module and applies all enabled fused patches (RMSNorm classes,
    RoPE function, softmax function) right after the module is exec'd. Each
    patch is individually gated by its env var, so this one hook serves all
    three fusions regardless of which combination is enabled.
    """

    def find_spec(self, fullname, path, target=None):  # noqa: D401, ARG002
        if fullname not in _ALL_FUSION_TARGETS:
            return None
        # Resolve the real spec using the remaining finders (skip ourselves to
        # avoid recursion). There is only one flagos finder now, so no mutual
        # re-entry is possible.
        real_spec = None
        for finder in sys.meta_path:
            if isinstance(finder, _FusionImportHook):
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            try:
                spec = find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None:
                real_spec = spec
                break
        if real_spec is None:
            try:
                real_spec = importlib.util.find_spec(fullname)
            except (ImportError, ValueError):
                real_spec = None
        if real_spec is None or real_spec.loader is None:
            return None
        real_loader = real_spec.loader
        if getattr(real_loader, "__wrapped_fusion_exec__", None) is not None:
            # Already wrapped by us — leave it untouched.
            return None

        class _PatchingLoader(importlib.abc.Loader):
            __wrapped_fusion_exec__ = _apply_all_fused_patches  # marker

            def create_module(self, spec):  # noqa: D401, ARG002
                if hasattr(real_loader, "create_module"):
                    return real_loader.create_module(spec)
                return None

            def exec_module(self, module):
                real_loader.exec_module(module)
                _apply_all_fused_patches(module)

        real_spec.loader = _PatchingLoader()
        return real_spec


def _install_fusion_import_hook() -> None:
    """Install the unified fusion meta path finder (once). Installed whenever
    ANY of the three fusion env vars is set."""
    global _FUSION_HOOK_INSTALLED
    with _lock:
        if _FUSION_HOOK_INSTALLED:
            return
        sys.meta_path.insert(0, _FusionImportHook())
        _FUSION_HOOK_INSTALLED = True


# Back-compat aliases for the older per-fusion install entry points (still
# referenced by the env-gated auto-install blocks above).
_install_import_hook = _install_fusion_import_hook
_install_rope_import_hook = _install_fusion_import_hook
_install_softmax_import_hook = _install_fusion_import_hook


def _maybe_auto_install() -> None:
    # Install the unified finder if ANY fusion env var is on, then patch
    # anything already loaded (in case transformers was imported before
    # torch_fl — uncommon on MACA, but harmless to check).
    _any_on = (
        os.environ.get(FLAGOS_RMSNORM_FUSED_ENV, "0") == "1"
        or os.environ.get(FLAGOS_ROPE_FUSED_ENV, "0") == "1"
        or os.environ.get(FLAGOS_SOFTMAX_FUSED_ENV, "0") == "1"
    )
    if _any_on:
        _install_fusion_import_hook()
        apply_fused_rmsnorm()
        apply_fused_rope()
        apply_fused_softmax()


_maybe_auto_install()
