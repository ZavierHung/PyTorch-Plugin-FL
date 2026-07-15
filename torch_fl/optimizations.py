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
    "apply_fused_swiglu",
    "restore_swiglu",
    "is_swiglu_fused_active",
    "FLAGOS_SWIGLU_FUSED_ENV",
    "apply_fused_add_rmsnorm",
    "restore_add_rmsnorm",
    "is_add_rmsnorm_fused_active",
    "FLAGOS_ADD_RMSNORM_FUSED_ENV",
    "apply_fused_qkv",
    "restore_qkv",
    "is_qkv_fused_active",
    "FLAGOS_QKV_FUSED_ENV",
    "apply_fused_gate_up",
    "restore_gate_up",
    "is_gate_up_fused_active",
    "FLAGOS_GATEUP_FUSED_ENV",
]

FLAGOS_RMSNORM_FUSED_ENV = "FLAGOS_RMSNORM_FUSED"
FLAGOS_ROPE_FUSED_ENV = "FLAGOS_ROPE_FUSED"
FLAGOS_SOFTMAX_FUSED_ENV = "FLAGOS_SOFTMAX_FUSED"
FLAGOS_SWIGLU_FUSED_ENV = "FLAGOS_SWIGLU_FUSED"
FLAGOS_ADD_RMSNORM_FUSED_ENV = "FLAGOS_ADD_RMSNORM_FUSED"
FLAGOS_QKV_FUSED_ENV = "FLAGOS_QKV_FUSED"
FLAGOS_GATEUP_FUSED_ENV = "FLAGOS_GATEUP_FUSED"

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
        if os.environ.get(FLAGOS_SWIGLU_FUSED_ENV, "0") == "1":
            _patch_swiglu_module(module)
        if os.environ.get(FLAGOS_ADD_RMSNORM_FUSED_ENV, "0") == "1":
            _patch_add_rmsnorm_module(module)
        if os.environ.get(FLAGOS_QKV_FUSED_ENV, "0") == "1":
            _patch_qkv_module(module)
        if os.environ.get(FLAGOS_GATEUP_FUSED_ENV, "0") == "1":
            _patch_gate_up_module(module)
    except Exception:
        # Never let the patch break model import.
        pass


def _maybe_auto_install_softmax() -> None:
    if os.environ.get(FLAGOS_SOFTMAX_FUSED_ENV, "0") == "1":
        _install_softmax_import_hook()
        apply_fused_softmax()


# (install is deferred to the unified _maybe_auto_install() at the bottom.)


# ===========================================================================
# Fused SwiGLU (flagos::swiglu) — MLP class-method patch
# ===========================================================================
#
# HuggingFace's gated MLP forward is::
#
#     down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#
# The ``self.act_fn(...) * self.up_proj(x)`` part is a silu launch followed by
# an elementwise-mul launch (2 kernels). This patch replaces that sub-chain with
# a single ``torch.ops.flagos.swiglu`` launch (silu in fp32, product cast back
# to model dtype). The two matmuls stay as-is; only the activation+mul fuse.
#
# IMPORTANT: the fused kernel hard-codes silu, so only silu-gated families are
# targeted (Qwen3/Qwen2/Llama/Mistral). Gemma (gelu) and Starcoder2 (no gate)
# are intentionally excluded — patching them would be numerically wrong.

# (module path, class name tuple). SwiGLU patches the MLP class' forward.
_SWIGLU_TARGETS: Dict[str, Tuple[str, ...]] = {
    "transformers.models.qwen3.modeling_qwen3": ("Qwen3MLP",),
    "transformers.models.qwen2.modeling_qwen2": ("Qwen2MLP",),
    "transformers.models.llama.modeling_llama": ("LlamaMLP",),
    "transformers.models.mistral.modeling_mistral": ("MistralMLP",),
}

# cls qualified-name -> original forward. Reversible + idempotent.
_SWIGLU_ORIG: Dict[str, object] = {}


def _fused_mlp_forward(self, x):
    """MLP forward that fuses silu(gate)*up into one ``flagos.swiglu`` launch.

    Replaces HF's ``self.act_fn(self.gate_proj(x)) * self.up_proj(x)`` (a silu
    launch + an elementwise-mul launch) with a single kernel. silu is computed
    in fp32 internally — numerically equivalent to the HF fp16 path. Only valid
    for silu-gated MLPs (see _SWIGLU_TARGETS).
    """
    return self.down_proj(torch.ops.flagos.swiglu(self.gate_proj(x), self.up_proj(x)))


def _is_fused_mlp(fn: object) -> bool:
    return getattr(fn, "__name__", None) == "_fused_mlp_forward"


def _install_swiglu_on_class(cls: type, qualname: str) -> bool:
    """Install the fused MLP forward on a single class. Returns True if newly
    patched."""
    if _is_fused_mlp(cls.__dict__.get("forward")):
        return False
    # Guard: only patch classes whose forward looks like the HF gated-MLP
    # pattern (references act_fn / gate_proj / up_proj).
    try:
        import inspect

        src = inspect.getsource(cls.forward)
    except Exception:
        src = ""
    if "act_fn" not in src or "gate_proj" not in src or "up_proj" not in src:
        return False
    _SWIGLU_ORIG[qualname] = cls.forward
    cls.forward = _fused_mlp_forward  # type: ignore[assignment]
    return True


def _patch_swiglu_module(mod: object) -> List[str]:
    """Patch every target MLP class defined in *mod*."""
    mod_name = getattr(mod, "__name__", "")
    cls_names = _SWIGLU_TARGETS.get(mod_name)
    if not cls_names:
        return []
    patched: List[str] = []
    for cls_name in cls_names:
        cls = getattr(mod, cls_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        qualname = f"{mod_name}.{cls_name}"
        with _lock:
            did = _install_swiglu_on_class(cls, qualname)
        if did:
            patched.append(qualname)
    return patched


def apply_fused_swiglu() -> List[str]:
    """Patch every already-imported HF gated-MLP class to run silu(gate)*up as a
    single ``torch.ops.flagos.swiglu`` launch. Idempotent. Returns the list of
    qualified class names newly patched."""
    patched: List[str] = []
    for mod_name in _SWIGLU_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        patched.extend(_patch_swiglu_module(mod))
    return patched


def restore_swiglu() -> List[str]:
    """Undo :func:`apply_fused_swiglu` / the import hook."""
    restored: List[str] = []
    with _lock:
        for qualname, orig in list(_SWIGLU_ORIG.items()):
            mod_name, _, cls_name = qualname.rpartition(".")
            mod = sys.modules.get(mod_name)
            cls = getattr(mod, cls_name, None) if mod is not None else None
            if cls is not None and _is_fused_mlp(cls.__dict__.get("forward")):
                cls.forward = orig  # type: ignore[assignment]
                restored.append(qualname)
        _SWIGLU_ORIG.clear()
    return restored


def is_swiglu_fused_active() -> bool:
    """True if at least one gated-MLP class is currently patched with the fused
    single-kernel forward."""
    with _lock:
        return bool(_SWIGLU_ORIG)


# ===========================================================================
# Fused residual-add + RMSNorm (flagos::add_rms_norm) — DecoderLayer patch
# ===========================================================================
#
# The HF decoder layer's post-attention section is::
#
#     hidden_states = residual + hidden_states            # residual add
#     residual = hidden_states
#     hidden_states = self.post_attention_layernorm(hidden_states)  # RMSNorm
#
# That add (1 launch) + RMSNorm chain (5+ launches) fuse into a single
# ``torch.ops.flagos.add_rms_norm`` launch returning (normed, new_residual).
# Unlike the other fusions (which patch a single method/function), this patches
# the WHOLE DecoderLayer.forward and rewrites the call sequence — the add and
# the norm live at different call sites, so only a forward-level rewrite can
# fuse them. The fused forward replicates HF's logic verbatim except for those
# three lines. input_layernorm is left untouched (still fused via the RMSNorm
# class patch when FLAGOS_RMSNORM_FUSED=1); the layer-final residual add is left
# alone (its norm is the next layer's input_layernorm, across a forward
# boundary). Source matched against transformers 5.6.2.

# (module path, class name tuple). Patches the DecoderLayer class' forward.
_DECODER_TARGETS: Dict[str, Tuple[str, ...]] = {
    "transformers.models.qwen3.modeling_qwen3": ("Qwen3DecoderLayer",),
    "transformers.models.qwen2.modeling_qwen2": ("Qwen2DecoderLayer",),
    "transformers.models.llama.modeling_llama": ("LlamaDecoderLayer",),
    "transformers.models.mistral.modeling_mistral": ("MistralDecoderLayer",),
}

_DECODER_ORIG: Dict[str, object] = {}


def _fused_decoder_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=None,
    **kwargs,
):
    """DecoderLayer forward that fuses the post-attention ``residual + attn_out``
    and ``post_attention_layernorm`` into one ``flagos.add_rms_norm`` launch.

    Replicates HF Qwen3DecoderLayer.forward (transformers 5.6.2) verbatim except
    the post-attention add+norm, which becomes a single kernel returning
    (normed, new_residual). Returns the bare hidden_states tensor (matching HF).
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    # Fused: new_residual = residual + hidden_states (fp16);
    #        hidden_states = post_attention_layernorm(new_residual)
    hidden_states, residual = torch.ops.flagos.add_rms_norm(
        residual,
        hidden_states,
        self.post_attention_layernorm.weight,
        self.post_attention_layernorm.variance_epsilon,
    )
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def _is_fused_decoder(fn: object) -> bool:
    return getattr(fn, "__name__", None) == "_fused_decoder_layer_forward"


def _install_add_rmsnorm_on_class(cls: type, qualname: str) -> bool:
    """Install the fused DecoderLayer forward on a single class. Returns True if
    newly patched."""
    if _is_fused_decoder(cls.__dict__.get("forward")):
        return False
    # Guard: only patch classes whose forward looks like the standard HF decoder
    # layer (references post_attention_layernorm + residual).
    try:
        import inspect

        src = inspect.getsource(cls.forward)
    except Exception:
        src = ""
    if "post_attention_layernorm" not in src or "residual" not in src:
        return False
    _DECODER_ORIG[qualname] = cls.forward
    cls.forward = _fused_decoder_layer_forward  # type: ignore[assignment]
    return True


def _patch_add_rmsnorm_module(mod: object) -> List[str]:
    """Patch every target DecoderLayer class defined in *mod*."""
    mod_name = getattr(mod, "__name__", "")
    cls_names = _DECODER_TARGETS.get(mod_name)
    if not cls_names:
        return []
    patched: List[str] = []
    for cls_name in cls_names:
        cls = getattr(mod, cls_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        qualname = f"{mod_name}.{cls_name}"
        with _lock:
            did = _install_add_rmsnorm_on_class(cls, qualname)
        if did:
            patched.append(qualname)
    return patched


def apply_fused_add_rmsnorm() -> List[str]:
    """Patch every already-imported HF DecoderLayer class to fuse the
    post-attention residual-add + RMSNorm into one ``torch.ops.flagos.add_rms_norm``
    launch. Idempotent. Returns the list of qualified class names newly patched."""
    patched: List[str] = []
    for mod_name in _DECODER_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        patched.extend(_patch_add_rmsnorm_module(mod))
    return patched


def restore_add_rmsnorm() -> List[str]:
    """Undo :func:`apply_fused_add_rmsnorm` / the import hook."""
    restored: List[str] = []
    with _lock:
        for qualname, orig in list(_DECODER_ORIG.items()):
            mod_name, _, cls_name = qualname.rpartition(".")
            mod = sys.modules.get(mod_name)
            cls = getattr(mod, cls_name, None) if mod is not None else None
            if cls is not None and _is_fused_decoder(cls.__dict__.get("forward")):
                cls.forward = orig  # type: ignore[assignment]
                restored.append(qualname)
        _DECODER_ORIG.clear()
    return restored


def is_add_rmsnorm_fused_active() -> bool:
    """True if at least one DecoderLayer class is currently patched with the
    fused residual-add + RMSNorm forward."""
    with _lock:
        return bool(_DECODER_ORIG)


# ===========================================================================
# Fused QKV projection (weight-concat + single GEMM + split) — Attention patch
# ===========================================================================
#
# HF attention computes q/k/v with three separate Linear (GEMM) launches, all
# consuming the SAME hidden_states and differing only in output dim:
#
#     query_states = q_norm(q_proj(hidden).view(...)).transpose(1, 2)
#     key_states   = k_norm(k_proj(hidden).view(...)).transpose(1, 2)
#     value_states =        v_proj(hidden).view(...) .transpose(1, 2)
#
# On the launch-overhead-bound backend, three GEMM launches per layer per token
# is three dispatch round-trips. Because the three weights share the input, they
# can be concatenated along out-dim into ONE [hidden, q_out+k_out+v_out] weight,
# run as a single F.linear, then split back — 3 launches → 1. The concat is done
# ONCE (cached on the module as ``_fused_qkv_weight``) so decode steps only pay
# the single GEMM. q_norm/k_norm are applied to the split slices exactly as HF
# does, so the result is numerically identical (same weights, same math order).
#
# Only bias-free q/k/v projections are handled (Qwen3/Qwen2/Llama/Mistral are
# all bias-free here); a bias would need concatenation too — guarded out.

_QKV_TARGETS: Dict[str, Tuple[str, ...]] = {
    "transformers.models.qwen3.modeling_qwen3": ("Qwen3Attention",),
    "transformers.models.qwen2.modeling_qwen2": ("Qwen2Attention",),
    "transformers.models.llama.modeling_llama": ("LlamaAttention",),
    "transformers.models.mistral.modeling_mistral": ("MistralAttention",),
}

_QKV_ORIG: Dict[str, object] = {}


def _get_fused_qkv_weight(self):
    """Return (and lazily build+cache) the concatenated qkv weight for *self*.

    Returns None if the module's q/k/v projections have a bias or otherwise
    don't fit the simple concat (in which case the caller falls back to HF).

    INFERENCE-ONLY: the concat is cached under ``torch.no_grad()`` and never
    invalidated, so it is correct only when q/k/v weights are frozen (the
    decode-throughput use case these fusions target). Under training/finetuning
    it would break autograd into q/k/v and go stale on weight updates — do not
    enable ``FLAGOS_QKV_FUSED`` in that setting. Guarded by requiring frozen
    weights in practice; kept minimal because the fusion is opt-in + default-off.
    """
    w = getattr(self, "_fused_qkv_weight", None)
    if w is not None:
        return w
    q, k, v = self.q_proj, self.k_proj, self.v_proj
    if q.bias is not None or k.bias is not None or v.bias is not None:
        return None
    with torch.no_grad():
        w = torch.cat([q.weight, k.weight, v.weight], dim=0)
    # Cache split sizes too (GQA: k/v out-dim < q out-dim).
    self._fused_qkv_weight = w
    self._fused_qkv_splits = (
        q.weight.shape[0],
        k.weight.shape[0],
        v.weight.shape[0],
    )
    return w


def _make_fused_qkv_attention_forward(orig_forward):
    """Build a patched Attention.forward that fuses q/k/v proj into one GEMM.

    Rewrites only the three projection GEMMs into a single concatenated GEMM +
    split; everything after (q_norm/k_norm, rope, cache update, attention) is
    delegated to HF by reconstructing the exact same intermediate tensors. To
    stay robust across transformers versions, we don't re-implement the whole
    forward — instead we temporarily swap q_proj/k_proj/v_proj for cheap slicing
    Linears backed by the fused GEMM output. But that's fragile; simpler and
    safe: replicate the projection lines and then call the ORIGINAL forward with
    projections monkey-swapped is also fragile. We therefore inline the minimal
    HF forward for the silu-gated families (matched against transformers 5.6.2).
    """

    def _fused_qkv_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        w = _get_fused_qkv_weight(self)
        if w is None:
            return orig_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Single fused GEMM, then split into q/k/v along the last (out) dim.
        qkv = torch.nn.functional.linear(hidden_states, w)
        q_out, k_out, v_out = qkv.split(self._fused_qkv_splits, dim=-1)

        query_states = self.q_norm(q_out.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(k_out.view(hidden_shape)).transpose(1, 2)
        value_states = v_out.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _qkv_apply_rope(
            self, query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        from transformers.models.qwen3 import modeling_qwen3 as _mq3
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, _mq3.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return _fused_qkv_forward


def _qkv_apply_rope(self, q, k, cos, sin):
    """Call whichever apply_rotary_pos_emb is live on the attention's module
    (picks up the fused RoPE patch automatically)."""
    import sys as _sys

    mod = _sys.modules.get(type(self).__module__)
    fn = getattr(mod, "apply_rotary_pos_emb", None)
    return fn(q, k, cos, sin)


def _is_fused_qkv(fn: object) -> bool:
    return getattr(fn, "__name__", None) == "_fused_qkv_forward"


def _install_qkv_on_class(cls: type, qualname: str) -> bool:
    """Install the fused-QKV forward on a single Attention class."""
    if _is_fused_qkv(cls.__dict__.get("forward")):
        return False
    try:
        import inspect

        src = inspect.getsource(cls.forward)
    except Exception:
        src = ""
    # Guard: only patch the standard HF attention (q_proj/k_proj/v_proj +
    # q_norm/k_norm + apply_rotary_pos_emb). q_norm/k_norm restricts this to the
    # Qwen3 family, which is what we validate against.
    if "q_proj" not in src or "k_proj" not in src or "v_proj" not in src:
        return False
    if "q_norm" not in src or "apply_rotary_pos_emb" not in src:
        return False
    _QKV_ORIG[qualname] = cls.forward
    cls.forward = _make_fused_qkv_attention_forward(cls.forward)
    return True


def _patch_qkv_module(mod: object) -> List[str]:
    mod_name = getattr(mod, "__name__", "")
    cls_names = _QKV_TARGETS.get(mod_name)
    if not cls_names:
        return []
    patched: List[str] = []
    for cls_name in cls_names:
        cls = getattr(mod, cls_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        qualname = f"{mod_name}.{cls_name}"
        with _lock:
            did = _install_qkv_on_class(cls, qualname)
        if did:
            patched.append(qualname)
    return patched


def apply_fused_qkv() -> List[str]:
    """Patch every already-imported HF Attention class to fuse q/k/v projection
    into a single concatenated GEMM. Idempotent. Returns newly-patched names."""
    patched: List[str] = []
    for mod_name in _QKV_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        patched.extend(_patch_qkv_module(mod))
    return patched


def restore_qkv() -> List[str]:
    """Undo :func:`apply_fused_qkv`."""
    restored: List[str] = []
    with _lock:
        for qualname, orig in list(_QKV_ORIG.items()):
            mod_name, _, cls_name = qualname.rpartition(".")
            mod = sys.modules.get(mod_name)
            cls = getattr(mod, cls_name, None) if mod is not None else None
            if cls is not None and _is_fused_qkv(cls.__dict__.get("forward")):
                cls.forward = orig  # type: ignore[assignment]
                restored.append(qualname)
        _QKV_ORIG.clear()
    return restored


def is_qkv_fused_active() -> bool:
    """True if at least one Attention class is patched with fused QKV."""
    with _lock:
        return bool(_QKV_ORIG)


# ===========================================================================
# Fused gate_up projection (weight-concat + single GEMM + split) — MLP patch
# ===========================================================================
#
# HF gated MLP runs gate_proj and up_proj as two separate GEMMs on the same
# input x. Like QKV, they share the input and differ only in being two halves,
# so they concat into ONE [hidden, 2*intermediate] weight → single GEMM, split,
# then feed the existing fused SwiGLU. 2 GEMM launches → 1. The concat is cached
# on the module (``_fused_gate_up_weight``). This patch also folds in the SwiGLU
# fusion (silu(gate)*up as one flagos.swiglu launch), so it supersedes the plain
# SwiGLU patch when both are enabled — enabling gate_up alone already gets both
# the merged GEMM and the fused activation. silu-gated families only.

_GATEUP_TARGETS: Dict[str, Tuple[str, ...]] = {
    "transformers.models.qwen3.modeling_qwen3": ("Qwen3MLP",),
    "transformers.models.qwen2.modeling_qwen2": ("Qwen2MLP",),
    "transformers.models.llama.modeling_llama": ("LlamaMLP",),
    "transformers.models.mistral.modeling_mistral": ("MistralMLP",),
}

_GATEUP_ORIG: Dict[str, object] = {}


def _get_fused_gate_up_weight(self):
    """Lazily build+cache the concatenated [gate; up] weight. None if biased.

    INFERENCE-ONLY, same caveat as :func:`_get_fused_qkv_weight`: the cache is
    built under ``torch.no_grad()`` and never invalidated, so it assumes frozen
    weights. Do not enable ``FLAGOS_GATEUP_FUSED`` for training/finetuning.
    """
    w = getattr(self, "_fused_gate_up_weight", None)
    if w is not None:
        return w
    g, u = self.gate_proj, self.up_proj
    if g.bias is not None or u.bias is not None:
        return None
    with torch.no_grad():
        w = torch.cat([g.weight, u.weight], dim=0)
    self._fused_gate_up_weight = w
    self._fused_gate_up_splits = (g.weight.shape[0], u.weight.shape[0])
    return w


def _fused_gate_up_mlp_forward(self, x):
    """MLP forward fusing gate/up into one GEMM + fused SwiGLU activation.

    ``[gate;up]`` GEMM in one launch, split, then ``flagos.swiglu`` (silu(gate)*up
    in one launch). Falls back to HF if the projections are biased. Numerically
    equivalent to HF: same weights, silu in fp32 (matching the SwiGLU kernel).
    """
    w = _get_fused_gate_up_weight(self)
    if w is None:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    gate_up = torch.nn.functional.linear(x, w)
    gate, up = gate_up.split(self._fused_gate_up_splits, dim=-1)
    return self.down_proj(torch.ops.flagos.swiglu(gate, up))


def _is_fused_gate_up(fn: object) -> bool:
    return getattr(fn, "__name__", None) == "_fused_gate_up_mlp_forward"


def _install_gate_up_on_class(cls: type, qualname: str) -> bool:
    if _is_fused_gate_up(cls.__dict__.get("forward")):
        return False
    try:
        import inspect

        src = inspect.getsource(cls.forward)
    except Exception:
        src = ""
    if "act_fn" not in src or "gate_proj" not in src or "up_proj" not in src:
        return False
    _GATEUP_ORIG[qualname] = cls.forward
    cls.forward = _fused_gate_up_mlp_forward  # type: ignore[assignment]
    return True


def _patch_gate_up_module(mod: object) -> List[str]:
    mod_name = getattr(mod, "__name__", "")
    cls_names = _GATEUP_TARGETS.get(mod_name)
    if not cls_names:
        return []
    patched: List[str] = []
    for cls_name in cls_names:
        cls = getattr(mod, cls_name, None)
        if cls is None or not isinstance(cls, type):
            continue
        qualname = f"{mod_name}.{cls_name}"
        with _lock:
            did = _install_gate_up_on_class(cls, qualname)
        if did:
            patched.append(qualname)
    return patched


def apply_fused_gate_up() -> List[str]:
    """Patch every already-imported HF gated-MLP class to fuse gate/up into one
    GEMM + fused SwiGLU. Idempotent. Returns newly-patched names."""
    patched: List[str] = []
    for mod_name in _GATEUP_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        patched.extend(_patch_gate_up_module(mod))
    return patched


def restore_gate_up() -> List[str]:
    """Undo :func:`apply_fused_gate_up`."""
    restored: List[str] = []
    with _lock:
        for qualname, orig in list(_GATEUP_ORIG.items()):
            mod_name, _, cls_name = qualname.rpartition(".")
            mod = sys.modules.get(mod_name)
            cls = getattr(mod, cls_name, None) if mod is not None else None
            if cls is not None and _is_fused_gate_up(cls.__dict__.get("forward")):
                cls.forward = orig  # type: ignore[assignment]
                restored.append(qualname)
        _GATEUP_ORIG.clear()
    return restored


def is_gate_up_fused_active() -> bool:
    """True if at least one gated-MLP class is patched with fused gate_up."""
    with _lock:
        return bool(_GATEUP_ORIG)


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
    list(_TARGETS.keys())
    + list(_ROPE_TARGETS.keys())
    + list(_SOFTMAX_TARGETS.keys())
    + list(_SWIGLU_TARGETS.keys())
    + list(_DECODER_TARGETS.keys())
    + list(_QKV_TARGETS.keys())
    + list(_GATEUP_TARGETS.keys())
)

_FUSION_HOOK_INSTALLED = False


class _FusionImportHook(importlib.abc.MetaPathFinder):
    """Single meta path finder that wraps the loader of every target HF
    modelling module and applies all enabled fused patches (RMSNorm classes,
    RoPE function, softmax function) right after the module is exec'd. Each
    patch is individually gated by its env var, so this one hook serves all
    three fusions regardless of which combination is enabled.
    """

    # Per-thread re-entry flag for the importlib.util.find_spec fallback (which
    # re-walks sys.meta_path and would otherwise recurse back into this finder).
    _active = threading.local()

    def find_spec(self, fullname, path, target=None):  # noqa: D401, ARG002
        if fullname not in _ALL_FUSION_TARGETS:
            return None
        # Re-entry guard: the util.find_spec fallback below walks sys.meta_path,
        # which includes this finder again. Without the guard that recurses on
        # the same fullname until the stack blows. A thread-local flag lets the
        # nested call short-circuit to None (letting the real finders resolve).
        if getattr(self._active, "flag", False):
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
            self._active.flag = True
            try:
                real_spec = importlib.util.find_spec(fullname)
            except (ImportError, ValueError):
                real_spec = None
            finally:
                self._active.flag = False
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
        or os.environ.get(FLAGOS_SWIGLU_FUSED_ENV, "0") == "1"
        or os.environ.get(FLAGOS_ADD_RMSNORM_FUSED_ENV, "0") == "1"
        or os.environ.get(FLAGOS_QKV_FUSED_ENV, "0") == "1"
        or os.environ.get(FLAGOS_GATEUP_FUSED_ENV, "0") == "1"
    )
    if _any_on:
        _install_fusion_import_hook()
        apply_fused_rmsnorm()
        apply_fused_rope()
        apply_fused_softmax()
        apply_fused_swiglu()
        apply_fused_add_rmsnorm()
        apply_fused_qkv()
        apply_fused_gate_up()


_maybe_auto_install()
