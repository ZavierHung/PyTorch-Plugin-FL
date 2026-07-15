"""
Fused-op correctness tests for the MetaX v2 opt-in fusion layer.

Covers the five custom ops registered by csrc/aten/register.cc and driven by
torch_fl.optimizations:
  - torch.ops.flagos.rms_norm
  - torch.ops.flagos.rope
  - torch.ops.flagos.masked_softmax
  - torch.ops.flagos.swiglu
  - torch.ops.flagos.add_rms_norm

Each op is checked against a CPU reference computed with the same fp32 math the
kernels use (reduce in fp32, keep I/O in the model dtype). These kernels only
have a MetaX v2 backend, so every test is marked ``metax`` and skipped
elsewhere by tests/integration/ops/conftest.py.

REGRESSION: the reduction kernels (rms_norm / masked_softmax / add_rms_norm)
clamp gridDim.x to kMaxGridV2 (65535) and grid-stride over rows. The
``*_large_rows`` cases push rows past 65535 to guard the tail rows that an
earlier ``row = blockIdx.x; if (row >= rows) return`` layout left uncomputed.

Usage:
    pytest tests/integration/ops/test_fused_ops_dispatch.py -v
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

# rows > kMaxGridV2 (65535) forces the grid-stride tail path. hidden is kept
# small so the tensor stays modest (65600 * 64 * 2B ~= 8 MiB).
_LARGE_ROWS = 65600
_SMALL_HIDDEN = 64


# --------------------------------------------------------------------------- #
# CPU references (fp32 reduction, I/O in model dtype — matches the kernels)
# --------------------------------------------------------------------------- #
def _ref_rms_norm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return (weight.float() * out).to(x.dtype)


def _ref_add_rms_norm(residual, hidden, weight, eps):
    # HF rounds the residual add back to model dtype before the norm.
    new_res = (residual + hidden).to(hidden.dtype)
    return _ref_rms_norm(new_res, weight, eps), new_res


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _ref_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos_u = cos.unsqueeze(unsqueeze_dim).float()
    sin_u = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q.float() * cos_u) + (_rotate_half(q.float()) * sin_u)
    k_embed = (k.float() * cos_u) + (_rotate_half(k.float()) * sin_u)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def _ref_masked_softmax(attn, mask, scaling):
    s = attn.float() * scaling
    if mask is not None:
        s = s + mask.float()
    return torch.softmax(s, dim=-1).to(attn.dtype)


def _ref_swiglu(gate, up):
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)


# --------------------------------------------------------------------------- #
# rms_norm
# --------------------------------------------------------------------------- #
@pytest.mark.metax
class TestFusedRmsNorm:
    @pytest.mark.parametrize("shape", [(4, 1024), (2, 8, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_cpu(self, shape, dtype):
        torch.manual_seed(0)
        hidden = shape[-1]
        x_cpu = torch.randn(*shape, dtype=dtype)
        w_cpu = torch.randn(hidden, dtype=dtype)
        ref = _ref_rms_norm(x_cpu, w_cpu, 1e-6)
        out = torch.ops.flagos.rms_norm(x_cpu.to(DEVICE), w_cpu.to(DEVICE), 1e-6)
        assert out.dtype == dtype
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)

    def test_large_rows(self):
        """rows > kMaxGridV2 exercises the grid-stride tail (regression)."""
        torch.manual_seed(1)
        x_cpu = torch.randn(_LARGE_ROWS, _SMALL_HIDDEN, dtype=torch.float16)
        w_cpu = torch.randn(_SMALL_HIDDEN, dtype=torch.float16)
        ref = _ref_rms_norm(x_cpu, w_cpu, 1e-6)
        out = torch.ops.flagos.rms_norm(x_cpu.to(DEVICE), w_cpu.to(DEVICE), 1e-6)
        # The last row is the one an un-strided kernel would have skipped.
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)
        assert torch.isfinite(out.cpu()[-1]).all()


# --------------------------------------------------------------------------- #
# add_rms_norm
# --------------------------------------------------------------------------- #
@pytest.mark.metax
class TestFusedAddRmsNorm:
    @pytest.mark.parametrize("shape", [(4, 1024), (2, 8, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_cpu(self, shape, dtype):
        torch.manual_seed(2)
        hidden = shape[-1]
        res_cpu = torch.randn(*shape, dtype=dtype)
        hid_cpu = torch.randn(*shape, dtype=dtype)
        w_cpu = torch.randn(hidden, dtype=dtype)
        ref_normed, ref_res = _ref_add_rms_norm(res_cpu, hid_cpu, w_cpu, 1e-6)
        normed, new_res = torch.ops.flagos.add_rms_norm(
            res_cpu.to(DEVICE), hid_cpu.to(DEVICE), w_cpu.to(DEVICE), 1e-6
        )
        torch.testing.assert_close(normed.cpu(), ref_normed, rtol=2e-2, atol=2e-2)
        # new_residual must match bit-for-bit (fp16 add rounded to dtype).
        torch.testing.assert_close(new_res.cpu(), ref_res, rtol=0, atol=0)

    def test_large_rows(self):
        torch.manual_seed(3)
        res_cpu = torch.randn(_LARGE_ROWS, _SMALL_HIDDEN, dtype=torch.float16)
        hid_cpu = torch.randn(_LARGE_ROWS, _SMALL_HIDDEN, dtype=torch.float16)
        w_cpu = torch.randn(_SMALL_HIDDEN, dtype=torch.float16)
        ref_normed, ref_res = _ref_add_rms_norm(res_cpu, hid_cpu, w_cpu, 1e-6)
        normed, new_res = torch.ops.flagos.add_rms_norm(
            res_cpu.to(DEVICE), hid_cpu.to(DEVICE), w_cpu.to(DEVICE), 1e-6
        )
        torch.testing.assert_close(normed.cpu(), ref_normed, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(new_res.cpu(), ref_res, rtol=0, atol=0)
        assert torch.isfinite(normed.cpu()[-1]).all()


# --------------------------------------------------------------------------- #
# masked_softmax
# --------------------------------------------------------------------------- #
@pytest.mark.metax
class TestFusedMaskedSoftmax:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_cpu_with_mask(self, dtype):
        torch.manual_seed(4)
        b, h, q, k = 2, 4, 8, 16
        attn = torch.randn(b, h, q, k, dtype=dtype)
        # Causal-style additive mask, broadcast over heads ([b, 1, q, k]).
        mask = torch.zeros(b, 1, q, k, dtype=dtype)
        mask[..., q // 2:] = float("-inf")
        scaling = 1.0 / (k ** 0.5)
        ref = _ref_masked_softmax(attn, mask, scaling)
        out = torch.ops.flagos.masked_softmax(
            attn.to(DEVICE), mask.to(DEVICE), scaling
        )
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_cpu_no_mask(self, dtype):
        torch.manual_seed(5)
        attn = torch.randn(2, 4, 8, 16, dtype=dtype)
        scaling = 0.125
        ref = _ref_masked_softmax(attn, None, scaling)
        out = torch.ops.flagos.masked_softmax(attn.to(DEVICE), None, scaling)
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)
        sums = out.float().sum(dim=-1).cpu()
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2)

    def test_large_rows(self):
        """rows = b*h*q > kMaxGridV2 exercises the grid-stride tail."""
        torch.manual_seed(6)
        # 66 * 32 * 32 = 67584 rows > 65535.
        b, h, q, k = 66, 32, 32, _SMALL_HIDDEN
        attn = torch.randn(b, h, q, k, dtype=torch.float16)
        scaling = 1.0 / (k ** 0.5)
        ref = _ref_masked_softmax(attn, None, scaling)
        out = torch.ops.flagos.masked_softmax(attn.to(DEVICE), None, scaling)
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)
        # Last row would be uncomputed (uninitialized) without grid-stride.
        assert torch.isfinite(out.cpu().reshape(-1, k)[-1]).all()


# --------------------------------------------------------------------------- #
# rope
# --------------------------------------------------------------------------- #
@pytest.mark.metax
class TestFusedRoPE:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_cpu(self, dtype):
        torch.manual_seed(7)
        b, h, s, d = 2, 4, 8, 64
        q = torch.randn(b, h, s, d, dtype=dtype)
        k = torch.randn(b, h, s, d, dtype=dtype)
        # HF cos/sin shape [b, s, d]; kernel unsqueezes dim 1 -> [b, 1, s, d].
        cos = torch.randn(b, s, d, dtype=torch.float32)
        sin = torch.randn(b, s, d, dtype=torch.float32)
        ref_q, ref_k = _ref_rope(q, k, cos, sin, unsqueeze_dim=1)
        out_q, out_k = torch.ops.flagos.rope(
            q.to(DEVICE), k.to(DEVICE), cos.to(DEVICE), sin.to(DEVICE), 1
        )
        torch.testing.assert_close(out_q.cpu(), ref_q, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_k.cpu(), ref_k, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# swiglu
# --------------------------------------------------------------------------- #
@pytest.mark.metax
class TestFusedSwiGLU:
    @pytest.mark.parametrize("shape", [(4, 2048), (2, 8, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_cpu(self, shape, dtype):
        torch.manual_seed(8)
        gate = torch.randn(*shape, dtype=dtype)
        up = torch.randn(*shape, dtype=dtype)
        ref = _ref_swiglu(gate, up)
        out = torch.ops.flagos.swiglu(gate.to(DEVICE), up.to(DEVICE))
        assert out.dtype == dtype
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)
