import pytest
import torch

from nvfp4_gemm_cute import Config, gemm, quantize_nvfp4, supported
from nvfp4_gemm_cute.testing import check, make_inputs, reference

pytestmark = pytest.mark.skipif(not supported(), reason="needs sm_103 and cutlass-dsl>=4.8")


@pytest.mark.parametrize("m,n,k", [(512, 512, 512), (1000, 1024, 768), (2048, 1536, 4096)])
def test_default_config(m, n, k):
    inp = make_inputs(m, n, k)
    out = gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"])
    ok, err = check(out, reference(inp))
    assert ok, err


@pytest.mark.parametrize(
    "cfg",
    [
        Config.make(cluster=(4, 1), fallback=None, fast_swz=2, raster_along_m=False),
        Config.make(cluster=(8, 2), fallback=(2, 1), fast_swz=4, l2_policy_a="evict_last"),
        Config.make(cluster=(2, 2), fallback=None, epi_store_pace_per_ktile_ns=0),
    ],
)
def test_configs(cfg):
    inp = make_inputs(2048, 2048, 2048)
    out = gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], config=cfg)
    ok, err = check(out, reference(inp))
    assert ok, err


def test_fast_swz_falls_back_when_m_not_divisible():
    # 3 M-clusters of 4x1 (M=1536) cannot be grouped by fast_swz=2.
    cfg = Config.make(cluster=(4, 1), fallback=None, fast_swz=2)
    inp = make_inputs(1536, 1024, 1024)
    with pytest.warns(UserWarning, match="fast_swz"):
        out = gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], config=cfg)
    ok, err = check(out, reference(inp))
    assert ok, err


def test_bf16_output():
    inp = make_inputs(1024, 1024, 1024)
    out = gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], out_dtype=torch.bfloat16)
    assert out.dtype == torch.bfloat16
    ok, err = check(out, reference(inp), rel_tol=8e-3)
    assert ok, err


def test_owned_8k_rejects_other_shapes():
    from nvfp4_gemm_cute import config_for

    inp = make_inputs(512, 512, 512)
    with pytest.raises(ValueError, match="M=N=K=8192"):
        gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], config=config_for(8192, 8192, 8192))


def test_quantizer_matches_flashinfer():
    fi = pytest.importorskip("flashinfer")
    x = torch.randn(256, 1024, device="cuda").to(torch.bfloat16)
    q, sf, _ = quantize_nvfp4(x)
    one = torch.ones(1, device="cuda")
    fq, fsf = fi.nvfp4_quantize(x, one, sfLayout=fi.SfLayout.layout_128x4, do_shuffle=False)
    # Scale factors and the 128x4 swizzle match exactly. FP4 codes differ only on exact rounding
    # ties (x/s = 1.25, 2.5, 5): we round to even, flashinfer's reciprocal-multiply lands above.
    assert torch.equal(sf, fsf.view(torch.uint8).reshape(-1))
    assert (q != fq.view(torch.uint8)).float().mean() < 2e-3


@pytest.mark.parametrize("size", [4096, 8192, 16384, 32768])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_presets_scaled_output(size, dtype):
    """Exercise tail pipeline wraparound and descriptor reuse across persistent tiles."""
    from nvfp4_gemm_cute import config_for, persisting_l2

    inp = make_inputs(size, size, size, seed=2)
    cfg = config_for(size, size, size)
    previous = Config.make(
        cluster=(8, 2),
        fallback=(2, 1),
        persist_mb=40,
        fast_swz=4,
        raster_along_m=False,
        l2_policy_a="evict_last",
        l2_policy_sfa="evict_last",
    )
    alpha = torch.tensor([-0.375], device="cuda", dtype=torch.float32)
    args = (inp["a"], inp["b"], inp["a_sf"], inp["b_sf"])
    with persisting_l2(cfg.persist_mb):
        expected = gemm(*args, alpha=alpha, out_dtype=dtype, config=previous)
        actual = gemm(*args, alpha=alpha, out_dtype=dtype, config=cfg)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_16k_preset_rejects_other_shapes():
    from nvfp4_gemm_cute import config_for

    inp = make_inputs(512, 512, 512)
    with pytest.raises(ValueError, match="M=N=K=16384"):
        gemm(inp["a"], inp["b"], inp["a_sf"], inp["b_sf"], config=config_for(16384, 16384, 16384))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cluster": (4, 1)}, "cluster="),
        ({"attrs": (("packed_ab_desc", True), ("swap_ab", True))}, "unsupported kernel attributes"),
    ],
)
def test_16k_preset_rejects_incompatible_config(overrides, message):
    from dataclasses import replace

    from nvfp4_gemm_cute import config_for

    # Validation happens before compilation or GPU access; expanded views avoid large allocations.
    a = torch.empty(1, dtype=torch.uint8).expand(16384, 8192)
    sf = torch.empty(1, dtype=torch.uint8)
    cfg = replace(config_for(16384, 16384, 16384), **overrides)
    with pytest.raises(ValueError, match=message):
        gemm(a, a, sf, sf, config=cfg)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"shape": 512}, "M=N=K=32768"),
        ({"cluster": (8, 2)}, "cluster="),
    ],
)
def test_32k_preset_guards(overrides, message):
    from dataclasses import replace
    from nvfp4_gemm_cute import config_for

    size = overrides.get("shape", 32768)
    a = torch.empty(1, dtype=torch.uint8).expand(size, size // 2)
    sf = torch.empty(1, dtype=torch.uint8)
    cfg = config_for(32768, 32768, 32768)
    if "cluster" in overrides:
        cfg = replace(cfg, cluster=overrides["cluster"])
    with pytest.raises(ValueError, match=message):
        gemm(a, a, sf, sf, config=cfg)
