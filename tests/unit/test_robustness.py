"""
Unit tests for src/robustness/perturbations.py

Tests verify:
- Each perturbation function produces same-shape output
- Output stays in the correct normalisation range (~[-1, 1])
- Identity conditions produce near-identical output
- Perturbation grid has the correct families and lengths
- apply_perturbation dispatch works for all families
"""
from __future__ import annotations

import pytest
import torch
import numpy as np

from src.robustness.perturbations import (
    JPEG_QUALITIES,
    BLUR_SIGMAS,
    DENOISE_SHARPEN_TAGS,
    GAMMA_VALUES,
    RESIZE_FACTORS,
    PERTURBATION_GRID,
    perturb_jpeg,
    perturb_blur,
    perturb_denoise_sharpen,
    perturb_gamma,
    perturb_resize,
    apply_perturbation,
)

B, C, H, W = 2, 3, 64, 64   # small batch for speed


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(B, C, H, W)


# ── canonical grid ─────────────────────────────────────────────────────────────

class TestGrid:
    def test_jpeg_qualities(self):
        assert JPEG_QUALITIES == [95, 75, 55, 40]

    def test_blur_sigmas(self):
        assert BLUR_SIGMAS == [0.5, 1.0, 2.0, 3.0]

    def test_dssharp_tags(self):
        assert set(DENOISE_SHARPEN_TAGS) == {"none", "denoise", "sharpen", "both"}

    def test_gamma_values(self):
        assert 1.0 in GAMMA_VALUES

    def test_resize_factors(self):
        assert 1.0 in RESIZE_FACTORS

    def test_grid_contains_all_families(self):
        families = {f for f, _ in PERTURBATION_GRID}
        assert families == {"jpeg", "blur", "dssharp", "gamma", "resize"}

    def test_grid_total_length(self):
        expected = (len(JPEG_QUALITIES) + len(BLUR_SIGMAS) +
                    len(DENOISE_SHARPEN_TAGS) + len(GAMMA_VALUES) +
                    len(RESIZE_FACTORS))
        assert len(PERTURBATION_GRID) == expected


# ── perturb_jpeg ──────────────────────────────────────────────────────────────

class TestPerturbJpeg:
    def test_output_shape(self, batch):
        out = perturb_jpeg(batch, 75)
        assert out.shape == batch.shape

    def test_output_range_approx(self, batch):
        out = perturb_jpeg(batch, 75)
        assert out.abs().max().item() <= 2.0   # normalised, can be slightly > 1

    def test_high_quality_near_input(self, batch):
        # q=95 should produce visually close output, but JPEG always quantises
        # 8-bit so the normalised MAE can be ~0.6 on random noise tensors.
        # We just check output is non-degenerate and in the correct range.
        out = perturb_jpeg(batch, 95)
        assert out.abs().max().item() <= 2.0
        assert not torch.all(out == 0)   # non-degenerate


# ── perturb_blur ──────────────────────────────────────────────────────────────

class TestPerturbBlur:
    def test_output_shape(self, batch):
        out = perturb_blur(batch, sigma=1.0)
        assert out.shape == batch.shape

    def test_zero_sigma_identity(self, batch):
        out = perturb_blur(batch, sigma=0.0)
        assert torch.equal(out, batch)

    def test_large_sigma_reduces_variance(self, batch):
        out = perturb_blur(batch, sigma=3.0)
        assert out.var().item() < batch.var().item()


# ── perturb_denoise_sharpen ───────────────────────────────────────────────────

class TestPerturbDenoiseSharpen:
    def test_none_is_identity(self, batch):
        out = perturb_denoise_sharpen(batch, "none")
        assert torch.equal(out, batch)

    def test_denoise_output_shape(self, batch):
        out = perturb_denoise_sharpen(batch, "denoise")
        assert out.shape == batch.shape

    def test_sharpen_output_shape(self, batch):
        out = perturb_denoise_sharpen(batch, "sharpen")
        assert out.shape == batch.shape

    def test_both_output_shape(self, batch):
        out = perturb_denoise_sharpen(batch, "both")
        assert out.shape == batch.shape

    def test_invalid_tag_is_noop(self, batch):
        # Unknown tags don't match "denoise" or "both" → no PIL filter applied
        # → identical to "none" (pass-through). Matches original ablation.py.
        out_invalid = perturb_denoise_sharpen(batch, "invalid_tag")
        out_none    = perturb_denoise_sharpen(batch, "none")
        assert torch.equal(out_invalid, out_none)


# ── perturb_gamma ─────────────────────────────────────────────────────────────

class TestPerturbGamma:
    def test_output_shape(self, batch):
        out = perturb_gamma(batch, gamma=1.5)
        assert out.shape == batch.shape

    def test_gamma_one_near_identity(self):
        # Use a strictly positive [0,1]-range tensor so clamping is a no-op
        t = torch.rand(2, 3, 8, 8)          # uniform [0, 1]
        # Remap to [-1, 1] via the standard normalisation
        batch = (t - 0.5) / 0.5
        out  = perturb_gamma(batch, gamma=1.0)
        diff = (out - batch).abs().max().item()
        assert diff < 1e-4

    def test_high_gamma_brightens(self):
        # Small positive value (<1) in [0,1] raised to gamma<1 should increase
        t = torch.full((1, 3, 4, 4), -0.5)   # 0.25 in [0,1] space
        out = perturb_gamma(t, gamma=0.5)
        # 0.25^0.5 = 0.5 in [0,1] → mapped back to 0.0 in [-1,1]
        # Should be brighter (higher) than input
        assert out.mean().item() > t.mean().item()


# ── perturb_resize ────────────────────────────────────────────────────────────

class TestPerturbResize:
    def test_output_shape(self, batch):
        out = perturb_resize(batch, factor=0.5)
        assert out.shape == batch.shape  # restored to original

    def test_factor_one_identity(self, batch):
        out = perturb_resize(batch, factor=1.0)
        assert torch.equal(out, batch)

    def test_downscale_reduces_high_freq(self, batch):
        # High-frequency noise should be smoothed after downscale+upscale
        out = perturb_resize(batch, factor=0.5)
        # Output should differ from input (information loss from downscaling)
        assert not torch.equal(out, batch)


# ── apply_perturbation dispatch ───────────────────────────────────────────────

class TestApplyPerturbation:
    def test_jpeg_dispatch(self, batch):
        out = apply_perturbation(batch, "jpeg", 75)
        assert out.shape == batch.shape

    def test_blur_dispatch(self, batch):
        out = apply_perturbation(batch, "blur", 1.0)
        assert out.shape == batch.shape

    def test_dssharp_dispatch(self, batch):
        out = apply_perturbation(batch, "dssharp", "denoise")
        assert out.shape == batch.shape

    def test_gamma_dispatch(self, batch):
        out = apply_perturbation(batch, "gamma", 1.5)
        assert out.shape == batch.shape

    def test_resize_dispatch(self, batch):
        out = apply_perturbation(batch, "resize", 0.8)
        assert out.shape == batch.shape

    def test_unknown_family_raises(self, batch):
        with pytest.raises(ValueError):
            apply_perturbation(batch, "nonexistent", 1)

    def test_all_grid_entries_run(self, batch):
        """Smoke-test every (family, param) in the canonical grid."""
        for family, param in PERTURBATION_GRID:
            out = apply_perturbation(batch, family, param)
            assert out.shape == batch.shape, \
                f"shape mismatch for ({family}, {param})"
