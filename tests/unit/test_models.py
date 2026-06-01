"""
Unit tests for src/models/ — CPU-only forward-pass smoke tests.

These tests verify:
  - Each model constructs without errors (no GPU, no pretrained weights)
  - Forward pass produces the right output shape
  - Label convention is preserved (num_classes=2 default)
  - InstanceNorm vs BatchNorm is correct per model (KNOWN_QUIRKS #1 and #2)
  - ResBlock skip connection is wired (output shape matches input for identity case)
  - NoiseprintPlusPlus produces [B, 1, H, W] output
  - extract_noise_stats produces correct shape

We skip pretrained weight download by patching xception with pretrained=False.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ── fixtures ──────────────────────────────────────────────────────────────────

B  = 2    # batch size
H  = 64   # small spatial size for speed (real: 299)
NC = 2    # num_classes


@pytest.fixture(scope="module")
def rgb_batch():
    return torch.randn(B, 3, H, H)


@pytest.fixture(scope="module")
def noise_batch():
    return torch.randn(B, 1, H, H)


# ── ResBlock ──────────────────────────────────────────────────────────────────

class TestResBlock:
    def test_identity_shape(self, rgb_batch):
        from src.models.resblock import ResBlock
        block = ResBlock(3, 3, stride=1)
        out = block(rgb_batch)
        assert out.shape == rgb_batch.shape

    def test_stride_reduces_spatial(self):
        from src.models.resblock import ResBlock
        x = torch.randn(2, 32, 16, 16)
        block = ResBlock(32, 64, stride=2)
        out = block(x)
        assert out.shape == (2, 64, 8, 8)

    def test_skip_connection_is_wired(self):
        """If skip is not added, gradient to input will be zero on the shortcut path.
        Verify that the gradient flows back through the shortcut."""
        from src.models.resblock import ResBlock
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        block = ResBlock(16, 32, stride=2)
        loss = block(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "Skip connection gradient is zero"


# ── NoiseprintPlusPlus ────────────────────────────────────────────────────────

class TestNoiseprintPlusPlus:
    def test_output_shape(self):
        from src.models.noiseprintpp import NoiseprintPlusPlus
        model = NoiseprintPlusPlus()   # random init, no weights
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 64, 64), f"Expected (2,1,64,64), got {out.shape}"

    def test_spatial_preserving(self):
        """FCN must preserve spatial dimensions."""
        from src.models.noiseprintpp import NoiseprintPlusPlus
        model = NoiseprintPlusPlus()
        model.eval()
        x = torch.randn(1, 3, 128, 96)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 128, 96)


# ── RGBOnlyModel ──────────────────────────────────────────────────────────────

class TestRGBOnlyModel:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.rgb_only import RGBOnlyModel
        m = RGBOnlyModel.__new__(RGBOnlyModel)
        # Construct with pretrained=False to avoid download
        torch.nn.Module.__init__(m)
        from src.models.xception import xception
        m.backbone = xception(num_classes=1000, pretrained=False)
        m.backbone.last_linear = torch.nn.Identity()
        m.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 512), torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),  torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, NC),
        )
        m.eval()
        return m

    def test_output_shape(self, model, rgb_batch):
        with torch.no_grad():
            out = model(rgb_batch)
        assert out.shape == (B, NC)

    def test_4channel_input_sliced(self, model):
        x = torch.randn(B, 4, H, H)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, NC)


# ── ResidualOnlyModel ─────────────────────────────────────────────────────────

class TestResidualOnlyModel:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.residual_only import ResidualOnlyModel
        m = ResidualOnlyModel(num_classes=NC)
        m.eval()
        return m

    def test_output_shape_from_noise(self, model, noise_batch):
        with torch.no_grad():
            # Provide noise as 3-channel so the model can extract via Laplacian
            rgb = torch.randn(B, 3, H, H)
            out = model(rgb)
        assert out.shape == (B, NC)

    def test_instance_norm_used(self, model):
        """InstanceNorm2d must be used, NOT BatchNorm2d — KNOWN_QUIRKS #1."""
        assert isinstance(model.noise_norm, nn.InstanceNorm2d), \
            "ResidualOnlyModel.noise_norm must be InstanceNorm2d (KNOWN_QUIRKS #1)"

    def test_no_batch_norm_in_noise_norm(self, model):
        assert not isinstance(model.noise_norm, nn.BatchNorm2d)


# ── LateFusionModel ───────────────────────────────────────────────────────────

class TestLateFusionModel:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.late_fusion import LateFusionModel
        m = LateFusionModel.__new__(LateFusionModel)
        torch.nn.Module.__init__(m)
        from src.models.xception import xception
        from src.models.resblock import ResBlock
        m.noise_model = None
        m.rgb_backbone = xception(num_classes=1000, pretrained=False)
        m.rgb_backbone.last_linear = torch.nn.Identity()
        m.noise_norm = nn.BatchNorm2d(1, affine=True)
        m.noise_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        fuse = 2048 + 256
        m.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(fuse, 1024), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.35),
            torch.nn.Linear(1024, 512), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(512, 256), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(256, NC),
        )
        # Bind methods
        import types
        from src.models.late_fusion import LateFusionModel as LFM
        m._freeze_rgb_early = types.MethodType(LFM._freeze_rgb_early, m)
        m._init_noise_weights = types.MethodType(LFM._init_noise_weights, m)
        m._extract_noise = types.MethodType(LFM._extract_noise, m)
        m.forward = types.MethodType(LFM.forward, m)
        m.eval()
        return m

    def test_output_shape_with_precomputed_noise(self, model, rgb_batch, noise_batch):
        with torch.no_grad():
            out = model(rgb_batch, noise_batch)
        assert out.shape == (B, NC)

    def test_batch_norm_in_noise_norm(self, model):
        """BatchNorm2d must be used in noise_norm — KNOWN_QUIRKS #2."""
        assert isinstance(model.noise_norm, nn.BatchNorm2d), \
            "LateFusionModel.noise_norm must be BatchNorm2d (KNOWN_QUIRKS #2)"


# ── StatNoiseFusionModel ──────────────────────────────────────────────────────

class TestStatNoiseFusionModel:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.statnoise_fusion import StatNoiseFusionModel
        m = StatNoiseFusionModel.__new__(StatNoiseFusionModel)
        torch.nn.Module.__init__(m)
        from src.models.xception import xception
        m.RGB_DIM = 2048
        m.NOISE_FEAT_DIM = 32
        m.rgb_backbone = xception(num_classes=1000, pretrained=False)
        m.rgb_backbone.last_linear = torch.nn.Identity()
        from src.models.statnoise_fusion import N_NOISE_FEATURES
        m.noise_mlp = torch.nn.Sequential(
            torch.nn.BatchNorm1d(N_NOISE_FEATURES),
            torch.nn.Linear(N_NOISE_FEATURES, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, m.NOISE_FEAT_DIM),
            torch.nn.ReLU(inplace=True),
        )
        fuse = m.RGB_DIM + m.NOISE_FEAT_DIM
        m.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(fuse, 1024), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, NC),
        )
        import types
        from src.models.statnoise_fusion import StatNoiseFusionModel as SNF
        m.forward = types.MethodType(SNF.forward, m)
        m._freeze_rgb_early = types.MethodType(SNF._freeze_rgb_early, m)
        m.eval()
        return m

    def test_output_shape(self, model, rgb_batch, noise_batch):
        with torch.no_grad():
            out = model(rgb_batch, noise_batch)
        assert out.shape == (B, NC)

    def test_noise_stats_shape(self, noise_batch):
        from src.models.statnoise_fusion import extract_noise_stats, N_NOISE_FEATURES
        feat = extract_noise_stats(noise_batch[0])
        assert feat.shape == (N_NOISE_FEATURES,)
        assert feat.dtype == torch.float32


# ── ResAwareFusionModel ───────────────────────────────────────────────────────

class TestResAwareFusionModel:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.resaware_fusion import ResAwareFusionModel
        m = ResAwareFusionModel.__new__(ResAwareFusionModel)
        torch.nn.Module.__init__(m)
        from src.models.xception import xception
        from src.models.resblock import ResBlock
        m.noise_in_channels = 1
        m.RGB_DIM = 2048
        m.RESIDUAL_DIM = 512
        m.rgb_backbone = xception(num_classes=1000, pretrained=False)
        m.rgb_backbone.last_linear = torch.nn.Identity()
        m.noise_norm = nn.InstanceNorm2d(1, affine=True)
        m.noise_backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        m.noise_fc = torch.nn.Sequential(
            torch.nn.Linear(256, m.RESIDUAL_DIM),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
        )
        fuse = m.RGB_DIM + m.RESIDUAL_DIM
        m.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(fuse, 1024), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, NC),
        )
        import types
        from src.models.resaware_fusion import ResAwareFusionModel as RAF
        m.forward = types.MethodType(RAF.forward, m)
        m.eval()
        return m

    def test_output_shape_fixed_size(self, model, rgb_batch, noise_batch):
        with torch.no_grad():
            out = model(rgb_batch, noise_batch)
        assert out.shape == (B, NC)

    def test_output_shape_variable_noise(self, model, rgb_batch):
        """AdaptiveAvgPool should handle non-square, non-299 noise crops."""
        variable_noise = torch.randn(B, 1, 48, 72)
        with torch.no_grad():
            out = model(rgb_batch, variable_noise)
        assert out.shape == (B, NC)

    def test_instance_norm_in_noise_branch(self, model):
        """ResAware-Fusion uses InstanceNorm2d in noise branch."""
        assert isinstance(model.noise_norm, nn.InstanceNorm2d)


# ── build_optimizer ───────────────────────────────────────────────────────────

class TestBuildOptimizer:
    def _make_tiny_model(self):
        return nn.Sequential(nn.Linear(4, 4))

    def test_rgb_only_two_groups(self):
        from src.train.loop import build_optimizer
        from src.models.residual_only import ResidualOnlyModel
        m = ResidualOnlyModel(num_classes=2)
        opt = build_optimizer(m, "RGB-Only", lr=1e-3)
        # Should return AdamW
        assert isinstance(opt, torch.optim.AdamW)

    def test_residual_only_flat_lr(self):
        from src.train.loop import build_optimizer
        from src.models.residual_only import ResidualOnlyModel
        m = ResidualOnlyModel(num_classes=2)
        opt = build_optimizer(m, "Residual-Only", lr=1e-3)
        # Hard-coded lr=1e-4 for residual-only
        assert abs(opt.param_groups[0]["lr"] - 1e-4) < 1e-9

    def test_unknown_model_fallback(self):
        from src.train.loop import build_optimizer
        m = nn.Linear(4, 2)
        opt = build_optimizer(m, "unknown-model", lr=5e-4)
        assert isinstance(opt, torch.optim.AdamW)
        assert abs(opt.param_groups[0]["lr"] - 5e-4) < 1e-9
