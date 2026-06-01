"""
Shared image-enhancement models for cross-notebook reuse.

Phase 2 defines Zero-DCE++, FFA-Net and Restormer inside its own kernel, so
Phase 3 (§3.3) and Phase 4 (§4.5) — which run in *separate* kernels — had no
way to apply the same enhancers to their degraded images. Phase 4 §4.5 already
looks for this module:

    from scripts import enhancers as _enh
    enhance_ffanet   = getattr(_enh, 'enhance_ffanet', None)
    enhance_zero_dce = getattr(_enh, 'enhance_zero_dce', None)
    enhance_restormer = getattr(_enh, 'enhance_restormer', None)

This file provides exactly those three callables. Each model is built lazily on
first use (so `import enhancers` stays cheap) and loads the SAME pretrained
weights Phase 2 uses, with the same download fallbacks. If pretrained weights
cannot be fetched the function still runs (random init) and prints a clear
[WARN] — matching Phase 2's own behaviour — so callers never crash.

All three functions take and return a BGR uint8 image (OpenCV convention).
"""

import os
import glob
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Anchor everything to the repo root so behaviour is independent of cwd.
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy singletons (populated on first enhance_* call)
_zero_dce = None
_ffa_net = None
_restormer = None


# ────────────────────────────────────────────────────────────────────────────
# Zero-DCE++  (Li et al., TPAMI 2021) — low-light
# ────────────────────────────────────────────────────────────────────────────
class _CSDN(nn.Module):
    """Depthwise-separable conv block (CSDN_Tem in official Zero-DCE++)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class _ZeroDCEpp(nn.Module):
    """Official Zero-DCE++ architecture (matches the published Epoch99.pth)."""

    def __init__(self, scale_factor=1):
        super().__init__()
        n = 32
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.e_conv1 = _CSDN(3, n);    self.e_conv2 = _CSDN(n, n)
        self.e_conv3 = _CSDN(n, n);    self.e_conv4 = _CSDN(n, n)
        self.e_conv5 = _CSDN(n * 2, n); self.e_conv6 = _CSDN(n * 2, n)
        self.e_conv7 = _CSDN(n * 2, 3)

    def forward(self, x):
        xd = (F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear',
                            align_corners=True)
              if self.scale_factor != 1 else x)
        x1 = self.relu(self.e_conv1(xd))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor != 1:
            x_r = F.interpolate(x_r, size=x.shape[2:], mode='bilinear', align_corners=True)
        enhanced = x
        for _ in range(8):
            enhanced = enhanced + x_r * (torch.pow(enhanced, 2) - enhanced)
        return enhanced, x_r


def _load_zero_dce():
    model = _ZeroDCEpp(scale_factor=1).to(DEVICE)
    wt = WEIGHTS_DIR / "zero_dce_pp.pth"

    if not wt.exists() or wt.stat().st_size < 10_000:
        try:
            repo = "/tmp/zdce_repo"
            if not os.path.isdir(repo):
                subprocess.run(
                    ["git", "clone", "--depth", "1",
                     "https://github.com/Li-Chongyi/Zero-DCE_extension.git", repo],
                    check=True, capture_output=True, text=True,
                )
            cand = (glob.glob(f"{repo}/**/Epoch99.pth", recursive=True)
                    or glob.glob(f"{repo}/**/*.pth", recursive=True))
            if cand:
                shutil.copy(cand[0], wt)
        except Exception as e:
            print(f"[enhancers] Zero-DCE++ weight download failed: {e}")

    loaded = False
    if wt.exists() and wt.stat().st_size > 10_000:
        try:
            ckpt = torch.load(wt, map_location=DEVICE)
            state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
            missing, _ = model.load_state_dict(state, strict=False)
            loaded = len(missing) == 0
        except Exception as e:
            print(f"[enhancers] Zero-DCE++ load failed: {e}")
    if not loaded:
        print("[enhancers] [WARN] Zero-DCE++ using RANDOM weights (pretrained unavailable).")
    model.eval()
    return model


def enhance_zero_dce(img_bgr):
    """Low-light enhancement with Zero-DCE++. BGR uint8 in/out."""
    global _zero_dce
    if _zero_dce is None:
        _zero_dce = _load_zero_dce()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        enhanced, _ = _zero_dce(t)
    out = enhanced.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ────────────────────────────────────────────────────────────────────────────
# FFA-Net  (Qin et al., AAAI 2020) — dehazing
# ────────────────────────────────────────────────────────────────────────────
class _PALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.pa(x)


class _CALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.ca(self.avg_pool(x))


def _default_conv(in_c, out_c, k, bias=True):
    return nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=bias)


class _FFABlock(nn.Module):
    def __init__(self, conv, dim, k):
        super().__init__()
        self.conv1 = conv(dim, dim, k, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, k, bias=True)
        self.calayer = _CALayer(dim)
        self.palayer = _PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x)) + x
        res = self.conv2(res)
        res = self.palayer(self.calayer(res)) + x
        return res


class _FFAGroup(nn.Module):
    def __init__(self, conv, dim, k, blocks):
        super().__init__()
        mods = [_FFABlock(conv, dim, k) for _ in range(blocks)]
        mods.append(conv(dim, dim, k))
        self.gp = nn.Sequential(*mods)

    def forward(self, x):
        return self.gp(x) + x


class _FFANet(nn.Module):
    def __init__(self, gps=3, blocks=19):
        super().__init__()
        self.gps = gps
        self.dim = 64
        k = 3
        self.g1 = _FFAGroup(_default_conv, self.dim, k, blocks)
        self.g2 = _FFAGroup(_default_conv, self.dim, k, blocks)
        self.g3 = _FFAGroup(_default_conv, self.dim, k, blocks)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, bias=True), nn.Sigmoid())
        self.palayer = _PALayer(self.dim)
        self.pre = nn.Sequential(_default_conv(3, self.dim, k))
        self.post = nn.Sequential(_default_conv(self.dim, 3, k))

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        return self.post(out) + x1


def _load_ffanet():
    model = _FFANet(gps=3, blocks=19).to(DEVICE)
    wt = WEIGHTS_DIR / "ffa_net_outdoor.pk"

    if not wt.exists():
        # Method 1: Kaggle dataset (outdoor / OTS model)
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", "balraj98/ffanet-pretrained-weights",
                 "-p", str(WEIGHTS_DIR), "--unzip", "-q"],
                check=True, capture_output=True, text=True,
            )
            cand = (glob.glob(f"{WEIGHTS_DIR}/**/ots_train_ffa_3_19.pk", recursive=True)
                    or glob.glob(f"{WEIGHTS_DIR}/**/*ots*.pk", recursive=True)
                    or glob.glob(f"{WEIGHTS_DIR}/**/*.pk", recursive=True))
            if cand:
                shutil.copy(cand[0], wt)
        except Exception as e:
            print(f"[enhancers] FFA-Net Kaggle download failed: {e}")

    if not wt.exists():
        # Method 2: official Google Drive folder
        try:
            import gdown
            gdown.download_folder(
                "https://drive.google.com/drive/folders/19_lSUPrpEUJJBM3YgmkVtKJIOc-GaFJM",
                output=str(WEIGHTS_DIR / "ffa_pretrained"), quiet=True)
            cand = (glob.glob(f"{WEIGHTS_DIR}/ffa_pretrained/**/ots_train_ffa_3_19.pk", recursive=True)
                    or glob.glob(f"{WEIGHTS_DIR}/ffa_pretrained/**/*.pk", recursive=True))
            if cand:
                shutil.copy(cand[0], wt)
        except Exception as e:
            print(f"[enhancers] FFA-Net Google Drive download failed: {e}")

    if wt.exists():
        try:
            ckpt = torch.load(wt, map_location=DEVICE, weights_only=False)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                state = ckpt['model']
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt
            state = {k.replace('module.', '', 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[enhancers] [WARN] FFA-Net load failed ({e}); using RANDOM weights.")
    else:
        print("[enhancers] [WARN] FFA-Net using RANDOM weights (pretrained unavailable).")
    model.eval()
    return model


def enhance_ffanet(img_bgr):
    """Dehazing with FFA-Net. BGR uint8 in/out."""
    global _ffa_net
    if _ffa_net is None:
        _ffa_net = _load_ffanet()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    _, _, h, w = img_t.shape
    pad_h, pad_w = (4 - h % 4) % 4, (4 - w % 4) % 4
    if pad_h or pad_w:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.no_grad():
        out = _ffa_net(img_t)
    out = out[:, :, :h, :w].squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ────────────────────────────────────────────────────────────────────────────
# Restormer  (Zamir et al., CVPR 2022) — deraining / general restoration
# ────────────────────────────────────────────────────────────────────────────
def _load_restormer():
    repo = BASE_DIR / "Restormer"
    if not repo.exists():
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/swz30/Restormer.git", str(repo)],
                check=True, capture_output=True, text=True)
        except Exception as e:
            print(f"[enhancers] Restormer repo clone failed: {e}")
            return None

    arch_file = repo / "basicsr" / "models" / "archs" / "restormer_arch.py"
    if not arch_file.exists():
        print("[enhancers] [WARN] Restormer arch file not found; deraining disabled.")
        return None
    try:
        from runpy import run_path
        Restormer = run_path(str(arch_file))["Restormer"]
    except Exception as e:
        print(f"[enhancers] [WARN] Restormer arch import failed ({e}); deraining disabled.")
        return None

    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
        LayerNorm_type='WithBias', dual_pixel_task=False,
    ).to(DEVICE)

    wt = WEIGHTS_DIR / "restormer_deraining.pth"
    if not wt.exists() or wt.stat().st_size < 1_000_000:
        try:
            import gdown
            gdown.download(id="1FFpA2BOVM3LUJF64cPKMSrPBqFVSVOuf", output=str(wt), quiet=True)
        except Exception as e:
            print(f"[enhancers] Restormer gdown failed: {e}")
        if not wt.exists() or wt.stat().st_size < 1_000_000:
            try:
                from huggingface_hub import hf_hub_download
                for fname in ["Restormer_Rain100H.pth", "deraining.pth"]:
                    try:
                        p = hf_hub_download(repo_id="deepinv/Restormer", filename=fname,
                                            local_dir=str(WEIGHTS_DIR))
                        shutil.copy(p, wt)
                        if wt.stat().st_size > 1_000_000:
                            break
                    except Exception:
                        pass
            except Exception as e:
                print(f"[enhancers] Restormer HF download failed: {e}")

    if wt.exists() and wt.stat().st_size > 1_000_000:
        try:
            ckpt = torch.load(wt, map_location=DEVICE)
            state = ckpt.get('params', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[enhancers] [WARN] Restormer load failed ({e}); using RANDOM weights.")
    else:
        print("[enhancers] [WARN] Restormer using RANDOM weights (pretrained unavailable).")
    model.eval()
    return model


def enhance_restormer(img_bgr, tile_size=512, tile_overlap=32):
    """Deraining with Restormer (tiled inference to bound memory). BGR uint8 in/out."""
    global _restormer
    if _restormer is None:
        _restormer = _load_restormer()
    if _restormer is None:
        raise RuntimeError("Restormer unavailable (clone/weights failed).")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img_rgb.shape[:2]
    pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
    if pad_h or pad_w:
        img_rgb = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    H, W = img_t.shape[2:]
    output = torch.zeros_like(img_t)
    weight = torch.zeros((1, 1, H, W), device=DEVICE)

    stride = tile_size - tile_overlap
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, x1 = y, x
            y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            tile = img_t[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                enh_tile = _restormer(tile).clamp(0, 1)
            output[:, :, y1:y2, x1:x2] += enh_tile
            weight[:, :, y1:y2, x1:x2] += 1

    out = (output / weight.clamp(min=1))[:, :, :h, :w]
    out_np = out.squeeze(0).cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
