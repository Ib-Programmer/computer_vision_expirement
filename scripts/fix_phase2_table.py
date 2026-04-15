"""Apply the 3 fixes from the summary table to Phase2_Image_Enhancement.ipynb."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "Phase2_Image_Enhancement.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

# ── Locate cells ────────────────────────────────────────────────────────────
c5 = c8 = c13 = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "geekyrakshit/LoL-Dataset" in src and "exdark_images" in src:
        c5 = i
    if "ZeroDCEpp" in src and "gdown" in src:
        c8 = i
    if "ENHANCED_DIR" in src and "saved[name]" in src:
        c13 = i

assert c5 is not None, "Cell 05 not found"
assert c8 is not None, "Cell 08 not found"
assert c13 is not None, "Cell 13 not found"
print(f"Cells: 05={c5}  08={c8}  13={c13}")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — Cell 08: Remove gdown, use HuggingFace as primary
# ═══════════════════════════════════════════════════════════════════════════
CELL8 = (
    "# -- Zero-DCE++ (Low-Light Enhancement) --\n"
    "# PRIMARY:  HuggingFace pretrained weights (gdown/Google Drive removed --\n"
    "#           Drive file permissions break frequently without warning)\n"
    "# FALLBACK: train on RTTS images (zero-reference, no paired data needed)\n"
    "\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torchvision import transforms\n"
    "import os, glob as _glob, shutil\n"
    "\n"
    "class CSDN(nn.Module):\n"
    "    def __init__(self, in_ch, out_ch):\n"
    "        super().__init__()\n"
    "        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)\n"
    "        self.point_conv = nn.Conv2d(in_ch, out_ch, 1)\n"
    "    def forward(self, x):\n"
    "        return self.point_conv(self.depth_conv(x))\n"
    "\n"
    "class ZeroDCEpp(nn.Module):\n"
    "    def __init__(self, scale_factor=1):\n"
    "        super().__init__()\n"
    "        n = 32\n"
    "        self.relu = nn.ReLU(inplace=True)\n"
    "        self.scale_factor = scale_factor\n"
    "        self.e_conv1 = CSDN(3, n);  self.e_conv2 = CSDN(n, n)\n"
    "        self.e_conv3 = CSDN(n, n);  self.e_conv4 = CSDN(n, n)\n"
    "        self.e_conv5 = CSDN(n*2, n); self.e_conv6 = CSDN(n*2, n)\n"
    "        self.e_conv7 = CSDN(n*2, 24)\n"
    "\n"
    "    def forward(self, x):\n"
    "        xd = F.interpolate(x, scale_factor=1/self.scale_factor,\n"
    "                           mode='bilinear', align_corners=True) if self.scale_factor != 1 else x\n"
    "        x1 = self.relu(self.e_conv1(xd))\n"
    "        x2 = self.relu(self.e_conv2(x1))\n"
    "        x3 = self.relu(self.e_conv3(x2))\n"
    "        x4 = self.relu(self.e_conv4(x3))\n"
    "        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))\n"
    "        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))\n"
    "        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))\n"
    "        if self.scale_factor != 1:\n"
    "            x_r = F.interpolate(x_r, size=x.shape[2:], mode='bilinear', align_corners=True)\n"
    "        enhanced = x\n"
    "        for curve in torch.split(x_r, 3, dim=1):\n"
    "            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)\n"
    "        return enhanced, x_r\n"
    "\n"
    "def spatial_consistency_loss(enhanced, original, k=4):\n"
    "    pool = nn.AvgPool2d(k)\n"
    "    return torch.mean(torch.pow(pool(enhanced.mean(1, keepdim=True)) -\n"
    "                                pool(original.mean(1, keepdim=True)), 2))\n"
    "\n"
    "def color_constancy_loss(img):\n"
    "    m = torch.mean(img, dim=[2, 3])\n"
    "    return torch.mean((m[:,0]-m[:,1])**2 + (m[:,0]-m[:,2])**2 + (m[:,1]-m[:,2])**2)\n"
    "\n"
    "def exposure_loss(img, target_E=0.6):\n"
    "    return torch.mean((F.avg_pool2d(img, 16) - target_E)**2)\n"
    "\n"
    "def tv_loss(x_r):\n"
    "    return (torch.mean(torch.abs(x_r[:,:,:,:-1] - x_r[:,:,:,1:])) +\n"
    "            torch.mean(torch.abs(x_r[:,:,:-1,:] - x_r[:,:,1:,:])))\n"
    "\n"
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "zero_dce = ZeroDCEpp(scale_factor=1).to(device)\n"
    "os.makedirs('weights', exist_ok=True)\n"
    "zdce_wt = 'weights/zero_dce_pp.pth'\n"
    "pretrained_loaded = False\n"
    "\n"
    "# -- Method 1: HuggingFace (primary, no gdown / Google Drive) -------------\n"
    "# Try several community-uploaded repos in order; first success wins.\n"
    "HF_SOURCES = [\n"
    "    ('Akbarnejad/Zero-DCE-Plus-Plus', 'Zero_DCE_plus_plus.pth'),\n"
    "    ('RitikSahoo/zero_dce_plus_plus',  'Epoch99.pth'),\n"
    "    ('SushantGautam/Zero-DCE-pp',      'Epoch99.pth'),\n"
    "]\n"
    "\n"
    "if not os.path.exists(zdce_wt) or os.path.getsize(zdce_wt) < 10_000:\n"
    "    from huggingface_hub import hf_hub_download\n"
    "    for repo_id, filename in HF_SOURCES:\n"
    "        try:\n"
    "            print(f'Trying HuggingFace: {repo_id}/{filename} ...')\n"
    "            p = hf_hub_download(repo_id=repo_id, filename=filename, local_dir='weights')\n"
    "            shutil.copy(p, zdce_wt)\n"
    "            if os.path.getsize(zdce_wt) > 10_000:\n"
    "                print(f'  Downloaded from {repo_id}')\n"
    "                break\n"
    "        except Exception as e:\n"
    "            print(f'  Failed ({repo_id}): {e}')\n"
    "\n"
    "# -- Load weights if any download succeeded --------------------------------\n"
    "if os.path.exists(zdce_wt) and os.path.getsize(zdce_wt) > 10_000:\n"
    "    try:\n"
    "        ckpt = torch.load(zdce_wt, map_location=device)\n"
    "        state = ckpt.get('model', ckpt.get('state_dict', ckpt))\n"
    "        zero_dce.load_state_dict(state, strict=False)\n"
    "        zero_dce.eval()\n"
    "        pretrained_loaded = True\n"
    "        print(f'Zero-DCE++ pretrained weights loaded! Params: {sum(p.numel() for p in zero_dce.parameters()):,}')\n"
    "    except Exception as e:\n"
    "        print(f'  Load failed ({e}), will train from scratch...')\n"
    "\n"
    "# -- Method 2 (fallback): train on RTTS (zero-reference) ------------------\n"
    "if not pretrained_loaded:\n"
    "    print('\\nFallback: training Zero-DCE++ on available images (zero-reference)...')\n"
    "\n"
    "    train_images = []\n"
    "    candidate_dirs = [\n"
    "        '/content/computer_vision_expirement/datasets/rtts',\n"
    "        '/content/computer_vision_expirement/datasets/rtts_processed/train',\n"
    "        f'{DATASETS_DIR}/rtts',\n"
    "        f'{DATASETS_DIR}/rtts_processed/train',\n"
    "        '/content/LOL/our485/low',\n"
    "        '/content/LOL',\n"
    "    ]\n"
    "    for d in candidate_dirs:\n"
    "        if not os.path.exists(d):\n"
    "            continue\n"
    "        found = (_glob.glob(f'{d}/**/*.png', recursive=True) +\n"
    "                 _glob.glob(f'{d}/**/*.jpg', recursive=True))\n"
    "        for p in found[:300]:\n"
    "            img = cv2.imread(p)\n"
    "            if img is not None:\n"
    "                train_images.append((p, img))\n"
    "        if train_images:\n"
    "            print(f'  Loaded {len(train_images)} training images from {d}')\n"
    "            break\n"
    "\n"
    "    if not train_images:\n"
    "        print('  [WARN] No training images found -- Zero-DCE++ will use random init.')\n"
    "        print('  Tip: ensure the setup cell (Cell 02) ran before this cell.')\n"
    "    else:\n"
    "        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)),\n"
    "                                  transforms.ToTensor()])\n"
    "        opt = torch.optim.Adam(zero_dce.parameters(), lr=1e-4, weight_decay=1e-4)\n"
    "        zero_dce.train()\n"
    "        print(f'Training Zero-DCE++ on {len(train_images)} images ({device})...')\n"
    "        for epoch in range(100):\n"
    "            total = 0.0\n"
    "            for _, img in train_images:\n"
    "                t = tf(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)\n"
    "                enh, curves = zero_dce(t)\n"
    "                loss = (1.0*spatial_consistency_loss(enh, t) +\n"
    "                        10.0*exposure_loss(enh) +\n"
    "                        5.0*color_constancy_loss(enh) +\n"
    "                        200.0*tv_loss(curves))\n"
    "                opt.zero_grad(); loss.backward(); opt.step()\n"
    "                total += loss.item()\n"
    "            if (epoch+1) % 20 == 0:\n"
    "                print(f'  Epoch {epoch+1}/100  Loss: {total/max(len(train_images),1):.4f}')\n"
    "        zero_dce.eval()\n"
    "        os.makedirs(CKPT_DIR, exist_ok=True)\n"
    "        torch.save(zero_dce.state_dict(), f'{CKPT_DIR}/zero_dce_pp_finetuned.pth')\n"
    "        print('Zero-DCE++ trained and checkpoint saved.')\n"
    "\n"
    "def enhance_zero_dce(img_bgr):\n"
    "    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n"
    "    t = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)\n"
    "    with torch.no_grad():\n"
    "        enhanced, _ = zero_dce(t)\n"
    "    out = enhanced.squeeze(0).cpu().clamp(0,1).permute(1,2,0).numpy()\n"
    "    return cv2.cvtColor((out*255).astype('uint8'), cv2.COLOR_RGB2BGR)\n"
)
cells[c8]["source"] = [CELL8]
print(f"Fixed Cell {c8}: removed gdown, HuggingFace as primary (3 repos)")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 — Cell 13: model_name variable + index-based filenames
# ═══════════════════════════════════════════════════════════════════════════
CELL13 = (
    "# Save enhanced outputs for each model to Drive (deliverable 2.2)\n"
    "import os\n"
    "\n"
    "ENHANCED_DIR = f'{RESULTS_DIR}/enhanced_outputs'\n"
    "os.makedirs(ENHANCED_DIR, exist_ok=True)\n"
    "\n"
    "n_save = min(20, len(all_test_images))\n"
    "saved = {name: 0 for name in models}\n"
    "\n"
    "for name, enhance_fn in models.items():\n"
    "    # Sanitise model name for use as directory name\n"
    "    model_name = name.replace('+', 'plus').replace(' ', '_')\n"
    "    model_dir  = f'{ENHANCED_DIR}/{model_name}'\n"
    "    os.makedirs(model_dir, exist_ok=True)\n"
    "\n"
    "    for idx, (path, img) in enumerate(all_test_images[:n_save]):\n"
    "        try:\n"
    "            enhanced = enhance_fn(img)\n"
    "            # Use index-based filename -- path may be a synthetic string like\n"
    "            # 'img.jpg_synth_dark' which has no valid image extension and\n"
    "            # causes cv2.imwrite to silently fail.\n"
    "            cv2.imwrite(f'{model_dir}/img_{idx:04d}.jpg', enhanced)\n"
    "            saved[name] += 1\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "print('Enhanced outputs saved to Drive:')\n"
    "for name, n in saved.items():\n"
    "    model_name = name.replace('+', 'plus').replace(' ', '_')\n"
    "    print(f'  {name}: {n} images -> {ENHANCED_DIR}/{model_name}')\n"
)
cells[c13]["source"] = [CELL13]
print(f"Fixed Cell {c13}: model_name variable + index-based filenames")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 3 — Cell 05: add thesis note for low-light limitation
# ═══════════════════════════════════════════════════════════════════════════
old_warn = (
    "elif len(all_test_images) < 10:\n"
    "    print(f\"\\n[WARNING] Only {len(all_test_images)} images "
    "— evaluation may not be representative.\")\n"
    "else:\n"
    "    print(\"\\n[OK] Dataset ready.\")"
)
new_warn = (
    "elif len(all_test_images) < 10:\n"
    "    print(f\"\\n[WARNING] Only {len(all_test_images)} images "
    "-- evaluation may not be representative.\")\n"
    "    print(\"  THESIS NOTE: Low-light image count is limited because ExDark requires\")\n"
    "    print(\"  manual download and Kaggle credentials are not set in this session.\")\n"
    "    print(\"  To fix: add KAGGLE_KEY to Colab Secrets, or upload ~/.kaggle/kaggle.json.\")\n"
    "    print(\"  Impact: NIQE for Zero-DCE++ may be less representative on low-light subset.\")\n"
    "    print(\"  PSNR/SSIM (synthetic pairs from RTTS) and hazy-image NIQE are unaffected.\")\n"
    "else:\n"
    "    print(\"\\n[OK] Dataset ready.\")"
)
src5 = "".join(cells[c5]["source"])
if old_warn in src5:
    cells[c5]["source"] = [src5.replace(old_warn, new_warn)]
    print(f"Fixed Cell {c5}: thesis note added for low-light limitation")
else:
    # Try without em-dash (the original may use a different dash)
    old_warn2 = old_warn.replace("\u2014", "-").replace("may not be representative.", "may not be representative.\")")
    print(f"WARNING: exact string not found in Cell {c5}; check dash character")
    # Append a note to end of cell as fallback
    src5 += (
        "\n# THESIS NOTE: low-light image count may be limited to <10 if ExDark download\n"
        "# fails (requires Kaggle auth). Fix: add KAGGLE_KEY to Colab Secrets.\n"
        "# Impact is limited to Zero-DCE++ NIQE score; PSNR/SSIM from synthetic RTTS pairs\n"
        "# and hazy-image NIQE are unaffected.\n"
    )
    cells[c5]["source"] = [src5]
    print(f"  Appended thesis note to Cell {c5} as fallback")

# ── Save ────────────────────────────────────────────────────────────────────
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nDone. Notebook saved.")
