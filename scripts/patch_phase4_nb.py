"""Patch Phase4_Face_Recognition.ipynb — fix all issues vs experiment guide."""
import json

nb_path = 'notebooks/Phase4_Face_Recognition.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


cells = nb['cells']

# ── ADD Colab badge at index 0 ────────────────────────────────────────────────
badge = md_cell([
    '<a href="https://colab.research.google.com/github/Ib-Programmer/computer_vision_expirement/blob/main/'
    'notebooks/Phase4_Face_Recognition.ipynb" target="_parent">'
    '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
])

# Insert badge → all original indices shift by +1
cells.insert(0, badge)
# New indices:
# [0]  badge (new)
# [1]  title markdown
# [2]  setup/download (old 01)
# [3]  duplicate pip install (old 02) → DELETE
# [4]  ## 4.1 heading (old 03)
# [5]  InsightFace init (old 04)
# [6]  LFW detection (old 05)
# [7]  ## 4.2 heading (old 06)
# [8]  LFW embedding extraction (old 07) → FIX
# [9]  WiderFace one-cell (old 08) → CLEAN
# [10] save results (old 09) → CLEAN
# [11] ## 4.3 heading (old 10)
# [12] FAISS (old 11) → FIX
# [13] ## 4.4 heading (old 12)
# [14] FAR/FRR eval (old 13) → REWRITE
# [15] ## 4.5 heading (old 14)
# [16] robustness (old 15) → CLEAN
# [17] ## 4.6 heading (old 16) → RENAME
# [18] final report (old 17) → REWRITE
# [19] debug LFW cell (old 18) → DELETE

# ── [3] DELETE duplicate pip install ─────────────────────────────────────────
del cells[3]
# Indices now shift back by -1 for cells 3+:
# [3]  ## 4.1 heading
# [4]  InsightFace init
# [5]  LFW detection
# [6]  ## 4.2 heading
# [7]  LFW embedding extraction → FIX
# [8]  WiderFace one-cell → CLEAN
# [9]  save results → CLEAN
# [10] ## 4.3 heading
# [11] FAISS → FIX
# [12] ## 4.4 heading
# [13] FAR/FRR eval → REWRITE
# [14] ## 4.5 heading
# [15] robustness → CLEAN
# [16] ## 4.6 heading → RENAME
# [17] final report → REWRITE
# [18] debug LFW cell → DELETE

# ── [7] FIX LFW embedding extraction — imports + label extraction ─────────────
cells[7] = code_cell([
    "import os\n",
    "import numpy as np\n",
    "\n",
    "def get_lfw_identity(path):\n",
    "    \"\"\"Extract person identity from LFW filename.\n",
    "    LFW format: Aaron_Eckhart_0001.jpg -> Aaron_Eckhart\n",
    "    \"\"\"\n",
    "    fname = os.path.splitext(os.path.basename(path))[0]  # Aaron_Eckhart_0001\n",
    "    parts = fname.split('_')\n",
    "    return '_'.join(parts[:-1])  # remove trailing sequence number\n",
    "\n",
    "embeddings_db = []\n",
    "labels_db = []\n",
    "\n",
    "print('Extracting ArcFace embeddings from LFW test images...')\n",
    "for img_path in lfw_test[:200]:\n",
    "    img = cv2.imread(img_path)\n",
    "    if img is None:\n",
    "        continue\n",
    "    faces = app.get(img)\n",
    "    for face in faces:\n",
    "        embeddings_db.append(face.embedding)  # 512-d ArcFace embedding\n",
    "        labels_db.append(get_lfw_identity(img_path))\n",
    "\n",
    "embeddings_db = np.array(embeddings_db).astype('float32') if embeddings_db else np.zeros((0, 512), dtype='float32')\n",
    "n_identities = len(set(labels_db)) if labels_db else 0\n",
    "print(f'Extracted {len(embeddings_db)} embeddings from {n_identities} identities')\n",
    "print(f'Embedding shape: {embeddings_db.shape}  (512-d ArcFace)')\n",
])

# ── [8] CLEAN WiderFace detection cell — remove emoji and informal comments ───
cells[8] = code_cell([
    "import cv2, numpy as np, glob, time, os\n",
    "from insightface.app import FaceAnalysis\n",
    "import faiss\n",
    "from tqdm import tqdm\n",
    "\n",
    "# Initialize InsightFace (RetinaFace detection + ArcFace recognition)\n",
    "app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])\n",
    "app.prepare(ctx_id=0, det_size=(640, 640))\n",
    "print('InsightFace loaded: RetinaFace (detection) + ArcFace (recognition)')\n",
    "\n",
    "DATASETS_DIR = '/content/computer_vision_expirement/datasets'\n",
    "widerface_test = sorted(glob.glob(f'{DATASETS_DIR}/widerface_processed/test/*.jpg'))\n",
    "if not widerface_test:\n",
    "    widerface_test = sorted(glob.glob(f'{DATASETS_DIR}/widerface/WIDER_val/images/*/*.jpg'))[:500]\n",
    "print(f'WiderFace test images: {len(widerface_test)}')\n",
    "\n",
    "# ── Face Detection Evaluation ─────────────────────────────────────────────\n",
    "print('\\n' + '='*55)\n",
    "print('FACE DETECTION EVALUATION (RetinaFace on WiderFace)')\n",
    "print('='*55)\n",
    "\n",
    "detection_times = []\n",
    "faces_per_image = []\n",
    "\n",
    "for img_path in tqdm(widerface_test[:100], desc='Detecting faces', unit='img'):\n",
    "    img = cv2.imread(img_path)\n",
    "    if img is None:\n",
    "        continue\n",
    "    start = time.time()\n",
    "    faces = app.get(img)\n",
    "    elapsed = (time.time() - start) * 1000\n",
    "    detection_times.append(elapsed)\n",
    "    faces_per_image.append(len(faces))\n",
    "\n",
    "print(f'\\nDetection results ({len(detection_times)} images):')\n",
    "print(f'  Avg time      : {np.mean(detection_times):.1f} ms  (target < 100 ms on GPU)')\n",
    "print(f'  Total faces   : {sum(faces_per_image)}')\n",
    "print(f'  Avg faces/img : {np.mean(faces_per_image):.2f}')\n",
    "\n",
    "# ── Feature Extraction (ArcFace) ─────────────────────────────────────────\n",
    "print('\\n' + '='*55)\n",
    "print('FEATURE EXTRACTION (ArcFace on WiderFace)')\n",
    "print('='*55)\n",
    "\n",
    "embeddings = []\n",
    "image_paths = []\n",
    "\n",
    "for img_path in tqdm(widerface_test[:50], desc='Extracting features', unit='img'):\n",
    "    img = cv2.imread(img_path)\n",
    "    if img is None:\n",
    "        continue\n",
    "    faces = app.get(img)\n",
    "    for face in faces:\n",
    "        embeddings.append(face.embedding)\n",
    "        image_paths.append(img_path)\n",
    "        break  # one face per image\n",
    "\n",
    "embeddings_array = np.array(embeddings).astype('float32') if embeddings else np.zeros((0, 512), dtype='float32')\n",
    "print(f'Extracted {len(embeddings_array)} embeddings (512-d ArcFace)')\n",
    "\n",
    "if len(embeddings_array) > 0:\n",
    "    index_wf = faiss.IndexFlatL2(512)\n",
    "    index_wf.add(embeddings_array)\n",
    "    n_test = min(10, len(embeddings_array))\n",
    "    dists, idxs = index_wf.search(embeddings_array[:n_test], 2)\n",
    "    correct_matches = sum(1 for i in range(n_test) if idxs[i][0] == i)\n",
    "    print(f'Self-match accuracy (top-1): {correct_matches}/{n_test} = {100*correct_matches/n_test:.1f}%')\n",
])

# ── [9] CLEAN save results — remove emoji, use globals().get safely ───────────
cells[9] = code_cell([
    "import json, os\n",
    "import numpy as np\n",
    "\n",
    "PROJECT_DIR = '/content/drive/MyDrive/computer_vision'\n",
    "RESULTS_DIR = f'{PROJECT_DIR}/results/phase4'\n",
    "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
    "\n",
    "det_times = globals().get('detection_times', [])\n",
    "fp_img    = globals().get('faces_per_image', [])\n",
    "emb_arr   = globals().get('embeddings_array', np.zeros((0, 512)))\n",
    "c_matches = globals().get('correct_matches', 0)\n",
    "n_test    = globals().get('num_test', max(len(emb_arr), 1))\n",
    "\n",
    "results = {\n",
    "    'detection': {\n",
    "        'avg_time_ms': round(float(np.mean(det_times)), 2) if det_times else 0,\n",
    "        'total_faces': int(sum(fp_img)),\n",
    "        'avg_faces_per_image': round(float(np.mean(fp_img)), 2) if fp_img else 0,\n",
    "        'num_images': len(det_times),\n",
    "    },\n",
    "    'recognition': {\n",
    "        'num_embeddings': int(len(emb_arr)),\n",
    "        'self_match_accuracy': round(float(c_matches / max(n_test, 1)), 4),\n",
    "    }\n",
    "}\n",
    "\n",
    "with open(f'{RESULTS_DIR}/phase4_results.json', 'w') as f:\n",
    "    json.dump(results, f, indent=2)\n",
    "\n",
    "print(f'Results saved: {RESULTS_DIR}/phase4_results.json')\n",
    "print(f'  Detection : {results[\"detection\"][\"avg_time_ms\"]} ms/img | '\n",
    "      f'{results[\"detection\"][\"total_faces\"]} faces in {results[\"detection\"][\"num_images\"]} images')\n",
    "if results['recognition']['num_embeddings'] > 0:\n",
    "    print(f'  Recognition: {results[\"recognition\"][\"self_match_accuracy\"]*100:.1f}% self-match accuracy')\n",
])

# ── [11] FIX FAISS cell — add import time, use globals().get ─────────────────
cells[11] = code_cell([
    "import faiss\n",
    "import time\n",
    "\n",
    "emb_arr = globals().get('embeddings_array', None)\n",
    "if emb_arr is None or len(emb_arr) == 0:\n",
    "    print('[SKIP] No embeddings found. Run feature extraction cell (4.2) first.')\n",
    "else:\n",
    "    emb_norm = emb_arr.copy()\n",
    "    faiss.normalize_L2(emb_norm)\n",
    "\n",
    "    dimension = emb_norm.shape[1]  # 512\n",
    "    index = faiss.IndexFlatIP(dimension)  # inner product = cosine after normalisation\n",
    "    index.add(emb_norm)\n",
    "    print(f'FAISS index built: {index.ntotal} vectors, {dimension}-d')\n",
    "\n",
    "    k = 5\n",
    "    n_query = min(50, len(emb_norm))\n",
    "    t0 = time.time()\n",
    "    distances, indices = index.search(emb_norm[:n_query], k)\n",
    "    search_ms = (time.time() - t0) * 1000\n",
    "\n",
    "    print(f'Search time for {n_query} queries: {search_ms:.1f} ms')\n",
    "    print(f'Avg per query: {search_ms / n_query:.2f} ms  (target < 100 ms)')\n",
    "    print('\\nSample matches (query -> top-2 similarity):')\n",
    "    for i in range(min(5, len(distances))):\n",
    "        print(f'  Query {i}: top1 sim={distances[i][0]:.4f}, top2 sim={distances[i][1]:.4f}')\n",
])

# ── [13] REWRITE FAR/FRR evaluation — fix completely broken logic ─────────────
cells[13] = code_cell([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import faiss\n",
    "\n",
    "# Use LFW embeddings with identity labels (Cell 07) for proper FAR/FRR.\n",
    "# Falls back to WiderFace embeddings (Cell 08) if LFW embeddings not available.\n",
    "emb_lfw = globals().get('embeddings_db')\n",
    "lbl_lfw = globals().get('labels_db')\n",
    "\n",
    "if emb_lfw is not None and len(emb_lfw) > 0 and lbl_lfw:\n",
    "    eval_emb = emb_lfw\n",
    "    eval_lbl = lbl_lfw\n",
    "    print(f'Using LFW: {len(eval_emb)} embeddings, {len(set(eval_lbl))} identities')\n",
    "elif globals().get('embeddings_array') is not None and len(globals().get('embeddings_array')) > 0:\n",
    "    eval_emb = globals()['embeddings_array']\n",
    "    eval_lbl = [f'id_{i}' for i in range(len(eval_emb))]  # all different\n",
    "    print(f'Using WiderFace embeddings: {len(eval_emb)} vectors (all treated as unique identities)')\n",
    "    print('[NOTE] For standard FAR/FRR benchmarks, run Cell 07 (LFW) first.')\n",
    "else:\n",
    "    print('[SKIP] No embeddings found. Run feature extraction cells first.')\n",
    "    eval_emb = None\n",
    "\n",
    "if eval_emb is not None and len(eval_emb) >= 4:\n",
    "    emb_norm = eval_emb.copy().astype('float32')\n",
    "    faiss.normalize_L2(emb_norm)\n",
    "\n",
    "    # Build genuine pairs (same identity) and impostor pairs (different identity)\n",
    "    genuine_sims, impostor_sims = [], []\n",
    "    MAX_PAIRS = 2000\n",
    "    n = len(emb_norm)\n",
    "\n",
    "    for i in range(n):\n",
    "        for j in range(i + 1, n):\n",
    "            if len(genuine_sims) + len(impostor_sims) >= MAX_PAIRS:\n",
    "                break\n",
    "            sim = float(np.dot(emb_norm[i], emb_norm[j]))\n",
    "            if eval_lbl[i] == eval_lbl[j]:\n",
    "                genuine_sims.append(sim)\n",
    "            else:\n",
    "                impostor_sims.append(sim)\n",
    "        if len(genuine_sims) + len(impostor_sims) >= MAX_PAIRS:\n",
    "            break\n",
    "\n",
    "    print(f'Genuine pairs: {len(genuine_sims)} | Impostor pairs: {len(impostor_sims)}')\n",
    "    if not genuine_sims:\n",
    "        print('[NOTE] No genuine pairs found — each identity has only one image.')\n",
    "        print('       LFW with multiple images per person is needed for FAR/FRR.')\n",
    "\n",
    "    thresholds = np.arange(0.1, 1.0, 0.05)\n",
    "    rows = []\n",
    "    for t in thresholds:\n",
    "        tp = sum(1 for s in genuine_sims  if s >= t)\n",
    "        fn = sum(1 for s in genuine_sims  if s <  t)\n",
    "        fp = sum(1 for s in impostor_sims if s >= t)\n",
    "        tn = sum(1 for s in impostor_sims if s <  t)\n",
    "        far = fp / max(fp + tn, 1)\n",
    "        frr = fn / max(fn + tp, 1)\n",
    "        acc = (tp + tn) / max(tp + fp + fn + tn, 1)\n",
    "        rows.append({'Threshold': round(float(t), 2), 'Accuracy': round(acc, 4),\n",
    "                     'FAR': round(far, 4), 'FRR': round(frr, 4)})\n",
    "\n",
    "    eval_df = pd.DataFrame(rows)\n",
    "    best = eval_df.loc[eval_df['Accuracy'].idxmax()]\n",
    "\n",
    "    print('\\nRecognition Performance at Different Thresholds:')\n",
    "    print(eval_df[['Threshold', 'Accuracy', 'FAR', 'FRR']].to_string(index=False))\n",
    "    print(f'\\nBest threshold : {best[\"Threshold\"]}')\n",
    "    print(f'  Accuracy : {best[\"Accuracy\"]*100:.1f}%  (target > 95%)')\n",
    "    print(f'  FAR      : {best[\"FAR\"]*100:.2f}%  (target < 1%)')\n",
    "    print(f'  FRR      : {best[\"FRR\"]*100:.2f}%  (target < 5%)')\n",
    "\n",
    "    eval_df.to_csv(f'{RESULTS_DIR}/recognition_metrics.csv', index=False)\n",
    "    print(f'\\nSaved: {RESULTS_DIR}/recognition_metrics.csv')\n",
])

# ── [15] CLEAN robustness cell — remove emoji ────────────────────────────────
src15 = ''.join(cells[15]['source'])
src15 = src15.replace('✅ Saved to:', 'Saved:')
src15 = src15.replace('✅ Plot saved to:', 'Plot saved:')
cells[15]['source'] = [src15]

# ── [16] RENAME 4.6 heading: Literature → Deliverables ───────────────────────
cells[16] = md_cell([
    "## 4.6 Deliverables\n",
    "\n",
    "Published benchmarks for context, followed by deliverables checklist.",
])

# ── [17] REWRITE final report — remove hardcoded results, add deliverables ────
cells[17] = code_cell([
    "import pandas as pd, numpy as np, json, os\n",
    "from datetime import datetime\n",
    "\n",
    "PROJECT_DIR = '/content/drive/MyDrive/computer_vision'\n",
    "RESULTS_DIR = f'{PROJECT_DIR}/results/phase4'\n",
    "\n",
    "# ── Literature Benchmarks ─────────────────────────────────────────────────\n",
    "recog_lit = pd.DataFrame([\n",
    "    {'Method': 'FaceNet',    'Year': 2015, 'LFW Acc%': 99.63, 'Embedding': '128-d', 'Loss': 'Triplet',   'Source': 'CVPR 2015'},\n",
    "    {'Method': 'SphereFace', 'Year': 2017, 'LFW Acc%': 99.42, 'Embedding': '512-d', 'Loss': 'A-Softmax', 'Source': 'CVPR 2017'},\n",
    "    {'Method': 'ArcFace',    'Year': 2019, 'LFW Acc%': 99.83, 'Embedding': '512-d', 'Loss': 'ArcFace',   'Source': 'CVPR 2019'},\n",
    "    {'Method': 'AdaFace',    'Year': 2022, 'LFW Acc%': 99.82, 'Embedding': '512-d', 'Loss': 'AdaMargin', 'Source': 'CVPR 2022'},\n",
    "    {'Method': 'Ours (buffalo_l)', 'Year': 2024, 'LFW Acc%': 'see metrics', 'Embedding': '512-d', 'Loss': 'ArcFace', 'Source': 'This work'},\n",
    "])\n",
    "\n",
    "det_lit = pd.DataFrame([\n",
    "    {'Method': 'MTCNN',      'Year': 2016, 'Easy AP%': 84.8, 'Medium AP%': 82.5, 'Hard AP%': 60.7, 'FPS (GPU)': 16, 'Source': 'IEEE SPL 2016'},\n",
    "    {'Method': 'RetinaFace', 'Year': 2020, 'Easy AP%': 96.9, 'Medium AP%': 96.1, 'Hard AP%': 91.4, 'FPS (GPU)': 21, 'Source': 'CVPR 2020'},\n",
    "    {'Method': 'SCRFD',      'Year': 2022, 'Easy AP%': 96.1, 'Medium AP%': 94.9, 'Hard AP%': 88.5, 'FPS (GPU)': 42, 'Source': 'arXiv 2022'},\n",
    "])\n",
    "\n",
    "print('=' * 65)\n",
    "print('TABLE 4.1: Face Recognition Benchmarks (LFW dataset)')\n",
    "print('=' * 65)\n",
    "print(recog_lit.to_string(index=False))\n",
    "\n",
    "print('\\n' + '=' * 65)\n",
    "print('TABLE 4.2: Face Detection Benchmarks (WiderFace dataset)')\n",
    "print('=' * 65)\n",
    "print(det_lit.to_string(index=False))\n",
    "\n",
    "recog_lit.to_csv(f'{RESULTS_DIR}/literature_recognition.csv', index=False)\n",
    "det_lit.to_csv(f'{RESULTS_DIR}/literature_detection.csv', index=False)\n",
    "\n",
    "# ── Deliverables Checklist ─────────────────────────────────────────────────\n",
    "print('\\n' + '=' * 65)\n",
    "print('PHASE 4 DELIVERABLES CHECKLIST')\n",
    "print('=' * 65)\n",
    "\n",
    "det_times = globals().get('detection_times', [])\n",
    "det_ok = len(det_times) > 0\n",
    "print(f'\\n1. Face detection and recognition accuracy report')\n",
    "print(f'   {\"[OK]\" if det_ok else \"[MISSING - run detection cell (4.1) first]\"}')\n",
    "if det_ok:\n",
    "    import numpy as _np\n",
    "    print(f'   Avg detection latency : {_np.mean(det_times):.1f} ms  (target < 100 ms on GPU)')\n",
    "    print(f'   Total faces detected  : {sum(globals().get(\"faces_per_image\", []))}')\n",
    "\n",
    "csv_ok = os.path.exists(f'{RESULTS_DIR}/recognition_metrics.csv')\n",
    "print(f'\\n2. FAR/FRR curves at different thresholds')\n",
    "print(f'   {\"[OK]\" if csv_ok else \"[MISSING - run evaluation cell (4.4) first]\"}')\n",
    "\n",
    "rob_ok = os.path.exists(f'{RESULTS_DIR}/robustness_analysis.csv')\n",
    "print(f'\\n3. Per-condition robustness analysis (fog, rain, low-light)')\n",
    "print(f'   {\"[OK]\" if rob_ok else \"[MISSING - run robustness cell (4.5) first]\"}')\n",
    "\n",
    "all_ok = det_ok and csv_ok and rob_ok\n",
    "print(f'\\n{\"=\" * 65}')\n",
    "print(f'Phase 4 Status: {\"COMPLETE\" if all_ok else \"IN PROGRESS\"}')\n",
    "print('=' * 65)\n",
    "if all_ok:\n",
    "    print('Next: Open Phase5_Model_Optimization.ipynb')\n",
])

# ── [18] DELETE debug LFW download cell ─────────────────────────────────────
del cells[18]

# ── Write out ─────────────────────────────────────────────────────────────────
nb['cells'] = cells
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Done. Notebook now has {len(cells)} cells.')
for i, c in enumerate(cells):
    src = c['source']
    first = (src[0][:72] if src else '(empty)').encode('ascii', 'replace').decode().strip()
    print(f'  [{i:02d}] {c["cell_type"]:8s} | {first}')
