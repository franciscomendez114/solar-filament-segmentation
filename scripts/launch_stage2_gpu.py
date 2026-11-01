# Starts a stage2_gpu_job.py run on a fresh Colab GPU: fuses the requested validation proposal sets locally from the
# cached detector boxes, writes configs.json, uploads code + weights + Kaggle token, and launches the job detached
# via scripts/colab_launch.py. Then run scripts/watch_colab_job.py.
# Edit PROPOSAL_SETS / CONFIGS for each batch; the batch name becomes the results folder.
# Usage (from the repo root): .venv/bin/python scripts/launch_stage2_gpu.py BATCH_NAME [GPU SESSION]
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.inference import fuse_boxes
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from colab_launch import start_job

SESSION, GPU = "s2", "A100"
CACHE_DIR = PROJECT_DIR / "data/processed/ensemble_proposals"
STAGE = PROJECT_DIR / "colab_transfer/stage2"
ALL7 = ["n", "s", "s_nomosaic", "s1280", "m", "l", "x"]
UNET_FILES = {"b5": "models/stage2/crop_unet_b5/best_crop_unet_b5.pt",
              "a050": "experiments/pipeline/20260920_110658_tversky_sweep/a050_b050/best.pt",
              "a030": "experiments/pipeline/20260920_110658_tversky_sweep/a030_b070/best.pt",
              "a070": "experiments/pipeline/20260920_110658_tversky_sweep/a070_b030/best.pt",
              "cropunet": "models/stage2/crop_unet/best_crop_unet.pt",
              "cropunet_ft": "models/stage2/crop_unet_proposal_finetuned/20260907_010647_940251/fit_20260907_012044_739450/best_proposal_unet.pt"}
UNET_FILES.update({name: f"experiments/pipeline/20261028_005250_unet_train/{name}.pt" for name in ["b5_sym_s1", "b5_sym_s2", "rx50_sym_s0"]})  # scripts/unet_train_job.py
# Round 7: the same recipes trained on train + teacher-drawn masks of the verified GONG pool (scripts/experiments/launch_unet_pseudo.py)
UNET_FILES.update({f"b5_sym_s{seed}_ps": f"experiments/pipeline/20261028_192123_unet_pseudo/u{seed}/b5_sym_s{seed}_ps.pt" for seed in [1, 2]})
# Full-data (train + val) retrains at each split run's chosen epoch; nothing can be validated with them
FULL_UNETS = ["full_b5", "full_a050", "full_b5_sym_s1", "full_b5_sym_s2"]
UNET_FILES.update({name: f"experiments/pipeline/20261028_015703_full_data_unets/{name}.pt" for name in FULL_UNETS})
ARCH = {"cropunet": "crop_unet", "cropunet_ft": "crop_unet"}
# label: (detector members, WBF cluster IoU, fused-score threshold)
# Batch 1 (20261028_000318_s2_gpu_sweep1) swept Stage 2 knobs on all7_t022: everything tied the control (0.4625); TTA +0.0028 best
# Batch 2: round-2 detector members (scripts/experiments/detector_ensemble.py) — flipped views lifted box F 0.673 -> 0.685
H, V = [f"{name}_hflip" for name in ALL7], [f"{name}_vflip" for name in ALL7]
ALL_MEMBERS = ALL7 + H + V + [f"{name}_up" for name in ALL7] + [f"{name}_last" for name in ALL7 if name != "x"]
# Batch 2 (20261028_001717_s2_gpu_detmembers): hv21 at 0.22 + TTA 0.4705, +0.0077 [-0.0001, +0.0151]; v-only and h-only less
# Batch 3: the round-3 detectors (yolo11m_flipud, yolo26s_flipud) inside the fusion, and yolo26s_flipud alone vs yolo26x alone
HV21 = ALL7 + H + V
NEW = ["y11m", "y11m_hflip", "y11m_vflip", "y26s_fu", "y26s_fu_hflip", "y26s_fu_vflip"]
PROPOSAL_SETS_BATCH3 = {"all7_t022": (ALL7, 0.40, 0.22), "hv21_t022": (HV21, 0.40, 0.22), "hv27_t022": (HV21 + NEW, 0.40, 0.22), "hv27_t024": (HV21 + NEW, 0.40, 0.24),
                 "hv24y11m_t022": (HV21 + NEW[:3], 0.40, 0.22), "all8_t022": (ALL7 + ["y26s_fu"], 0.40, 0.22),
                 "x_t020": (["x"], 0.40, 0.20), "y26sfu_t020": (["y26s_fu"], 0.40, 0.20), "y26sfu_t024": (["y26s_fu"], 0.40, 0.24)}
PROPOSAL_SETS_BATCH5 = {"all7_t022": (ALL7, 0.40, 0.22), "test_hv24y11m_t022": (HV21 + NEW[:3], 0.40, 0.22)}
# Batch 6: full-data detectors, 24 views at 0.24 — scripts/experiments/calibrate_full_threshold.py matches the split hv24 test-box count at 0.22
FULL24 = [f"full_{name}{suffix}" for name in ["yolo26n", "yolo26s", "yolo26s_nomosaic", "yolo26s_1280", "yolo26m", "yolo26l", "yolo26x", "yolo11m_flipud"] for suffix in ["", "_hflip", "_vflip"]]
PROPOSAL_SETS_BATCH6 = {"test_full24_t024": (FULL24, 0.40, 0.24)}
# Batch 7: mask TTA over crop views (settings["crop_views"] in predict_instances) on the public-0.39 inputs
PROPOSAL_SETS_BATCH7 = {"hv24y11m_t022": (HV21 + NEW[:3], 0.40, 0.22)}
# Batch 8: round-4 yolo26s trained at 1536 (mAP50-95 0.4048, best detector yet) inside the hv24 fusion, and alone vs yolo26x alone
HV24 = HV21 + NEW[:3]
S1536 = ["s1536", "s1536_hflip", "s1536_vflip"]
PROPOSAL_SETS_BATCH8 = {"hv24y11m_t022": (HV24, 0.40, 0.22), "hv27s1536_t022": (HV24 + S1536, 0.40, 0.22), "hv27s1536_t024": (HV24 + S1536, 0.40, 0.24),
                 "x_t020": (["x"], 0.40, 0.20), "s1536_t020": (["s1536"], 0.40, 0.20)}
# Batch 9: the public-0.39 recipe's fused threshold, which trades TP for FP inside the noise locally; plus a test set at 0.28
PROPOSAL_SETS_BATCH9 = {f"hv24y11m_t{t:03d}": (HV24, 0.40, t / 100) for t in [22, 25, 28, 31]}
PROPOSAL_SETS_BATCH9["test_hv24y11m_t028"] = (HV24, 0.40, 0.28)
# Batch 10: detectors trained on the union of annotators (detector_train_job.py "labels": "union"), alone vs their per-record twins
# (same recipe and seed) and inside the fusion: swapped in for y11m, or added
views = lambda name: [name, f"{name}_hflip", f"{name}_vflip"]
PROPOSAL_SETS_BATCH10 = {"hv24y11m_t022": (HV24, 0.40, 0.22), "hv24un_t022": (HV21 + views("y11m_un"), 0.40, 0.22),
                 "hv30un_t022": (HV24 + views("y11m_un") + views("s1536_un"), 0.40, 0.22), "hv30un_t025": (HV24 + views("y11m_un") + views("s1536_un"), 0.40, 0.25),
                 "s1536_t020": (["s1536"], 0.40, 0.20), "s1536un_t020": (["s1536_un"], 0.40, 0.20), "y11m_t020": (["y11m"], 0.40, 0.20), "y11mun_t020": (["y11m_un"], 0.40, 0.20)}
CHAMPION = ["b5", "a050"]


def config(label, proposals, unets, tta=False, mask_threshold=0.45, **pipeline):
    return {"label": label, "proposals": proposals, "unets": [[name, ARCH.get(name, "b5")] for name in unets], "tta": tta, "mask_threshold": mask_threshold, "pipeline": pipeline}


# Batch 3 (20261028_011511_s2_gpu_newdets): hv24y11m_t022 + TTA 0.4730, +0.0102 [+0.0023, +0.0181] WIN; yolo26s_flipud alone only +0.005 over yolo26x alone
# Batch 4 (20261028_012648_s2_gpu_unet3): third U-Net +0.0006 on top; hv24y11m_t022 + 3 U-Nets + TTA 0.4736, +0.0109 [+0.0024, +0.0191]
# Batch 5: the test submission for that config
CONFIGS_BATCH5 = [config("ctrl", "all7_t022", CHAMPION), {**config("sub_hv24y11m_3net_tta", "test_hv24y11m_t022", CHAMPION + ["b5_sym_s1"], tta=True), "split": "test"}]
# Batch 6: the same recipe retrained on train + val (public 0.39 config, plus b5_sym_s2 since a fourth U-Net tied a third)
CONFIGS_BATCH6 = [{**config("sub_full24_4net_tta", "test_full24_t024", FULL_UNETS, tta=True), "split": "test"}]  # public 0.39, same as split-trained
# Batch 7 (20261028_053745_s2_gpu_cropviews): control is the public-0.39 config (local 0.4736); variants average each box's mask over
# [crop expansion, prompt-box scale] views — all tied: ev3w +0.0005, mix5 +0.0003, ev3 +0.0002, ps3 -0.0010. Mask TTA is saturated.
NET3 = CHAMPION + ["b5_sym_s1"]
CONFIGS_BATCH7 = [config("ctrl", "hv24y11m_t022", NET3, tta=True),
           config("ev3", "hv24y11m_t022", NET3, tta=True, crop_views=[[1.3, 1.0], [1.5, 1.0], [1.7, 1.0]]),
           config("ev3w", "hv24y11m_t022", NET3, tta=True, crop_views=[[1.5, 1.0], [1.75, 1.0], [2.0, 1.0]]),
           config("ps3", "hv24y11m_t022", NET3, tta=True, crop_views=[[1.5, 0.9], [1.5, 1.0], [1.5, 1.1]]),
           config("mix5", "hv24y11m_t022", NET3, tta=True, crop_views=[[1.3, 1.0], [1.5, 0.9], [1.5, 1.0], [1.5, 1.1], [1.7, 1.0]])]
# Batch 8 (20261028_064736_s2_gpu_s1536): s1536 alone 0.4415 vs x alone 0.4351 (+0.0064 [-0.0099, +0.0239], tied); inside the fusion
# hv27s1536 -0.0035 / at 0.24 -0.0015 (tied). Detector fusion is saturated too.
CONFIGS_BATCH8 = [config("ctrl", "hv24y11m_t022", NET3, tta=True), config("hv27s1536", "hv27s1536_t022", NET3, tta=True), config("hv27s1536_t024", "hv27s1536_t024", NET3, tta=True),
           config("x_alone", "x_t020", CHAMPION), config("s1536_alone", "s1536_t020", CHAMPION)]
# Batch 9 (20261028_070411_s2_gpu_threshold): t025 -0.0010, t028 -0.0015, t031 -0.0078, all tied; t028 test submission public 0.39
CONFIGS_BATCH9 = [config("ctrl", "hv24y11m_t022", NET3, tta=True)] + [config(f"t{t:03d}", f"hv24y11m_t{t:03d}", NET3, tta=True) for t in [25, 28, 31]]
CONFIGS_BATCH9 += [{**config("sub_hv24y11m_t028_3net_tta", "test_hv24y11m_t028", NET3, tta=True), "split": "test"}]
# Batch 10 (20261028_090216_s2_gpu_union): union labels do not help — s1536_union alone -0.0013 vs its twin (tied), y11m_union alone
# -0.0183 [-0.0334, -0.0035] (loss: FP 660 vs 448); in the fusion hv24un -0.0018, hv30un -0.0004, hv30un_t025 +0.0019 [-0.0031, +0.0068] (all tied)
CONFIGS_BATCH10 = [config("ctrl", "hv24y11m_t022", NET3, tta=True)] + [config(label, f"{label}_t022" if label != "hv30un_t025" else label, NET3, tta=True) for label in ["hv24un", "hv30un", "hv30un_t025"]]
CONFIGS_BATCH10 += [config(f"{name}_alone", f"{name}_t020", CHAMPION) for name in ["s1536", "s1536un", "y11m", "y11mun"]]
# Batch 11: test submission for hv30un_t025 (best local 0.4756, tied) — competition data only, uses the day's last free submission
PROPOSAL_SETS_BATCH11 = {"test_hv30un_t025": (HV24 + views("y11m_un") + views("s1536_un"), 0.40, 0.25)}
CONFIGS_BATCH11 = [{**config("sub_hv30un_t025_3net_tta", "test_hv30un_t025", NET3, tta=True), "split": "test"}]  # public 0.39
# Batch 12 (20261028_102325_s2_gpu_pseudo): s1536_pl, the 1536 recipe (same seed) trained on train + 1000 pseudo-labeled GONG frames (mAP50-95 0.4173 vs 0.4048).
# Alone 0.4519 vs 0.4415, +0.0101 [-0.0033, +0.0233] — best single detector yet; one such member added to hv24 ties (-0.0007, at 0.25 -0.0026)
PROPOSAL_SETS_BATCH12 = {"hv24y11m_t022": (HV24, 0.40, 0.22), "hv27pl_t022": (HV24 + views("s1536_pl"), 0.40, 0.22), "hv27pl_t025": (HV24 + views("s1536_pl"), 0.40, 0.25),
                 "s1536_t020": (["s1536"], 0.40, 0.20), "s1536pl_t020": (["s1536_pl"], 0.40, 0.20)}
CONFIGS_BATCH12 = [config("ctrl", "hv24y11m_t022", NET3, tta=True), config("hv27pl", "hv27pl_t022", NET3, tta=True), config("hv27pl_t025", "hv27pl_t025", NET3, tta=True),
           config("s1536_alone", "s1536_t020", CHAMPION), config("s1536pl_alone", "s1536pl_t020", CHAMPION)]
# Batch 13 (20261028_113149_s2_gpu_students): y11m_pl (mAP50-95 0.4105 vs 0.3738) and the first student-only fusion (s1536_pl + y11m_pl, 6 views) vs hv24.
# y11m_pl alone +0.0072 [-0.0064, +0.0219] over y11m; pl6 0.4631 (loss vs hv24: two detectors are not eight); hv30pl -0.0010 (tied)
PL2 = views("s1536_pl") + views("y11m_pl")
PROPOSAL_SETS_BATCH13 = {"hv24y11m_t022": (HV24, 0.40, 0.22), "pl6_t022": (PL2, 0.40, 0.22), "pl6_t025": (PL2, 0.40, 0.25), "hv30pl_t022": (HV24 + PL2, 0.40, 0.22),
                 "y11m_t020": (["y11m"], 0.40, 0.20), "y11mpl_t020": (["y11m_pl"], 0.40, 0.20)}
CONFIGS_BATCH13 = [config("ctrl", "hv24y11m_t022", NET3, tta=True)] + [config(label, f"{label}_t022" if label != "pl6_t025" else label, NET3, tta=True) for label in ["pl6", "pl6_t025", "hv30pl"]]
CONFIGS_BATCH13 += [config("y11m_alone", "y11m_t020", CHAMPION), config("y11mpl_alone", "y11mpl_t020", CHAMPION)]
# Batch 14 (20261028_170606_s2_gpu_ps24): every fusion ties the 0.4736 control —
# hv45_t025 -0.0004, ps24_t028 -0.0009, fu24 -0.0014, hv45_t022 -0.0027, ps24_t025 -0.0041, ps24_t022 -0.0068. Alone, l_ps 0.4566 vs l
# 0.4349 (+0.022, the pseudo pool helps a single detector) while m_ps ties m (0.4433 vs 0.4437: +71 TP, +162 FP at 0.20).
# Batch 14: the real student fusion. ps24 = the five _ps students (trained on train + pseudo pool with "pseudo": true) + s1536_pl + y11m_pl
# + teacher x (no x student), each x3 views, against the hv24 teachers; hv45 = teachers + all student views; fu24 = hv24 with the
# flipud-only retrains (misnamed _pl, no pseudo data) swapped in for their originals. m_ps / l_ps alone vs their teacher twins.
STUDENTS = [view for name in ["n_ps", "s_ps", "s1280_ps", "m_ps", "l_ps", "s1536_pl", "y11m_pl"] for view in views(name)]
PS24 = STUDENTS + views("x")
FU24 = [view for name in ["n_pl", "s_pl", "s_nomosaic_pl", "s1280_pl", "m_pl", "l_pl", "x", "y11m"] for view in views(name)]
PROPOSAL_SETS_BATCH14 = {"hv24y11m_t022": (HV24, 0.40, 0.22), "ps24_t022": (PS24, 0.40, 0.22), "ps24_t025": (PS24, 0.40, 0.25), "ps24_t028": (PS24, 0.40, 0.28),
                 "hv45_t022": (HV24 + STUDENTS, 0.40, 0.22), "hv45_t025": (HV24 + STUDENTS, 0.40, 0.25), "fu24_t022": (FU24, 0.40, 0.22),
                 "m_t020": (["m"], 0.40, 0.20), "mps_t020": (["m_ps"], 0.40, 0.20), "l_t020": (["l"], 0.40, 0.20), "lps_t020": (["l_ps"], 0.40, 0.20)}
CONFIGS_BATCH14 = [config("ctrl", "hv24y11m_t022", NET3, tta=True)] + [config(label, label, NET3, tta=True) for label in ["ps24_t022", "ps24_t025", "ps24_t028", "hv45_t022", "hv45_t025", "fu24_t022"]]
CONFIGS_BATCH14 += [config(f"{name}_alone", f"{name}_t020", CHAMPION) for name in ["m", "mps", "l", "lps"]]
# Batch 15: the last untried combinations — every detector family together (hv24 + students + union, 51 views) and higher fused
# thresholds for the larger fusions — plus test CSVs for the two best not-yet-submitted variants (hv45_t025, ps24_t028)
UNION = views("y11m_un") + views("s1536_un")
PROPOSAL_SETS_BATCH15 = {"hv24y11m_t022": (HV24, 0.40, 0.22), "hv45_t028": (HV24 + STUDENTS, 0.40, 0.28), "hv51_t025": (HV24 + STUDENTS + UNION, 0.40, 0.25),
                 "hv51_t028": (HV24 + STUDENTS + UNION, 0.40, 0.28), "ps24_t031": (PS24, 0.40, 0.31),
                 "test_hv45_t025": (HV24 + STUDENTS, 0.40, 0.25), "test_ps24_t028": (PS24, 0.40, 0.28)}
CONFIGS_BATCH15 = [config("ctrl", "hv24y11m_t022", NET3, tta=True)] + [config(label, label, NET3, tta=True) for label in ["hv45_t028", "hv51_t025", "hv51_t028", "ps24_t031"]]
CONFIGS_BATCH15 += [{**config(f"sub_{label}_3net_tta", f"test_{label}", NET3, tta=True), "split": "test"} for label in ["hv45_t025", "ps24_t028"]]
# Batch 15 (20261028_173123_s2_gpu_combo): all tied — hv51_t025 +0.0008 (best), ps24_t031 -0.0018, hv45_t028 -0.0034, hv51_t028 -0.0041
# Batch 16: test CSV for hv51_t025
PROPOSAL_SETS_BATCH16 = {"test_hv51_t025": (HV24 + STUDENTS + UNION, 0.40, 0.25)}
CONFIGS_BATCH16 = [{**config("sub_hv51_t025_3net_tta", "test_hv51_t025", NET3, tta=True), "split": "test"}]  # public 0.40 (locally tied)
# Batch 17 (20261028_202507_s2_gpu_unet_ps): U-Net students (round 7) alone against their seed twins on the hv24 boxes, and inside the 3-net TTA average.
# Pseudo masks do not help Stage 2: alone s1_ps 0.4696 vs s1 0.4654 (+0.004) but s2_ps 0.4605 vs s2 0.4691 (-0.009); in the average
# add_both +0.0014, add_s2ps +0.0006, swap_s1ps -0.0003 (all tied). A student of the teacher's masks cannot draw better than the teacher.
PROPOSAL_SETS_BATCH17 = {"hv24y11m_t022": (HV24, 0.40, 0.22)}
CONFIGS_BATCH17 = [config("ctrl", "hv24y11m_t022", NET3, tta=True), config("swap_s1ps", "hv24y11m_t022", CHAMPION + ["b5_sym_s1_ps"], tta=True),
           config("add_s2ps", "hv24y11m_t022", NET3 + ["b5_sym_s2_ps"], tta=True), config("add_both", "hv24y11m_t022", NET3 + ["b5_sym_s1_ps", "b5_sym_s2_ps"], tta=True)]
CONFIGS_BATCH17 += [config(f"{name}_alone", "hv24y11m_t022", [name]) for name in ["b5_sym_s1", "b5_sym_s1_ps", "b5_sym_s2", "b5_sym_s2_ps"]]
# Batch 18 (20261028_204318_s2_gpu_lean): L1tta (1 detector pass, 8 U-Net passes per box) 0.4666, -0.0069 tied, at 0.53 s/image of
# Stage 2 vs the control's 1.27 (+24 detector passes); L3u2 -0.0058, S9u2 -0.0064, S3 -0.0070, L3 -0.0086 (tied); no TTA L1t25 -0.0104,
# L1 at 0.20 -0.0156 (losses). TTA matters more than extra detectors once the detector is a pseudo-label student.
# Batch 18: lean pipelines for feasibility — how much of the control survives with 1-9 detector passes and 1-16 U-Net passes per box.
# l_ps is the best single detector (alone 0.4566), b5_sym_s2 the best single competition-data U-Net, b5 + a050 the measured pair.
LEAN3 = ["l_ps", "s1536_pl", "m_ps"]
PROPOSAL_SETS_BATCH18 = {"hv24y11m_t022": (HV24, 0.40, 0.22), "lps_t020": (["l_ps"], 0.40, 0.20), "lps_t025": (["l_ps"], 0.40, 0.25), "lps3_t025": (views("l_ps"), 0.40, 0.25),
                 "lean3_t025": (LEAN3, 0.40, 0.25), "lean9_t025": ([view for name in LEAN3 for view in views(name)], 0.40, 0.25)}
CONFIGS_BATCH18 = [config("ctrl", "hv24y11m_t022", NET3, tta=True), config("L1", "lps_t020", ["b5_sym_s2"]), config("L1t25", "lps_t025", ["b5_sym_s2"]),
           config("L1tta", "lps_t025", ["b5_sym_s2"], tta=True), config("L3", "lps3_t025", ["b5_sym_s2"], tta=True), config("L3u2", "lps3_t025", CHAMPION, tta=True),
           config("S3", "lean3_t025", ["b5_sym_s2"], tta=True), config("S9u2", "lean9_t025", CHAMPION, tta=True)]
# Batch 19: test CSV for the lean pipeline L1tta (l_ps fused >= 0.25 + b5_sym_s2 with 8-way TTA) — its public score goes in the report
PROPOSAL_SETS_BATCH19 = {"test_lps_t025": (["l_ps"], 0.40, 0.25)}
CONFIGS_BATCH19 = [{**config("sub_lean_lps_b5s2_tta", "test_lps_t025", ["b5_sym_s2"], tta=True), "split": "test"}]  # public 0.39
# Batch 20: report numbers for the lean pipeline (public 0.39). Dumped validation masks of lean, the 24-view control and the C0 shape
# (x + b5) for scripts/official_self_evaluation.py; an ablation of each lean ingredient with the U-Net and TTA fixed — detector
# l (human labels, no flipud) / l_pl (flipud only) / l_ps (flipud + GONG pseudo-labels) at 0.20 and 0.25, the C0 U-Net b5 in place of
# b5_sym_s2, no TTA; and lean end to end from raw images (detector on the VM, timed), which must reproduce the cached-box score.
DETECTORS = {"l_ps.pt": "colab_transfer/detectors/l_ps.pt"}  # extracted from experiments/stage1/20261028_073435_pseudo_pl6/l_ps.tar.gz
PROPOSAL_SETS_BATCH20 = {"hv24y11m_t022": (HV24, 0.40, 0.22), "x_t020": (["x"], 0.40, 0.20)}
PROPOSAL_SETS_BATCH20.update({f"{name.replace('_', '')}_t0{t}": ([name], 0.40, t / 100) for name in ["l", "l_pl", "l_ps"] for t in [20, 25]})
LEAN = ["b5_sym_s2"]
CONFIGS_BATCH20 = [{**config("ctrl", "hv24y11m_t022", NET3, tta=True), "dump": True}, {**config("lean", "lps_t025", LEAN, tta=True), "dump": True},
           {**config("c0", "x_t020", ["b5"]), "dump": True}, config("lean_notta", "lps_t025", LEAN), config("lean_b5", "lps_t025", ["b5"], tta=True)]
CONFIGS_BATCH20 += [config(f"det_{name}_t0{t}", f"{name}_t0{t}", LEAN, tta=True) for name in ["l", "lpl", "lps"] for t in [20, 25] if (name, t) != ("lps", 25)]
CONFIGS_BATCH20 += [{**config("lean_e2e", "lps_t025", LEAN, tta=True), "detector": {"weights": "l_ps.pt", "imgsz": 1280, "iou": 0.40, "threshold": 0.25}}]
# Batch 20 (20261029_125607_s2_gpu_report, scripts/lean_report.py): lean 0.4666 vs c0 0.4272, +0.0392 [+0.0269, +0.0516] WIN; vs ctrl -0.0069 tied.
# Ablation vs lean: no TTA -0.0035 (loss), b5 for b5_sym_s2 -0.0102 (loss), l (no flipud, unpinned batch) -0.0304 (loss), l_pl (flipud, same
# recipe as l_ps minus the pseudo pool) -0.0006 (tied) — in the lean pipeline the pseudo-labels add nothing over flipud. lean_e2e reproduces 0.4666.
# Batch 21: lean end to end alone on a T4 (the free Colab / Kaggle GPU a reviewer would run the notebook on), for the feasibility timing
PROPOSAL_SETS_BATCH21 = {}
CONFIGS_BATCH21 = CONFIGS_BATCH20[-1:]  # 20261029_125747_s2_gpu_t4_e2e: PQ 0.4666 again; detector 0.08 + Stage 2 1.48 = 1.57 s/image on a T4 (A100 0.44)

# Batch 22: test CSV for lean with l_pl (the l_ps recipe on competition data only; tied with l_ps in batch 20), so the report has a
# public score for the pipeline without external data
PROPOSAL_SETS_BATCH22 = {"test_lpl_t025": (["l_pl"], 0.40, 0.25)}
CONFIGS_BATCH22 = [{**config("sub_lean_lpl_b5s2_tta", "test_lpl_t025", LEAN, tta=True), "split": "test"}]  # public 0.38
# Batch 23: the report's hyperparameter tables, re-measured on the final lean pipeline (every earlier sweep used the September
# Nano/Small detectors). One knob at a time against lean: detector confidence (cached boxes), box NMS IoU, NMS-free end2end
# and inference size (detector run on the VM; WBF IoU 0.99 so single-member fusion cannot merge what NMS kept), crop
# expansion and mask threshold.
PROPOSAL_SETS_BATCH23 = {f"lps_t{t:03d}": (["l_ps"], 0.40, t / 100) for t in [5, 10, 15, 20, 25, 30, 35, 40]}
detector = lambda **spec: {"weights": "l_ps.pt", "imgsz": 1280, "iou": 0.99, "threshold": 0.25, **spec}
CONFIGS_BATCH23 = [config("lean", "lps_t025", LEAN, tta=True)] + [config(f"conf_{t:03d}", f"lps_t{t:03d}", LEAN, tta=True) for t in [5, 10, 15, 20, 30, 35, 40]]
CONFIGS_BATCH23 += [{**config(f"nms_{int(iou * 100):03d}", "lps_t025", LEAN, tta=True), "detector": detector(nms_iou=iou)} for iou in [0.10, 0.20, 0.30, 0.50, 0.70]]
CONFIGS_BATCH23 += [{**config("nms_free", "lps_t025", LEAN, tta=True), "detector": detector(end2end=True)}]
CONFIGS_BATCH23 += [{**config(f"imgsz_{size}", "lps_t025", LEAN, tta=True), "detector": detector(imgsz=size, iou=0.40)} for size in [1024, 1536, 2048]]
CONFIGS_BATCH23 += [config(f"exp_{int(e * 100):03d}", "lps_t025", LEAN, tta=True, crop_expansion=e) for e in [1.25, 1.75, 2.00]]
CONFIGS_BATCH23 += [config(f"mask_{int(m * 100):03d}", "lps_t025", LEAN, tta=True, mask_threshold=m) for m in [0.35, 0.55]]
# Batch 24: the contrast-input detector l_cl (l_ps recipe on gray + CLAHE 1.5 + CLAHE 3.0, mAP50-95 0.4248 vs 0.4168) on the lean Stage 2,
# paired against lean on identical inputs. Thresholds fixed before scoring: the same three l_ps tied on.
PROPOSAL_SETS = {"lps_t025": (["l_ps"], 0.40, 0.25), **{f"lcl_t{t:03d}": (["l_cl"], 0.40, t / 100) for t in [20, 25, 30]}}
CONFIGS = [config("lean", "lps_t025", LEAN, tta=True)] + [config(f"lcl_t{t:03d}", f"lcl_t{t:03d}", LEAN, tta=True) for t in [20, 25, 30]]
# Batch 24 (20261031_235016_s2_gpu_contrast, L4): lean reproduces 0.4666; lcl_t025 0.4551, -0.0114 [-0.0233, -0.00004]; t030 -0.0144, t020 -0.0147.
# Lower at all three thresholds despite the best mAP so far, so l_cl is not adopted.


def main():
    batch = sys.argv[1]
    gpu, session = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else (GPU, SESSION)
    STAGE.mkdir(parents=True, exist_ok=True)
    for label, (members, iou_threshold, threshold) in PROPOSAL_SETS.items():
        split = "test" if label.startswith("test_") else "validation"  # test_* sets feed "split": "test" submission configs
        caches = [json.loads((CACHE_DIR / f"{split}_{name}.json").read_text()) for name in members]
        fused = {}
        for stem in caches[0]:
            boxes, scores = fuse_boxes([(np.asarray(c[stem]["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(c[stem]["scores"], dtype=np.float32)) for c in caches],
                                       [1.0] * len(members), iou_threshold, 0.01)
            fused[stem] = {"boxes": boxes[scores >= threshold].tolist(), "scores": scores[scores >= threshold].tolist()}
        (STAGE / f"{label}.json").write_text(json.dumps(fused))
    (STAGE / "configs.json").write_text(json.dumps(CONFIGS, indent=1))
    (PROJECT_DIR / f"experiments/pipeline/{batch}").mkdir(parents=True, exist_ok=False)
    (PROJECT_DIR / f"experiments/pipeline/{batch}/configs.json").write_text(json.dumps({"proposal_sets": PROPOSAL_SETS, "configs": CONFIGS}, indent=1))
    uploads = [(STAGE / "configs.json", "/content/configs.json")] + [(STAGE / f"{label}.json", f"/content/proposals/{label}.json") for label in PROPOSAL_SETS]
    uploads += [(PROJECT_DIR / UNET_FILES[name], f"/content/unets/{name}.pt") for name in sorted({name for c in CONFIGS for name, _ in c["unets"]})]
    uploads += [(PROJECT_DIR / DETECTORS[name], f"/content/detectors/{name}") for name in sorted({c["detector"]["weights"] for c in CONFIGS if "detector" in c})]
    start_job(session, gpu, PROJECT_DIR / "scripts/stage2_gpu_job.py", uploads)


if __name__ == "__main__":
    main()
