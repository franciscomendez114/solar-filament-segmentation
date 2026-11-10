# Compressing a Detector Ensemble into a Compact Two-Stage Filament Segmenter

Francisco Mendez, Rice University. Our solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

| | |
|---|---|
| **Technical report** | [`report/report.pdf`](report/report.pdf) |
| **Demo notebook** | [`notebooks/07_final_pipeline.ipynb`](notebooks/07_final_pipeline.ipynb), [open in Colab](https://colab.research.google.com/github/franciscomendez114/solar-filament-segmentation/blob/main/notebooks/07_final_pipeline.ipynb) |
| **Kaggle submission** | [`submissions/kaggle_submission.csv`](submissions/kaggle_submission.csv), the exact file submitted (public leaderboard 0.39) |
| **Trained models** | [`weights/`](weights/): `l_ps.pt` (detector) and `b5_sym_s2.pt` (U-Net) |

## In short

The task is to find every solar filament in a 2048×2048 GONG H-alpha image of the Sun and return one non-overlapping mask per filament, scored by panoptic quality (PQ) against each annotator's labels.

Our pipeline has two stages. A YOLO26l detector draws a box around each filament, and a U-Net segments the one filament inside a crop around each box. An ensemble of 17 detectors and 3 U-Nets reached 0.40 on the public leaderboard. The final pipeline, one detector and one U-Net, scores 0.39 and is statistically tied with that ensemble on our validation split, with 85% fewer parameters and one detector pass per image.

## Quick start

1. Open the notebook [in Colab](https://colab.research.google.com/github/franciscomendez114/solar-filament-segmentation/blob/main/notebooks/07_final_pipeline.ipynb) with a GPU runtime. A free T4 is enough.
2. Provide the competition data, which is not in this repo because it cannot be redistributed. Either put a Kaggle API token at `~/.kaggle/access_token` from an account that joined the competition, and the first cell downloads the data, or put the unzipped competition download at `/content/kaggle_data`.
3. Run all cells. In about 10 minutes on a T4, including installation and the data download, the notebook:
   - clones this repo and installs the pinned packages,
   - loads the two trained models from `weights/`,
   - shows the boxes and masks on one validation image,
   - scores our validation split with the official metric: expected **PQ 0.4666 (TP 828, FP 360, FN 430)**,
   - writes `submission.csv` for the 180 test images.

Nothing is retrained by default. A rerun matches our Kaggle file closely but not byte for byte, because GPU inference is not deterministic: in our last run from this public repo it wrote 1,239 masks against the submitted 1,237, and every submitted mask has a rerun mask with IoU above 0.9 (98.8% above 0.99). Set `RUN_VALIDATION` or `RUN_TEST` to `False` to skip those steps.

## How it works

```
2048² image → YOLO26l @1280 proposes one box per filament (confidence ≥ 0.25, NMS IoU 0.20)
→ each box becomes a square crop (1.5× its longer side), resized to 512²
→ U-Net (EfficientNet-B5 encoder, scSE attention) on 3 channels: grayscale, CLAHE, box prompt
→ mask averaged over 8 rotations and flips of the crop (test-time augmentation)
→ pasted back at full size, thresholded at 0.45, duplicates removed, contested pixels resolved
→ one non-overlapping mask per filament
```

- **Why two stages:** filaments are thin and small. Cropping around each box means 97% of validation crops are enlarged, not shrunk, when resized to the U-Net's 512² input, so thin filaments keep their width.
- **Why a box prompt:** the third crop channel marks which box to segment, so the U-Net returns one filament even when a crop contains several.
- **Detector training:** YOLO26l fine-tuned on the competition images plus 1,000 unlabeled GONG images pseudo-labeled by our ensemble (see [External data](#external-data)).
- **U-Net training:** competition data only, selected by running the whole pipeline on validation after each epoch and keeping the best official PQ.

The report gives every setting and the evidence for each choice.

## Results

All validation numbers use our held-out split (106 images, 170 annotator records, 1,258 filaments) and the official metric. The organizers' evaluation code gives the same PQ to six decimals. A *view* is one detector run on the original, horizontally flipped or vertically flipped image.

| Pipeline | Detector views | Parameters | Validation PQ | Public leaderboard |
|---|---|---|---|---|
| Baseline (YOLO26x + B5 U-Net) | 1 | 90M | 0.427 | 0.36 |
| Ensemble, 8 detectors + 3 U-Nets | 24 | 253M | 0.474 | 0.39 |
| Ensemble, 17 detectors + 3 U-Nets | 51 | 383M | 0.474 | 0.40 |
| **Final: 1 detector + 1 U-Net** | 1 | 57.5M | 0.467 | 0.39 |

- **Final vs. baseline:** +0.039 PQ (95% CI +0.027 to +0.052), finding 45 more filaments with 104 fewer false positives.
- **Final vs. ensembles:** tied (the paired 95% intervals include zero).
- **Mask quality (organizers' code, final pipeline):** mean IoU 0.60 and Dice 0.73 over all overlapping prediction–annotation pairs, 3.5% fragmentation and 2.0% over-merging.
- **Speed:** 1.57 s per image on a T4, about five minutes for the 180 test images.

## Training (optional)

The notebook's Training section can retrain either model from scratch. Set `TRAIN_DETECTOR` or `TRAIN_UNET` to `True`, and the rest of the notebook uses the new weights. It runs the same job scripts that trained the submitted models, with the exact settings in `configs/`.

| Model | Code | Settings | Hardware |
|---|---|---|---|
| Detector `l_ps` | `scripts/detector_train_job.py` | `configs/final_detector.json` | A100 (~24 GB at batch 8), about 2 hours |
| U-Net `b5_sym_s2` | `scripts/unet_train_job.py`, training loop in `src/models.py` | `configs/final_unet.json` | A100 recommended |

Inputs the training uses, all included in this repo:

- **The frozen split:** `data/processed/split_manifest.json` (seed 42, 15% of physical images).
- **GONG frames:** 1,000 unlabeled images in this repo's release [`gong-pool-v1`](https://github.com/franciscomendez114/solar-filament-segmentation/releases/tag/gong-pool-v1), downloaded by the notebook.
- **Pseudo-labels:** `data/processed/pseudo_boxes.json`, the 4,840 boxes our 24-view teacher ensemble drew on those frames.
- **Hard negatives and validation boxes for the U-Net:** `data/processed/hard_negatives/` and `data/processed/validation_proposals_yolo26x.json`.

The teacher detectors themselves are not in this repo, but their labels are, so the final detector can be retrained without them. The training cells were tested on Colab with shortened runs (1 detector epoch at batch 4 on an L4, 2 U-Net epochs): every cell ran and the rest of the notebook used the new weights. A full retrain was not repeated from this repo, and GPU training is not deterministic, so a retrain will be close to the saved weights but not identical.

## Repository

| Path | Contents |
|---|---|
| `notebooks/07_final_pipeline.ipynb` | The demo notebook: the whole pipeline, with optional training. |
| `notebooks/01`–`06` | Development notebooks. They expect our Google Drive project folder, so they document the work rather than run here. |
| `src/` | The pipeline code: data split, crops, models, training loop, inference, official scorer. |
| `weights/` | The two final models. The U-Net is stored as fp16 to fit GitHub's file limit. |
| `configs/` | Exact training settings of the final models, and the teacher list. |
| `scripts/` | Training jobs, GONG pool selection and checks, evaluation, and the report's figures. `official_eval.py` is the organizers' evaluation code, unmodified. |
| `scripts/experiments/` | One-off experiments and the ensemble pipeline, kept as a record. Most need outputs that are not in this repo. |
| `data/processed/` | Split, pseudo-labels, GONG pool list and test-overlap check, hard negatives, cached validation boxes. |
| `experiments/registry.csv` | Every experiment run with its score. |
| `submissions/` | `kaggle_submission.csv` (the submitted file) and `leaderboard_log.csv` (every Kaggle score). |
| `report/` | The report PDF, its LaTeX source and figures. |
| `requirements.txt` | Pinned packages. The notebook installs what it needs itself. |
| `build_notebooks.py` | Generates all notebooks; edit it rather than the notebooks. |

## External data

The organizers permit unlabeled GONG H-alpha images. Detector training used 1,000 such frames from the NSO archive, pseudo-labeled by our own ensemble. Each frame was checked against all 180 test images by timestamp, file hash and image correlation (`scripts/check_gong_pool.py`), and none was flagged. The public MAGFiLO archive, which contains test labels, was never used, and test images are used only for inference. The final detector scores the same on validation without this data (PQ 0.466 vs. 0.467).

GONG data courtesy of the NSO Integrated Synoptic Program (NISP), National Solar Observatory, operated by AURA, Inc. under a cooperative agreement with the National Science Foundation.

## License and acknowledgment

AGPL-3.0, matching Ultralytics. As the report states, this project used Claude Code to help write and run experiment code, create figures, and help revise the report.
