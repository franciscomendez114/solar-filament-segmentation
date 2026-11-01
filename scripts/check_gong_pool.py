# Proves no GONG pool image (scripts/gong_pool.py) is a competition test image, before any of it is used for pseudo-labeling.
# Three independent checks, any one of which removes an image: (1) observation timestamp within ±1 hour of a test timestamp
# (the host's rule; the pool was drawn ±1 day away), (2) byte-identical file (MD5), (3) same frame re-encoded, which MD5 cannot
# catch: normalized correlation of fine on-disk structure (1024², minus a σ=2 blur, so brightness and contrast drop out) against
# every test image. Raw pixel difference does not work — the competition copy of a frame is brightness-adjusted. The threshold
# is the midpoint between a known same-frame pair (the archive copy of a test frame: 0.968) and the closest pair of different
# test frames (0.651); the same station one hour later scores 0.573. Writes data/processed/gong_pool_check.json and deletes
# flagged images from data/raw/gong_pool/.
# Usage (from the repo root): .venv/bin/python scripts/check_gong_pool.py SCRATCH_DIR
import hashlib
import json
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
POOL_DIR, TEST_DIR = PROJECT_DIR / "data/raw/gong_pool", PROJECT_DIR / "data/raw/test/test_images"
CALIBRATION = "20150620161314Mh"  # a test frame; its archive copy is the same observation, re-encoded
SIZE, SIGMA = 1024, 2
stamp = lambda name: datetime.strptime(name[:14], "%Y%m%d%H%M%S")


def structure(path):
    full = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(full, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    detail = image - cv2.GaussianBlur(image, (0, 0), SIGMA)
    disk = cv2.erode((cv2.resize(full, (SIZE, SIZE)) > 20).astype(np.uint8), np.ones((SIZE // 20, SIZE // 20), np.uint8)) > 0
    values = np.where(disk, detail - detail[disk].mean(), 0)
    return (values / np.linalg.norm(values)).ravel()


pool_paths, test_paths = sorted(POOL_DIR.glob("*.jpeg")), sorted(TEST_DIR.glob("*.jpeg"))
test_times = np.array([stamp(p.name).timestamp() for p in test_paths])
test_md5 = {hashlib.md5(p.read_bytes()).hexdigest() for p in test_paths}
test_features = np.stack([structure(p) for p in test_paths])

with tempfile.TemporaryDirectory(dir=sys.argv[1]) as folder:
    copy = Path(folder) / f"{CALIBRATION}.jpg"
    copy.write_bytes(urllib.request.urlopen(f"https://gong2.nso.edu/HA/hag/{CALIBRATION[:6]}/{CALIBRATION[:8]}/{CALIBRATION}.jpg", timeout=60).read())
    same_frame = float(test_features[[p.stem for p in test_paths].index(CALIBRATION)] @ structure(copy))
between_tests = test_features @ test_features.T
np.fill_diagonal(between_tests, -1)
different_frames = float(between_tests.max())
threshold = (same_frame + different_frames) / 2
print(f"same frame re-encoded {same_frame:.3f}; closest different test frames {different_frames:.3f}; threshold {threshold:.3f}", flush=True)
assert same_frame - different_frames > 0.15, "structure check cannot separate a re-encoded copy from a different frame"

flagged, closest = {}, []
for path in pool_paths:
    reasons = []
    if np.abs(test_times - stamp(path.name).timestamp()).min() <= 3600:
        reasons.append("within 1 hour of a test timestamp")
    if hashlib.md5(path.read_bytes()).hexdigest() in test_md5:
        reasons.append("byte-identical to a test file")
    similarity = test_features @ structure(path)
    closest.append(float(similarity.max()))
    if similarity.max() >= threshold:
        reasons.append(f"same frame as test {test_paths[int(similarity.argmax())].stem} (correlation {similarity.max():.3f})")
    if reasons:
        flagged[path.stem] = reasons
report = {"pool_images": len(pool_paths), "test_images": len(test_paths), "flagged": flagged,
          "min_gap_to_test_days": round(min(np.abs(test_times - stamp(p.name).timestamp()).min() for p in pool_paths) / 86400, 2),
          "structure_threshold": round(threshold, 3), "calibration_same_frame": round(same_frame, 3), "calibration_closest_different_test_frames": round(different_frames, 3),
          "max_pool_to_test_correlation": round(max(closest), 3)}
print(json.dumps(report, indent=1))
for stem in flagged:
    (POOL_DIR / f"{stem}.jpeg").unlink()
(PROJECT_DIR / "data/processed/gong_pool_check.json").write_text(json.dumps(report, indent=1) + "\n")
