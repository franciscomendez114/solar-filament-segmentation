# Builds a pool of unlabeled GONG H-alpha full-disk images for pseudo-labeling (the host permits unlabeled GONG H-alpha;
# the ban is on external ground truth). Same product as the competition JPEGs (2048², same file naming). Days within
# ±1 day of any train, validation or test image are excluded so no frame near-duplicates a competition scene; years follow
# the train distribution and stations are drawn uniformly among the six in the competition (B, C, L, M, T, U). Output: data/processed/gong_pool.json, a list of {name, url}.
# --download then fetches the pool here as grayscale JPEGs into data/raw/gong_pool/ (Colab VMs time out on the NSO archive).
# Usage (from the repo root): .venv/bin/python scripts/gong_pool.py [COUNT] | --download
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE = "https://gong2.nso.edu/HA/hag"
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
POOL_FILE, POOL_DIR = PROJECT_DIR / "data/processed/gong_pool.json", PROJECT_DIR / "data/raw/gong_pool"

if "--download" in sys.argv:
    import cv2
    import numpy as np
    POOL_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        target = POOL_DIR / item["name"].replace(".jpg", ".jpeg")
        if not target.exists():
            raw = urllib.request.urlopen(item["url"], timeout=60).read()
            cv2.imwrite(str(target), cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE), [cv2.IMWRITE_JPEG_QUALITY, 95])

    with ThreadPoolExecutor(8) as executor:
        list(executor.map(fetch, json.loads(POOL_FILE.read_text())))
    print(f"{len(list(POOL_DIR.glob('*.jpeg')))} pool images in {POOL_DIR}")
    sys.exit()

coco = json.loads((PROJECT_DIR / "data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
names = {image["file_name"] for image in coco["images"]} | {path.name for path in (PROJECT_DIR / "data/raw/test/test_images").glob("*.jpeg")}
day = lambda name: date(int(name[:4]), int(name[4:6]), int(name[6:8]))
blocked = {day(name) + timedelta(offset) for name in names for offset in [-1, 0, 1]}
year_weights = Counter(name[:4] for name in {image["file_name"] for image in coco["images"]})
random.seed(0)
candidates = [d for d in (date(2011, 1, 1) + timedelta(i) for i in range((date(2022, 12, 31) - date(2011, 1, 1)).days + 1)) if d not in blocked]
random.shuffle(candidates)
candidates.sort(key=lambda d: random.random() ** (1 / year_weights[str(d.year)]), reverse=True)  # weighted sampling without replacement
print(f"{len(candidates)} unblocked days", flush=True)


def pick(d):
    folder = f"{ARCHIVE}/{d:%Y%m}/{d:%Y%m%d}/"
    try:
        listing = urllib.request.urlopen(folder, timeout=30).read().decode()
    except Exception:
        return None
    files = sorted(set(re.findall(r'href="(\d{14}[BCLMTU]h\.jpg)"', listing)))  # the six competition stations only
    if not files:
        return None
    station = random.Random(d.toordinal()).choice(sorted({f[14] for f in files}))
    chosen = random.Random(d.toordinal() + 1).choice([f for f in files if f[14] == station])
    return {"name": chosen, "url": folder + chosen}


pool = []
with ThreadPoolExecutor(8) as executor:
    for start in range(0, len(candidates), 200):
        pool += [item for item in executor.map(pick, candidates[start:start + 200]) if item]
        print(f"{len(pool)} images", flush=True)
        if len(pool) >= COUNT:
            break
pool = sorted(pool[:COUNT], key=lambda item: item["name"])
assert not {item["name"][:8] for item in pool} & {f"{d:%Y%m%d}" for d in blocked}
print(f"years {sorted(Counter(i['name'][:4] for i in pool).items())}, stations {sorted(Counter(i['name'][14] for i in pool).items())}")
POOL_FILE.write_text(json.dumps(pool, indent=0) + "\n")
