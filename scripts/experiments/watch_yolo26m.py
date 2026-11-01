# Watches the detached YOLO26m training on the Colab VM: one progress line a minute, and as soon as training
# finishes it downloads the run into models/stage1/ and stops the session so the VM stops billing.
# Usage (from the repo root): .venv/bin/python scripts/experiments/watch_yolo26m.py
import json
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from colab_cli.client import Accelerator, Assignment, Variant
from colab_cli.common import state as colab_state
from colab_cli.state import SessionState

PROJECT_DIR = Path(__file__).resolve().parents[2]
COLAB = str(PROJECT_DIR / ".venv/bin/colab")
SESSION = "yolo26m"
# The runtime proxy token Colab mints for a session expires after one hour; when it does the CLI gets a 401,
# prunes its local record and the session looks dead even though the VM and the detached training are fine.
# Re-fetching the assignment for this run's notebook hash mints a fresh token, so refresh every cycle.
# This UUID is the key to *this* run's VM — a new run gets a new one (see the launch note in the summary).
NOTEBOOK_HASH = uuid.UUID("e6bd540c-4521-49d2-ab18-3d3a08f4cf89")
RUN_NAME = "yolo26m_filament_detection"
LOCAL_RUN_DIR = PROJECT_DIR / "models/stage1" / RUN_NAME
TRANSFER_DIR = PROJECT_DIR / "colab_transfer"
POLL_SECONDS = 60
MAX_HOURS = 9  # hard stop: a wedged VM must not bill all day

PROBE = r'''
import json, pathlib, subprocess

state = {}
done = pathlib.Path("/content/DONE.json")
if done.exists():
    state["done"] = json.loads(done.read_text())
log = pathlib.Path("/content/train.log")
if log.exists():
    lines = [line for line in log.read_text(errors="replace").strip().split("\n") if line.strip()]
    state["log_tail"] = lines[-1][-160:] if lines else ""
results = pathlib.Path("/content/runs/yolo26m_filament_detection/results.csv")
if results.exists():
    rows = [row for row in results.read_text().strip().split("\n") if row]
    header = rows[0].split(",")
    values = [dict(zip(header, row.split(","))) for row in rows[1:]]
    if values:
        last = values[-1]
        best = max(values, key=lambda row: float(row["metrics/mAP50-95(B)"]))
        state.update(epoch=int(float(last["epoch"])), elapsed=float(last["time"]),
                     map50=float(last["metrics/mAP50(B)"]), map50_95=float(last["metrics/mAP50-95(B)"]),
                     best_map50_95=float(best["metrics/mAP50-95(B)"]), best_epoch=int(float(best["epoch"])))
gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True)
state["gpu"] = gpu.stdout.strip().replace("\n", " ")
print("PROBE " + json.dumps(state))
'''


def refresh_session():
    # _get_assignment is the read-only lookup; assign() would allocate a second VM if this one were gone.
    assignment = colab_state.client._get_assignment(NOTEBOOK_HASH, variant=Variant.GPU, accelerator=Accelerator.L4)
    if not isinstance(assignment, Assignment):
        raise RuntimeError("the VM for this notebook hash is gone — training is lost")
    current = colab_state.store.get(SESSION)
    proxy = assignment.runtime_proxy_info
    colab_state.store.add(SessionState(name=SESSION, token=proxy.token, url=proxy.url, endpoint=assignment.endpoint,
                                       variant="GPU", accelerator="L4",
                                       kernel_id=current.kernel_id if current else None))  # keep the kernel, don't leak one a minute
    colab_state.client.keep_alive_assignment(assignment.endpoint)  # the spawned daemon dies with the pruned record


def pull_checkpoint(epoch):
    # Cheap insurance: the VM can vanish, and last.pt is the only copy of the run.
    TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    target = TRANSFER_DIR / f"yolo26m_last_e{epoch}.pt"
    subprocess.run([COLAB, "download", "/content/runs/yolo26m_filament_detection/weights/last.pt",
                    str(target), "-s", SESSION], capture_output=True)
    # Sort by epoch number, not by name: lexicographically "e100" sorts before "e75" and would delete the newest.
    for old in sorted(TRANSFER_DIR.glob("yolo26m_last_e*.pt"), key=lambda path: int(path.stem.rsplit("_e", 1)[1]))[:-1]:
        old.unlink()
    return target


def probe():
    result = subprocess.run([COLAB, "exec", "-s", SESSION, "--timeout", "120"], input=PROBE,
                            capture_output=True, text=True, timeout=240)
    line = next((l for l in result.stdout.splitlines() if l.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError((result.stdout + result.stderr).strip()[-400:] or "no output from colab exec")
    return json.loads(line[len("PROBE "):])


def format_duration(seconds):
    hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def progress_line(state):
    stamp = datetime.now().strftime("%H:%M:%S")
    if "epoch" not in state:
        return f"[{stamp}] setting up | {state.get('log_tail', '')}"
    epoch, elapsed = state["epoch"], state["elapsed"]
    per_epoch = elapsed / max(epoch, 1)
    remaining = format_duration((300 - epoch) * per_epoch)
    return (f"[{stamp}] e{epoch}/300 mAP50 {state['map50']:.3f} mAP50-95 {state['map50_95']:.3f} | "
            f"best {state['best_map50_95']:.3f} @ e{state['best_epoch']} | {per_epoch:.0f}s/ep | "
            f"elapsed {format_duration(elapsed)} | eta<={remaining} | gpu {state['gpu']}")


def collect_run():
    TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TRANSFER_DIR / "yolo26m_artifacts.tar.gz"
    subprocess.run([COLAB, "download", "/content/yolo26m_artifacts.tar.gz", str(archive_path), "-s", SESSION], check=True)
    staging = TRANSFER_DIR / "yolo26m_extract"
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(staging)
    destination = LOCAL_RUN_DIR
    if destination.exists():  # never overwrite an existing run directory
        destination = destination.with_name(f"{RUN_NAME}_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.move(str(staging / RUN_NAME), str(destination))
    return destination


def stop_session():
    subprocess.run([COLAB, "stop", "-s", SESSION])
    print(subprocess.run([COLAB, "sessions"], capture_output=True, text=True).stdout.strip())


def main():
    deadline = time.time() + MAX_HOURS * 3600
    failures, last_pull = 0, 0
    while True:
        try:
            refresh_session()
            state = probe()
            failures = 0
            if state.get("epoch", 0) - last_pull >= 25:
                last_pull = state["epoch"]
                print(f"  pulled {pull_checkpoint(last_pull).name}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[{datetime.now():%H:%M:%S}] probe failed ({failures}/10): {error}", flush=True)
            if failures >= 10:
                print("Giving up on the probe. The VM may still be running — check: colab sessions", flush=True)
                return 1
            time.sleep(POLL_SECONDS)
            continue

        done = state.get("done")
        if done:
            print(f"[{datetime.now():%H:%M:%S}] training finished: {done.get('status')}", flush=True)
            if done["status"] == "ok":
                destination = collect_run()
                print(json.dumps({k: v for k, v in done.items() if k != "error"}, indent=2), flush=True)
                print(f"Run saved to {destination}", flush=True)
            else:
                print(done.get("error", "")[-2000:], flush=True)
            stop_session()
            return 0 if done["status"] == "ok" else 1

        print(progress_line(state), flush=True)
        if time.time() > deadline:
            print(f"Hit the {MAX_HOURS}h cap without a DONE flag — stopping the VM to cap the cost.", flush=True)
            stop_session()
            return 1
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
