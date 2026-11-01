# Watches the detached Tversky sweep on the Colab VM: one progress line a minute, each variant's artifacts pulled
# as soon as it finishes, and the session stopped the moment the third one is done so the VM stops billing.
# Usage (from the repo root): .venv/bin/python scripts/experiments/watch_tversky_sweep.py
import json
import re
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
SESSION = "tversky"
VARIANTS = ["a030_b070", "a050_b050", "a070_b030"]
LOCAL_DIR = PROJECT_DIR / "experiments/pipeline" / datetime.now().strftime("%Y%m%d_%H%M%S_tversky_sweep")
TRANSFER_DIR = PROJECT_DIR / "colab_transfer"
COLAB_LOG = Path.home() / ".config/colab-cli/colab.log"
POLL_SECONDS = 60
MAX_HOURS = 16  # hard stop for a wedged VM; three full 30-epoch runs at ~8 min/epoch fit inside it
STALL_MINUTES = 45  # the real cost guard: a live sweep advances an epoch every ~8 min, so silence means stuck

PROBE = r'''
import json, pathlib, subprocess

VARIANTS = ["a030_b070", "a050_b050", "a070_b030"]
state = {}
for name, path in [("done", "/content/DONE.json"), ("progress", "/content/PROGRESS.json")]:
    flag = pathlib.Path(path)
    if flag.exists():
        state[name] = json.loads(flag.read_text())
log = pathlib.Path("/content/sweep.log")
if log.exists():
    lines = [line for line in log.read_text(errors="replace").strip().split("\n") if line.strip()]
    state["log_tail"] = lines[-1][-160:] if lines else ""
for name in VARIANTS:
    history = pathlib.Path(f"/content/sweep/{name}/history.csv")
    if not history.exists():
        continue
    rows = [row for row in history.read_text().strip().split("\n") if row]
    header, values = rows[0].split(","), [dict(zip(rows[0].split(","), row.split(","))) for row in rows[1:]]
    if values:
        best = max(values, key=lambda row: float(row.get("official_pq") or 0))
        state["current"] = {"variant": name, "epoch": int(float(values[-1]["epoch"])),
                            "official_pq": float(values[-1].get("official_pq") or 0),
                            "best_pq": float(best.get("official_pq") or 0), "best_epoch": int(float(best["epoch"])),
                            "train_loss": float(values[-1]["train_loss"])}
gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True)
state["gpu"] = gpu.stdout.strip().replace("\n", " ")
print("PROBE " + json.dumps(state))
'''


def discover_notebook_hash():
    # The CLI logs "POST /tun/m/assign?nbh=..." when it allocates; the newest one is this session's VM. The nbh is
    # the notebook UUID with dashes swapped for underscores and dot-padded to 44 characters (uuid_to_web_safe_base64).
    matches = re.findall(r"nbh=([0-9a-f]{8}(?:_[0-9a-f]{4}){3}_[0-9a-f]{12})", COLAB_LOG.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"no notebook hash in {COLAB_LOG} — start the session first")
    return uuid.UUID(matches[-1].replace("_", "-"))


def refresh_session(notebook_hash):
    # _get_assignment is the read-only lookup; assign() would allocate a second VM if this one were gone.
    assignment = colab_state.client._get_assignment(notebook_hash, variant=Variant.GPU, accelerator=Accelerator.L4)
    if not isinstance(assignment, Assignment):
        raise RuntimeError("the VM for this notebook hash is gone — the sweep is lost")
    current = colab_state.store.get(SESSION)
    proxy = assignment.runtime_proxy_info
    colab_state.store.add(SessionState(name=SESSION, token=proxy.token, url=proxy.url, endpoint=assignment.endpoint,
                                       variant="GPU", accelerator="L4",
                                       kernel_id=current.kernel_id if current else None))  # keep the kernel, don't leak one a minute
    colab_state.client.keep_alive_assignment(assignment.endpoint)  # the spawned daemon dies with the pruned record


def collect_variant(name):
    # Each variant is tarred on the VM the moment it finishes, so a VM that dies later costs only the run in flight.
    TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TRANSFER_DIR / f"tversky_{name}.tar.gz"
    result = subprocess.run([COLAB, "download", f"/content/{name}.tar.gz", str(archive_path), "-s", SESSION], capture_output=True, text=True)
    if result.returncode != 0 or not archive_path.exists():
        return None
    with tarfile.open(archive_path) as archive:
        archive.extractall(LOCAL_DIR)
    return LOCAL_DIR / name


def probe():
    result = subprocess.run([COLAB, "exec", "-s", SESSION, "--timeout", "120"], input=PROBE,
                            capture_output=True, text=True, timeout=240)
    line = next((l for l in result.stdout.splitlines() if l.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError((result.stdout + result.stderr).strip()[-400:] or "no output from colab exec")
    return json.loads(line[len("PROBE "):])


def progress_line(state, started):
    stamp = datetime.now().strftime("%H:%M:%S")
    elapsed = int(time.time() - started)
    done = len(state.get("progress", {}).get("completed", []))
    current = state.get("current")
    if not current:
        return f"[{stamp}] setting up ({elapsed // 60}m) | {state.get('log_tail', '')}"
    return (f"[{stamp}] {current['variant']} ({done + 1}/3) e{current['epoch']}/30 "
            f"PQ {current['official_pq']:.4f} | best {current['best_pq']:.4f} @ e{current['best_epoch']} | "
            f"loss {current['train_loss']:.4f} | elapsed {elapsed // 3600}h{elapsed % 3600 // 60:02d}m | gpu {state['gpu']}")


def stop_session():
    subprocess.run([COLAB, "stop", "-s", SESSION])
    print(subprocess.run([COLAB, "sessions"], capture_output=True, text=True).stdout.strip())


def report(results):
    print("\nvariant      alpha/beta   best PQ  epoch   matched  middle  missed   middle area ratio", flush=True)
    for row in results:
        print(f"{row['variant']:12s} {row['tversky_alpha']:.2f}/{row['tversky_beta']:.2f}   {row['official_pq']:.4f}  "
              f"{row['best_epoch']:5d}   {row['matched_share']:6.1%}  {row['middle_share']:5.1%}  {row['missed_share']:5.1%}   "
              f"{row['middle_median_area_ratio']:.2f}x", flush=True)


def main():
    notebook_hash = discover_notebook_hash()
    print(f"watching notebook {notebook_hash} as session '{SESSION}'", flush=True)
    started, deadline = time.time(), time.time() + MAX_HOURS * 3600
    failures, collected = 0, set()
    last_marker, last_progress = None, time.time()
    while True:
        try:
            refresh_session(notebook_hash)
            state = probe()
            failures = 0
        except Exception as error:
            failures += 1
            print(f"[{datetime.now():%H:%M:%S}] probe failed ({failures}/10): {error}", flush=True)
            if failures >= 10:
                print("Giving up on the probe. The VM may still be running — check: colab sessions", flush=True)
                return 1
            time.sleep(POLL_SECONDS)
            continue

        for finished in state.get("progress", {}).get("completed", []):
            if finished["variant"] in collected:
                continue
            collected.add(finished["variant"])
            destination = collect_variant(finished["variant"])
            print(f"[{datetime.now():%H:%M:%S}] {finished['variant']} done: PQ {finished['official_pq']:.4f} "
                  f"@ e{finished['best_epoch']}, middle bucket {finished['middle_share']:.1%} at "
                  f"{finished['middle_median_area_ratio']:.2f}x area -> {destination}", flush=True)

        done = state.get("done")
        if done:
            print(f"[{datetime.now():%H:%M:%S}] sweep finished: {done.get('status')}", flush=True)
            if done.get("results"):
                report(done["results"])
                (LOCAL_DIR / "sweep_summary.json").write_text(json.dumps(done["results"], indent=2) + "\n")
            if done["status"] != "ok":
                print(done.get("error", "")[-2000:], flush=True)
            stop_session()
            return 0 if done["status"] == "ok" else 1

        print(progress_line(state, started), flush=True)
        marker = (state.get("current", {}).get("variant"), state.get("current", {}).get("epoch"), len(collected))
        if marker != last_marker:
            last_marker, last_progress = marker, time.time()
        elif time.time() - last_progress > STALL_MINUTES * 60:
            print(f"No epoch in {STALL_MINUTES} min — stopping the VM rather than paying for a stuck run.", flush=True)
            stop_session()
            return 1
        if time.time() > deadline:
            print(f"Hit the {MAX_HOURS}h cap without a DONE flag — stopping the VM to cap the cost.", flush=True)
            stop_session()
            return 1
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
