# Watches a detached job on a Colab VM (stage2_gpu_job.py or anything writing the same PROGRESS.json / DONE.json
# flags): refreshes the one-hour proxy token every cycle, pulls each finished results/<label>.csv as soon as it
# appears, and stops the VM the moment the job is done, fails, or stalls, so it never bills idle.
# Usage (from the repo root): .venv/bin/python scripts/watch_colab_job.py SESSION ACCELERATOR LOCAL_DIR [NOTEBOOK_HASH]
# Without NOTEBOOK_HASH it attaches to the newest assignment in the CLI log, so a watcher started after a second VM was
# created latches onto that VM; pass the hash (the nbh= of this session's POST /tun/m/assign) when launching in parallel.
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from colab_cli.client import Accelerator, Assignment, Variant
from colab_cli.common import state as colab_state
from colab_cli.state import SessionState

PROJECT_DIR = Path(__file__).resolve().parents[1]
COLAB = str(PROJECT_DIR / ".venv/bin/colab")
COLAB_LOG = Path.home() / ".config/colab-cli/colab.log"
SESSION, ACCELERATOR, LOCAL_DIR = sys.argv[1], sys.argv[2], PROJECT_DIR / sys.argv[3]
POLL_SECONDS, STALL_MINUTES, MAX_HOURS = 60, 40, float(os.environ.get("COLAB_MAX_HOURS", 6))

PROBE = r'''
import json, pathlib
state = {}
for name in ["DONE", "PROGRESS"]:
    flag = pathlib.Path(f"/content/{name}.json")
    if flag.exists():
        state[name.lower()] = json.loads(flag.read_text())
log = pathlib.Path("/content/job.log")
if log.exists():
    lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    state["log_tail"] = lines[-1][-200:] if lines else ""
print("PROBE " + json.dumps(state))
'''


def refresh_session(notebook_hash):
    # _get_assignment is the read-only lookup; assign() would allocate a second VM
    assignment = colab_state.client._get_assignment(notebook_hash, variant=Variant.GPU, accelerator=Accelerator(ACCELERATOR))
    if not isinstance(assignment, Assignment):
        raise RuntimeError("the VM for this notebook hash is gone")
    current = colab_state.store.get(SESSION)
    proxy = assignment.runtime_proxy_info
    colab_state.store.add(SessionState(name=SESSION, token=proxy.token, url=proxy.url, endpoint=assignment.endpoint, variant="GPU",
                                       accelerator=ACCELERATOR, kernel_id=current.kernel_id if current else None))
    colab_state.client.keep_alive_assignment(assignment.endpoint)


def probe():
    result = subprocess.run([COLAB, "exec", "-s", SESSION, "--timeout", "120"], input=PROBE, capture_output=True, text=True, timeout=240)
    line = next((l for l in result.stdout.splitlines() if l.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError((result.stdout + result.stderr).strip()[-400:] or "no output from colab exec")
    return json.loads(line[len("PROBE "):])


def pull(remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run([COLAB, "download", remote, str(local), "-s", SESSION], capture_output=True, text=True).returncode == 0 and local.exists()


def stop_session():
    subprocess.run([COLAB, "stop", "-s", SESSION])
    print(subprocess.run([COLAB, "sessions"], capture_output=True, text=True).stdout.strip(), flush=True)


def main():
    matches = re.findall(r"nbh=([0-9a-f]{8}(?:_[0-9a-f]{4}){3}_[0-9a-f]{12})", COLAB_LOG.read_text(errors="replace"))
    notebook_hash = uuid.UUID(sys.argv[4] if len(sys.argv) > 4 else matches[-1].replace("_", "-"))
    print(f"watching {notebook_hash} as '{SESSION}' ({ACCELERATOR}) -> {LOCAL_DIR.relative_to(PROJECT_DIR)}", flush=True)
    started, failures, pulled = time.time(), 0, set()
    last_marker, last_change = None, time.time()
    while True:
        try:
            refresh_session(notebook_hash)
            state = probe()
            failures = 0
        except Exception as error:
            failures += 1
            print(f"[{datetime.now():%H:%M}] probe failed ({failures}/10): {error}", flush=True)
            if failures >= 10:
                print("Giving up; the VM may still be running — check `colab sessions`.", flush=True)
                return 1
            time.sleep(POLL_SECONDS)
            continue
        done = state.get("done")
        completed = (done or state.get("progress") or {}).get("completed", [])
        for label in completed:
            name = label if "." in label else f"{label}.csv"  # stage2 jobs list labels, other jobs list file names
            if label not in pulled and pull(f"/content/results/{name}", LOCAL_DIR / name):
                pulled.add(label)
                print(f"[{datetime.now():%H:%M}] pulled {label}", flush=True)
        if done:
            pull("/content/job.log", LOCAL_DIR / "job.log")
            print(f"[{datetime.now():%H:%M}] job finished: {done['status']}" + ("" if done["status"] == "ok" else "\n" + done.get("error", "")[-1500:]), flush=True)
            stop_session()
            return 0 if done["status"] == "ok" else 1
        marker = (len(completed), state.get("log_tail"))
        if marker != last_marker:
            last_marker, last_change = marker, time.time()
            print(f"[{datetime.now():%H:%M}] {len(completed)} done | {state.get('log_tail', '')[:150]}", flush=True)
        elif time.time() - last_change > STALL_MINUTES * 60:
            print(f"No progress in {STALL_MINUTES} min — stopping the VM.", flush=True)
            stop_session()
            return 1
        if time.time() - started > MAX_HOURS * 3600:
            print(f"Hit the {MAX_HOURS}h cap — stopping the VM.", flush=True)
            stop_session()
            return 1
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
