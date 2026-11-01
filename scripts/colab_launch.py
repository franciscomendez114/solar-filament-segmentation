# Rents a Colab GPU, uploads src/ + the split manifest + the Kaggle token + any extra files, and starts a job script
# detached as /content/<job>.py > /content/job.log (Popen with a new session survives the one-hour proxy-token expiry).
# Follow it with scripts/watch_colab_job.py, which also stops the VM.
# Usage (from the repo root): .venv/bin/python scripts/colab_launch.py SESSION GPU scripts/JOB.py
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
COLAB = str(PROJECT_DIR / ".venv/bin/colab")
STAGE = PROJECT_DIR / "colab_transfer/launch"
CHUNK = 15 * 1024 * 1024  # the upload endpoint returns 500 for large files


def run(*command):
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def execute(session, code):
    return subprocess.run([COLAB, "exec", "-s", session, "--timeout", "120"], input=code, text=True, capture_output=True, check=True).stdout


def upload(session, local, remote):
    data = Path(local).read_bytes()
    if len(data) <= CHUNK:
        run(COLAB, "upload", str(local), remote, "-s", session)
        return
    parts = []
    for index in range(0, len(data), CHUNK):
        part = STAGE / f"chunk_{index // CHUNK:03d}"
        part.write_bytes(data[index:index + CHUNK])
        parts.append(f"{remote}.part{index // CHUNK:03d}")
        run(COLAB, "upload", str(part), parts[-1], "-s", session)
    output = execute(session, f"import os\nparts = {parts!r}\nwith open({remote!r}, 'wb') as out:\n"
                              "    for part in parts:\n        out.write(open(part, 'rb').read()); os.remove(part)\n"
                              f"print({remote!r}, os.path.getsize({remote!r}))\n")
    assert str(len(data)) in output, f"rejoined size mismatch for {remote}: {output[-300:]}"
    for part in STAGE.glob("chunk_*"):
        part.unlink()


def start_job(session, gpu, job_script, extra_uploads=()):
    STAGE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(STAGE / "src.tar.gz", "w:gz") as archive:
        archive.add(PROJECT_DIR / "src", arcname="src", filter=lambda info: None if "__pycache__" in info.name else info)
    run(COLAB, "new", "-s", session, "--gpu", gpu)
    uploads = [(STAGE / "src.tar.gz", "/content/src.tar.gz"), (PROJECT_DIR / "data/processed/split_manifest.json", "/content/split_manifest.json"),
               (Path(job_script), f"/content/{Path(job_script).name}"), (Path.home() / ".kaggle/access_token", "/root/.kaggle/access_token"), *extra_uploads]
    folders = sorted({str(Path(remote).parent) for _, remote in uploads})
    execute(session, f"import os\nfor d in {folders!r}: os.makedirs(d, exist_ok=True)\n")
    for local, remote in uploads:
        upload(session, local, remote)
    print(execute(session, "import subprocess, tarfile, os\n"
                           "tarfile.open('/content/src.tar.gz').extractall('/content', filter='data')\n"
                           "os.chmod('/root/.kaggle/access_token', 0o600)\n"
                           f"subprocess.Popen('cd /content && python {Path(job_script).name} > /content/job.log 2>&1', shell=True, start_new_session=True)\n"
                           "print('launched')\n").strip())


if __name__ == "__main__":
    start_job(sys.argv[1], sys.argv[2], PROJECT_DIR / sys.argv[3])
