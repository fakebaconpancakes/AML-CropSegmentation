"""Build and verify a complete source/evidence ZIP from a fixed Git revision.

Uses only the standard library. No model training or dependencies are installed.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


VERSION = "experiment-2-handover-v1"
REQUIRED = {
    "EXPERIMENT_2_HANDOVER.md", "README.md", "requirements.txt", "requirements-dev.txt",
    "configs/baseline.json", "notebooks/kaggle_baseline.ipynb",
    "src/dataset.py", "src/model.py", "src/train.py", "src/evaluate.py", "src/utils.py",
    "src/overfit_diagnostic.py", "tests/test_overfit_diagnostic.py",
    "experiments/baseline/test/metrics.json", "experiments/baseline/class_distribution.json",
    "experiments/overfit_diagnostic/summary.json", "experiments/overfit_diagnostic/summary.txt",
    "experiments/overfit_diagnostic/selected_samples.json",
    "experiments/overfit_diagnostic/selected_tensors.pt",
    "experiments/overfit_diagnostic/history.csv",
    "scripts/package_handover.py",
}
ALLOWED_TENSORS = "experiments/overfit_diagnostic/selected_tensors.pt"
EXCLUDED_PARTS = {".git", ".venv", "data", "old_project", "results", "outputs",
                  "__pycache__", ".ipynb_checkpoints"}


def build(ref, output, update_kaggle_alias=False):
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=root, text=True).strip()
    blob = subprocess.check_output(["git", "archive", "--format=zip", commit], cwd=root)
    with ZipFile(io.BytesIO(blob)) as archive:
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        missing = REQUIRED - names
        if missing:
            raise ValueError(f"Revision {ref} is not a complete handover: missing {sorted(missing)}")
        for name in names:
            path = PurePosixPath(name)
            if set(path.parts) & EXCLUDED_PARTS or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Raw/generated/private path must not be packaged: {name}")
            if path.suffix in {".pt", ".pth", ".pyc"} and name != ALLOWED_TENSORS:
                raise ValueError(f"Model weight/cache must not be packaged: {name}")
        hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}
    manifest = {"version": ref, "commit": commit, "file_count": len(names),
                "files_sha256": hashes,
                "scope": "Baseline code/evidence, completed diagnostic code/evidence, Experiment 2 guide; no Experiment 2 implementation or model checkpoints"}
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    with ZipFile(output, "a") as archive:
        # Fixed metadata keeps repeated builds of one revision byte-identical.
        info = ZipInfo("HANDOVER_VERSION.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Packaged file checksum mismatch: {name}")
        if json.loads(archive.read("HANDOVER_VERSION.json")) != manifest:
            raise ValueError("Packaged version manifest mismatch.")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8")
    if update_kaggle_alias:
        alias = output.parent / "kaggle_source.zip"
        if alias != output:
            shutil.copy2(output, alias)
        alias.with_suffix(alias.suffix + ".sha256").write_text(
            f"{digest}  {alias.name}\n", encoding="utf-8")
    print(f"Version: {ref}\nCommit: {commit}\nFiles: {len(names)} + version manifest")
    print(f"Archive: {output}\nSHA256: {digest}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=VERSION)
    parser.add_argument("--output", default="outputs/experiment_2_handover_v1.zip")
    parser.add_argument("--update-kaggle-alias", action="store_true")
    args = parser.parse_args()
    build(args.ref, args.output, args.update_kaggle_alias)


if __name__ == "__main__":
    main()
