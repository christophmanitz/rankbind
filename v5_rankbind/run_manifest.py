"""
v5_rankbind/run_manifest.py — Provenance tracking for every train/eval run.

Publishability invariant: every number quoted in the paper must trace to a
manifest.json written by this module. The manifest records:

    - run_id  (timestamp + source hash + config stem)
    - source hashes of every .py in v5_rankbind/   (repo is not a git repo)
    - resolved config (merged defaults + overrides)
    - python / torch / cuda / host / slurm env
    - sha256 of every input file that was read
    - split statistics (pairs/proteins per fold)
    - model parameter counts (trainable vs frozen)
    - per-epoch metrics (copied in at finish time)
    - output filenames + sha256 hashes

Two entry points:
    RunManifest.start(config_path, out_root)  -> manifest with run_id, dir
    manifest.finish(extra_metrics=...)        -> writes final manifest.json
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent


def sha256_of(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """Streaming sha256 for a file on disk. Returns hex digest or '' if absent."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def source_tree_hash(root: Path) -> dict:
    """Hash every .py under root (non-recursive into caches).

    Returns a mapping {relpath: sha256}. Also returns a combined short hash.
    """
    files: dict[str, str] = {}
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(root))
        files[rel] = sha256_of(p)
    combined = hashlib.sha256()
    for rel, digest in sorted(files.items()):
        combined.update(rel.encode())
        combined.update(digest.encode())
    files["_combined_short"] = combined.hexdigest()[:10]
    return files


def _get_env() -> dict:
    try:
        import torch
        torch_v = torch.__version__
        cuda_v = torch.version.cuda or ""
        cudnn_v = str(torch.backends.cudnn.version() or "")
        cuda_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda_available else ""
    except Exception:
        torch_v = cuda_v = cudnn_v = gpu = ""
        cuda_available = False

    try:
        import numpy as np
        np_v = np.__version__
    except Exception:
        np_v = ""

    try:
        import transformers
        hf_v = transformers.__version__
    except Exception:
        hf_v = ""

    return {
        "python": platform.python_version(),
        "torch": torch_v,
        "cuda_available": cuda_available,
        "cuda": cuda_v,
        "cudnn": cudnn_v,
        "gpu_name": gpu,
        "numpy": np_v,
        "transformers": hf_v,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "platform": platform.platform(),
    }


def _try_git_commit() -> dict:
    """If the project happens to be under git, return commit + dirty flag.

    Gracefully returns empty dict when the directory is not a repository
    (which is currently the case on the HPC box)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return {}
        commit = out.stdout.strip()
        dirty_out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return {
            "git_commit": commit,
            "git_dirty": bool(dirty_out.stdout.strip()),
        }
    except Exception:
        return {}


@dataclass
class RunManifest:
    run_id: str
    run_dir: Path
    config_path: str
    config: dict
    data: dict = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        config_path: str,
        config: dict,
        out_root: str | os.PathLike,
        tag: str = "",
    ) -> "RunManifest":
        now = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        src_hash = source_tree_hash(_HERE)
        stem = Path(config_path).stem if config_path else "adhoc"
        suffix = f"_{tag}" if tag else ""
        run_id = f"{now}_{src_hash['_combined_short']}_{stem}{suffix}"
        run_dir = Path(out_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": run_id,
            "tag": tag,
            "started_at": _dt.datetime.now().astimezone().isoformat(),
            "finished_at": "",
            "config_path": str(config_path),
            "config_resolved": config,
            "source_hashes": src_hash,
            **_try_git_commit(),
            "env": _get_env(),
            "inputs": {},
            "split": {},
            "model": {},
            "metrics": {},
            "outputs": {},
            "notes": [],
        }
        inst = cls(run_id=run_id, run_dir=run_dir, config_path=config_path,
                   config=config, data=data)
        inst.flush()
        return inst

    # ── recorders ──────────────────────────────────────────────────────────

    def record_inputs(self, paths: dict[str, str]) -> None:
        """paths: {label: absolute-or-project-relative path}."""
        entries = {}
        for label, p in paths.items():
            p_abs = p if os.path.isabs(p) else str(PROJECT_ROOT / p)
            entries[label] = {
                "path": p_abs,
                "exists": os.path.exists(p_abs),
                "sha256": sha256_of(p_abs),
                "size_bytes": os.path.getsize(p_abs) if os.path.exists(p_abs) else 0,
            }
        self.data["inputs"].update(entries)
        self.flush()

    def record_split(self, **stats: int) -> None:
        self.data["split"].update(stats)
        self.flush()

    def record_model(self, **stats: Any) -> None:
        self.data["model"].update(stats)
        self.flush()

    def record_metrics(self, **metrics: Any) -> None:
        self.data["metrics"].update(metrics)
        self.flush()

    def record_output(self, label: str, path: str) -> None:
        """Register a produced file. Its sha256 is computed on finish()."""
        self.data["outputs"][label] = {
            "path": str(path),
            "sha256": "",
            "size_bytes": 0,
        }
        self.flush()

    def note(self, msg: str) -> None:
        self.data["notes"].append({
            "at": _dt.datetime.now().astimezone().isoformat(),
            "msg": msg,
        })
        self.flush()

    # ── writers ────────────────────────────────────────────────────────────

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def open_jsonl(self, name: str):
        return (self.run_dir / name).open("a", buffering=1)

    def flush(self) -> None:
        (self.run_dir / "manifest.json").write_text(
            json.dumps(self.data, indent=2, default=str)
        )

    def finish(self, extra_metrics: dict | None = None) -> Path:
        if extra_metrics:
            self.data["metrics"].update(extra_metrics)
        # recompute output hashes
        for label, entry in self.data["outputs"].items():
            p = Path(entry["path"])
            if p.exists():
                entry["sha256"] = sha256_of(p)
                entry["size_bytes"] = p.stat().st_size
        self.data["finished_at"] = _dt.datetime.now().astimezone().isoformat()
        self.flush()
        return self.run_dir / "manifest.json"


# ──────────────────────────────────────────────────────────────────────────────
# Config loader (JSON only — no PyYAML dependency)
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str | os.PathLike, overrides: dict | None = None) -> dict:
    """Load a JSON config, merge with defaults if 'extends' key points at another config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = json.loads(path.read_text())

    # Resolve single-level inheritance
    parent_key = cfg.pop("extends", None)
    if parent_key:
        parent_path = (path.parent / parent_key).resolve()
        parent = load_config(parent_path)
        merged = _deep_merge(parent, cfg)
        cfg = merged

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    cfg["_config_path"] = str(path)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: v for k, v in base.items()}
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_deterministic_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


if __name__ == "__main__":
    # Smoke test
    m = RunManifest.start(
        config_path=str(_HERE / "configs" / "default.json"),
        config={"seed": 42, "_smoke_test": True},
        out_root=str(PROJECT_ROOT / "results" / "v5_rankbind"),
        tag="smoke",
    )
    m.record_inputs({"csv": "data/dataset_with_decoys.csv"})
    m.record_split(n_train_pairs=0, n_val_pairs=0, n_test_pairs=0)
    m.record_model(n_parameters_trainable=0, n_parameters_frozen=0)
    m.record_metrics(note="smoke")
    m.finish()
    print(f"Smoke manifest written to: {m.run_dir}")
