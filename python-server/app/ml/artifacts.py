from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def save_model(
    model: Any,
    model_name: str,
    model_dir: Path,
    metrics: dict[str, float] | None = None,
    extra_meta: dict[str, Any] | None = None,
    extra_artifacts: dict[str, Any] | None = None,
    feature_schema: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> tuple[Path, dict[str, Any], list[Path]]:
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    target_dir = model_dir
    if run_id:
        target_dir = model_dir / safe_model_name / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sha = git_sha() or "nogit"
    filename = f"{model_name}_{timestamp}_{sha}.pkl"
    model_path = target_dir / filename

    joblib.dump(model, model_path)

    try:
        relative_model_path = model_path.relative_to(model_dir).as_posix()
    except ValueError:
        relative_model_path = model_path.name

    meta: dict[str, Any] = {
        "model_name": model_name,
        "model_version": f"{timestamp}_{sha}",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics or {},
        "artifact": relative_model_path,
    }
    if run_id:
        meta["run_id"] = run_id
    if extra_meta:
        meta.update(extra_meta)
    meta_path = model_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    artifact_paths = [model_path, meta_path]
    if extra_artifacts:
        for name, obj in extra_artifacts.items():
            artifact_path = target_dir / f"{model_name}_{timestamp}_{sha}_{name}.joblib"
            joblib.dump(obj, artifact_path)
            artifact_paths.append(artifact_path)
    if feature_schema:
        schema_path = target_dir / f"{model_name}_{timestamp}_{sha}_schema.json"
        schema_path.write_text(
            json.dumps(feature_schema, indent=2), encoding="utf-8"
        )
        try:
            meta["feature_schema"] = schema_path.relative_to(model_dir).as_posix()
        except ValueError:
            meta["feature_schema"] = schema_path.name
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        artifact_paths.append(schema_path)
    return model_path, meta, artifact_paths


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
