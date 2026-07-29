from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.db.models import ModelArtifact, ModelRun
from app.db.session import SessionLocal
from app.ml.artifacts import file_checksum


@dataclass
class ArtifactInfo:
    artifact_type: str
    path: str
    checksum: str
    size_bytes: int


def _to_serializable(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _to_serializable(value.item())
        except Exception:
            return value
    return value


def build_artifact_info(path: Path, artifact_type: str, base_dir: Path | None = None) -> ArtifactInfo:
    resolved = path.resolve()
    display_path = resolved
    if base_dir is not None:
        try:
            display_path = resolved.relative_to(base_dir.resolve())
        except ValueError:
            display_path = resolved
    checksum = file_checksum(resolved)
    size_bytes = resolved.stat().st_size
    return ArtifactInfo(
        artifact_type=artifact_type,
        path=str(display_path),
        checksum=checksum,
        size_bytes=size_bytes,
    )


def record_run(
    *,
    run_id: str,
    model_name: str,
    model_version: str,
    feature_version: str,
    commit_hash: str | None,
    league_code: str | None,
    season_label: str | None,
    metrics: dict | None,
    params: dict | None,
    data_summary: dict | None,
    artifacts: list[ArtifactInfo],
    status: str = "completed",
) -> None:
    run = ModelRun(
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        feature_version=feature_version,
        commit_hash=commit_hash,
        league_code=league_code,
        season_label=season_label,
        status=status,
        metrics=_to_serializable(metrics) if metrics else None,
        params=_to_serializable(params) if params else None,
        data_summary=_to_serializable(data_summary) if data_summary else None,
    )
    with SessionLocal() as session:
        session.add(run)
        session.flush()
        for artifact in artifacts:
            session.add(
                ModelArtifact(
                    model_run_id=run.id,
                    artifact_type=artifact.artifact_type,
                    path=artifact.path,
                    checksum=artifact.checksum,
                    size_bytes=artifact.size_bytes,
                )
            )
        session.commit()
