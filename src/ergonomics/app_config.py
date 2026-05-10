from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from .paths import PROJECT_ROOT, YOLO_POSE_WEIGHTS_PATH


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppConfig:
    database_path: Path
    secret_key: str
    require_auth: bool
    yolo_device: str
    yolo_pose_weights_path: Path
    allowed_origins: list[str]
    seed_default_users: bool
    max_upload_bytes: int


def _load_or_create_local_secret(database_path: Path) -> str:
    secret_path = Path(os.getenv("ERGONOMICS_SECRET_PATH", database_path.with_suffix(".key")))
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()
    secret = secrets.token_urlsafe(48)
    secret_path.write_text(secret, encoding="utf-8")
    try:
        secret_path.chmod(0o600)
    except OSError:
        pass
    return secret


def load_app_config() -> AppConfig:
    raw_origins = os.getenv("ERGONOMICS_ALLOWED_ORIGINS", "*")
    database_path = Path(os.getenv("ERGONOMICS_DB_PATH", PROJECT_ROOT / "data" / "app" / "postureos.sqlite3"))
    secret_key = os.getenv("ERGONOMICS_SECRET_KEY") or _load_or_create_local_secret(database_path)
    return AppConfig(
        database_path=database_path,
        secret_key=secret_key,
        require_auth=_bool_env("ERGONOMICS_REQUIRE_AUTH", default=True),
        yolo_device=os.getenv("YOLO_DEVICE", "auto"),
        yolo_pose_weights_path=Path(os.getenv("YOLO_POSE_WEIGHTS_PATH", YOLO_POSE_WEIGHTS_PATH)),
        allowed_origins=[origin.strip() for origin in raw_origins.split(",") if origin.strip()],
        seed_default_users=_bool_env("ERGONOMICS_SEED_DEFAULT_USERS", default=True),
        max_upload_bytes=_int_env("ERGONOMICS_MAX_UPLOAD_MB", 8) * 1024 * 1024,
    )
