"""Настройки HTTP(S)-сервера из env."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


class ServerConfig:
    """Читает env при создании экземпляра (после load_dotenv / systemd)."""

    def __init__(self) -> None:
        self.env = os.environ.get("SPEECHLAB_ENV", "development").strip().lower()
        self.api_key = os.environ.get("SPEECHLAB_API_KEY", "").strip()
        self.host = os.environ.get("SPEECHLAB_SERVER_HOST", "0.0.0.0")
        self.port = _int("SPEECHLAB_SERVER_PORT", 8443)
        self.ssl_cert = os.environ.get("SPEECHLAB_SSL_CERT", "").strip()
        self.ssl_key = os.environ.get("SPEECHLAB_SSL_KEY", "").strip()
        self.upload_dir = Path(os.environ.get("SPEECHLAB_UPLOAD_DIR", ROOT / "server" / "uploads"))
        self.projects_root = Path(os.environ.get("SPEECHLAB_PROJECTS_ROOT", ROOT / "data" / "projects"))
        self.jobs_dir = Path(os.environ.get("SPEECHLAB_JOBS_DIR", ROOT / "server" / "data" / "jobs"))
        self.logs_dir = Path(os.environ.get("SPEECHLAB_LOGS_DIR", ROOT / "server" / "data" / "logs"))
        self.max_upload_mb = _int("SPEECHLAB_MAX_UPLOAD_MB", 500)
        self.require_api_key = _bool("SPEECHLAB_REQUIRE_API_KEY", default=False)

    @property
    def max_content_length(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def ssl_context(self):
        if self.ssl_cert and self.ssl_key:
            return (self.ssl_cert, self.ssl_key)
        return None

    @property
    def video_roots(self) -> list[Path]:
        return [self.upload_dir.resolve(), self.projects_root.resolve()]

    def ensure_dirs(self) -> None:
        for d in (self.upload_dir, self.projects_root, self.jobs_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Жёсткие проверки перед продакшеном."""
        if self.env == "production":
            if not self.api_key:
                print("FATAL: SPEECHLAB_API_KEY обязателен при SPEECHLAB_ENV=production", file=sys.stderr)
                sys.exit(1)
            if not self.require_api_key:
                self.require_api_key = True
