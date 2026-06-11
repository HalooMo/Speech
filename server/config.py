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
    env: str = os.environ.get("SPEECHLAB_ENV", "development").strip().lower()
    api_key: str = os.environ.get("SPEECHLAB_API_KEY", "").strip()
    host: str = os.environ.get("SPEECHLAB_SERVER_HOST", "0.0.0.0")
    port: int = _int("SPEECHLAB_SERVER_PORT", 8443)
    ssl_cert: str = os.environ.get("SPEECHLAB_SSL_CERT", "").strip()
    ssl_key: str = os.environ.get("SPEECHLAB_SSL_KEY", "").strip()
    upload_dir: Path = Path(os.environ.get("SPEECHLAB_UPLOAD_DIR", ROOT / "server" / "uploads"))
    projects_root: Path = Path(os.environ.get("SPEECHLAB_PROJECTS_ROOT", ROOT / "data" / "projects"))
    jobs_dir: Path = Path(os.environ.get("SPEECHLAB_JOBS_DIR", ROOT / "server" / "data" / "jobs"))
    logs_dir: Path = Path(os.environ.get("SPEECHLAB_LOGS_DIR", ROOT / "server" / "data" / "logs"))
    max_upload_mb: int = _int("SPEECHLAB_MAX_UPLOAD_MB", 500)
    require_api_key: bool = _bool("SPEECHLAB_REQUIRE_API_KEY", default=False)

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
