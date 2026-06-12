"""Локальный запуск: python -m server"""
from pathlib import Path

from dotenv import load_dotenv

from server.app import create_app
from server.config import ServerConfig

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)
load_dotenv(_root / "config" / ".env", override=True)


def main() -> None:
    cfg = ServerConfig()
    app = create_app(cfg)
    ssl = cfg.ssl_context
    scheme = "https" if ssl else "http"
    print(f"SpeechLab API [{cfg.env}] → {scheme}://{cfg.host}:{cfg.port}")
    if cfg.env != "production" and not ssl:
        print("  (prod: SPEECHLAB_ENV=production + nginx TLS)")
    app.run(host=cfg.host, port=cfg.port, ssl_context=ssl, threaded=True)


if __name__ == "__main__":
    main()
