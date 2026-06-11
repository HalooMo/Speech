"""Проверки путей и доступа для продакшена."""
from __future__ import annotations

from pathlib import Path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def allowed_video_path(path: Path, allowed_roots: list[Path]) -> bool:
    """Файл должен лежать внутри одного из разрешённых корней."""
    if not path.is_file():
        return False
    resolved = path.resolve()
    for root in allowed_roots:
        if root.is_dir() and _is_under(resolved, root):
            return True
    return False


def allowed_result_path(path: Path, projects_root: Path) -> bool:
    """Результат дубляжа — только внутри projects_root."""
    if not path.is_file():
        return False
    return _is_under(path, projects_root)
