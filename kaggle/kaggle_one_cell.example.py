# =============================================================================
# SpeechLab — одна ячейка Kaggle (скопируйте файл целиком)
# =============================================================================
# GPU + dataset с кодом + видео + токены ниже (вариант B).
#
# Важно: pip меняет numpy в НОВОМ процессе; пайплайн тоже в subprocess,
# чтобы не ломать numpy в ядре Jupyter ("cannot load module more than once").

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ========================= НАСТРОЙКИ ==========================================
# Меняйте только DATASET_NAME — slug датасета на Kaggle (пусто = авто-поиск).
KAGGLE_USER = "salimkazhaerov"
DATASET_NAME = "speechlab-fourth"
VIDEO_FILE = "diolog.mp4"

PROJECT_NAME = "kaggle_dub"
SOURCE_LANG = "en"
TARGET_LANG = "ru"
USE_GIT = False
GIT_URL = ""
SKIP_DEMUCS = False
WHISPER_MODEL = "medium"
COMPUTE_TYPE = "float16"

# --- Секреты: скопируйте в kaggle_one_cell.py (он в .gitignore) ---
HF_TOKEN = "hf_your_token_here"
OPENAI_API_KEY = "sk_your_key_here"
OPENAI_BASE_URL = "https://api.agentplatform.ru/v1"
OPENAI_MODEL = "openai/gpt-5.5"
GROQ_API_KEY = "gsk_your_key_here"
# ==============================================================================

WORK = Path("/kaggle/working")
RUNTIME = WORK / "speechlab_runtime"
OUTPUT_ROOT = WORK / "speechlab_projects"
CHILD_SCRIPT = WORK / "_speechlab_child.py"

# numba (через librosa) в Kaggle образе требует numpy <= 2.0.x
_NUMPY_PIN = "numpy==2.0.2"
_SCIPY_PIN = "scipy==1.14.1"

_KAGGLE_INSTALL_ORDER = [
    ["python-dotenv>=1.0.0", "openai", "soundfile", "einops", "accelerate"],
    ["pandas", "huggingface_hub", "librosa"],
    ["transformers>=4.57.3,<5"],
    ["speechbrain"],
    ["pyannote.audio"],
    ["whisperx"],
    ["demucs", "qwen-tts>=0.1.0"],
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _pip_run(cmd: list[str], *, label: str = "") -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        _log(f"pip FAIL {label or cmd[-1]}:\n{r.stderr[-4000:]}")
        raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)


def _pin_numpy_scipy() -> None:
    py = sys.executable
    _pip_run([py, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", _NUMPY_PIN], label="pin-numpy")
    _pip_run([py, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", _SCIPY_PIN], label="pin-scipy")


def _pip_install(spec: str) -> None:
    py = sys.executable
    try:
        _pip_run(
            [py, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", spec],
            label=spec,
        )
    except subprocess.CalledProcessError:
        _log(f"  retry --no-deps: {spec}")
        _pip_run([py, "-m", "pip", "install", "-q", "--no-deps", spec], label=f"{spec} (no-deps)")


def _verify_numpy_subprocess() -> tuple[str, str]:
    """Проверка numpy/scipy в отдельном процессе — не импортируем numpy в ядре."""
    code = (
        "import numpy, scipy; "
        "v = tuple(int(x) for x in numpy.__version__.split('.')[:2]); "
        "assert v <= (2, 0), numpy.__version__; "
        "print(numpy.__version__, scipy.__version__)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        _log(r.stderr)
        raise RuntimeError(
            "numpy/scipy не работают в чистом процессе. "
            "Session → Restart session → запустите ячейку один раз."
        )
    parts = r.stdout.strip().split()
    return parts[0], parts[1]


def install_deps() -> None:
    """Только pip + проверка в subprocess (без import numpy в ядре)."""
    if "numpy" in sys.modules:
        _log(
            "⚠ numpy уже загружен в ядре — pip откатит файлы, но ядро может остаться сломанным. "
            "Лучше: Restart session → Run."
        )
    py = sys.executable
    _log("  pin numpy/scipy (start)")
    _pin_numpy_scipy()

    for i, batch in enumerate(_KAGGLE_INSTALL_ORDER):
        for spec in batch:
            _log(f"  install {spec}")
            _pip_install(spec)
        if batch == ["whisperx"]:
            _log("  pin numpy/scipy (after whisperx)")
            _pin_numpy_scipy()

    _log("  pin numpy/scipy (end)")
    _pin_numpy_scipy()
    nv, sv = _verify_numpy_subprocess()
    _log(f"OK (subprocess): numpy {nv}, scipy {sv}")


def _dataset_slug() -> str:
    return (DATASET_NAME or "").strip()


def _dataset_dir_candidates() -> list[Path]:
    """Пути к корню датасета по DATASET_NAME + KAGGLE_USER."""
    slug = _dataset_slug()
    if not slug:
        return []
    user = (KAGGLE_USER or "").strip()
    out: list[Path] = [Path(f"/kaggle/input/{slug}")]
    if user:
        out.insert(0, Path(f"/kaggle/input/datasets/{user}/{slug}"))
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for base in input_root.iterdir():
            if base.is_dir():
                out.append(base / slug)
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _expected_dataset_root() -> str:
    for p in _dataset_dir_candidates():
        if p.is_dir():
            return str(p)
    cands = _dataset_dir_candidates()
    return str(cands[0]) if cands else ""


def _expected_video_path() -> str:
    for root in _dataset_dir_candidates():
        vid = root / VIDEO_FILE
        if vid.is_file():
            return str(vid)
    slug = _dataset_slug()
    if not slug:
        return ""
    user = (KAGGLE_USER or "").strip()
    if user:
        return f"/kaggle/input/datasets/{user}/{slug}/{VIDEO_FILE}"
    return f"/kaggle/input/{slug}/{VIDEO_FILE}"


_CODE_MARKERS = ("main.py", "test.py", "llm.py", "dubbing.py", "env_config.py")


def _speechlab_score(path: Path) -> int:
    return sum(1 for name in _CODE_MARKERS if (path / name).is_file())


def _is_speechlab_dir(path: Path) -> bool:
    return (path / "main.py").is_file() and (path / "test.py").is_file()


def _list_input_tree(max_depth: int = 3) -> None:
    root = Path("/kaggle/input")
    if not root.is_dir():
        _log("  /kaggle/input отсутствует — датасет не подключён к ноутбуку?")
        return
    _log("  Содержимое /kaggle/input:")
    for base in sorted(root.iterdir()):
        if not base.is_dir():
            continue
        names = sorted(p.name for p in base.iterdir())[:20]
        _log(f"    {base}/ → {names}")
        if base.name == "datasets":
            for owner in sorted(base.iterdir()):
                if not owner.is_dir():
                    continue
                for slug in sorted(owner.iterdir()):
                    if not slug.is_dir():
                        continue
                    files = sorted(p.name for p in slug.iterdir())[:20]
                    _log(f"    {slug}/ → {files}")


def _find_code_source() -> Path:
    candidates: list[Path] = []
    expected = _expected_dataset_root()
    if expected:
        candidates.append(Path(expected))
    candidates.extend(_dataset_dir_candidates())

    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for main_py in sorted(input_root.rglob("main.py")):
            candidates.append(main_py.parent)

    seen: set[str] = set()
    best: tuple[int, Path] | None = None
    for p in candidates:
        key = str(p.resolve()) if p.is_dir() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.is_dir() or not _is_speechlab_dir(p):
            continue
        score = _speechlab_score(p)
        if best is None or score > best[0]:
            best = (score, p.resolve())

    if best:
        _log(f"  найден датасет (score {best[0]}): {best[1]}")
        return best[1]

    _list_input_tree()
    raise FileNotFoundError(
        "Не найден SpeechLab (main.py + test.py в одной папке).\n"
        "1) Add Data → подключите датасет к ноутбуку.\n"
        "2) Файлы должны лежать в корне датасета (без tools/).\n"
        f"3) Или задайте DATASET_NAME = \"<slug>\" (сейчас: {DATASET_NAME!r})."
    )


def _resolve_video_path(code_src: Path | None = None) -> Path:
    hint = _expected_video_path()
    if hint:
        p = Path(hint)
        if p.is_file():
            return p.resolve()

    search_roots: list[Path] = []
    if code_src:
        search_roots.append(code_src)
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        search_roots.append(input_root)

    media_names = (
        VIDEO_FILE, "diolog.mp4", "dialog.mp4", "dialog.wav", "diolog.wav",
        "video.mp4", "vocals.wav",
    )
    for root in search_roots:
        for name in media_names:
            cand = root / name
            if cand.is_file():
                _log(f"  видео/аудио: {cand}")
                return cand.resolve()

    exts = ("diolog.mp4", "dialog.mp4", "*.mp4", "dialog.wav", "*.wav")
    found: list[Path] = []
    for root in search_roots:
        for ext in exts:
            found.extend(root.glob(ext) if "*" in ext else [root / ext])
    found = [p for p in found if p.is_file()]
    if found:
        chosen = sorted(found, key=lambda x: x.as_posix())[0].resolve()
        _log(f"  видео/аудио (авто): {chosen}")
        return chosen

    raise FileNotFoundError(
        f"Видео не найдено (ожидалось: {hint or VIDEO_FILE}).\n"
        f"Положите {VIDEO_FILE} в датасет {DATASET_NAME!r} или смените VIDEO_FILE."
    )


def _sync_runtime(src: Path) -> Path:
    if RUNTIME.is_dir():
        shutil.rmtree(RUNTIME)
    shutil.copytree(
        src, RUNTIME,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "output_*"),
    )
    return RUNTIME


def patch_runtime(repo: Path, code_src: Path) -> None:
    """Синхронизация из плоского dataset (все файлы в корне) → runtime config/ + tools/."""
    required = (
        "main.py", "test.py", "prompt.py",
        "llm.py", "get_param.py", "dubbing.py", "fit_audio.py", "env_config.py",
    )
    missing = [n for n in required if not (code_src / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"В {code_src} не хватает файлов: {', '.join(missing)}. "
            f"Загрузите плоский датасет (без tools/ и kaggle/)."
        )

    src_ke = code_src / "kaggle_entry.py"
    dst_ke = repo / "kaggle_entry.py"
    if src_ke.is_file() and "def run_kaggle" in src_ke.read_text(encoding="utf-8"):
        shutil.copy2(src_ke, dst_ke)
        _log(f"  {_dataset_slug() or 'dataset'}/kaggle_entry.py → runtime")

    for name in ("main.py", "test.py", "prompt.py"):
        src = code_src / name
        if src.is_file():
            shutil.copy2(src, repo / name)
            _log(f"  {_dataset_slug() or 'dataset'}/{name} → runtime")

    (repo / "tools").mkdir(parents=True, exist_ok=True)
    (repo / "config").mkdir(parents=True, exist_ok=True)

    tool_modules = ("llm.py", "get_param.py", "dubbing.py", "fit_audio.py")
    for name in tool_modules:
        src = code_src / name
        if src.is_file():
            shutil.copy2(src, repo / "tools" / name)
            _log(f"  {_dataset_slug() or 'dataset'}/{name} → runtime/tools/")

    src_env = code_src / "env_config.py"
    if src_env.is_file():
        shutil.copy2(src_env, repo / "config" / "env_config.py")
        _log(f"  {_dataset_slug() or 'dataset'}/env_config.py → runtime/config/")

    if not (repo / "tools" / "__init__.py").is_file():
        (repo / "tools" / "__init__.py").write_text(
            '"""SpeechLab tools."""\n', encoding="utf-8"
        )
    if not (repo / "config" / "__init__.py").is_file():
        (repo / "config" / "__init__.py").write_text(
            '"""SpeechLab config."""\n', encoding="utf-8"
        )

    for legacy in ("llm.py", "get_param.py", "dubbing.py", "fit_audio.py", "env_config.py"):
        old = repo / legacy
        if old.is_file():
            old.unlink()
            _log(f"  удалён устаревший {legacy} в корне runtime")


def _set_secret(name: str, value: str) -> None:
    v = (value or "").strip()
    if not v or "..." in v or "your_token" in v.lower() or "your_key" in v.lower():
        return
    if v in ("hf_", "sk-", "gsk_"):
        return
    if not os.environ.get(name):
        os.environ[name] = v


def load_secrets(repo: Path | None = None, code_src: Path | None = None) -> None:
    """Секреты: переменные ячейки → .env датасета → Kaggle Secrets."""
    _set_secret("HF_TOKEN", HF_TOKEN)
    _set_secret("OPENAI_API_KEY", OPENAI_API_KEY)
    _set_secret("OPENAI_BASE_URL", OPENAI_BASE_URL)
    _set_secret("OPENAI_MODEL", OPENAI_MODEL)
    _set_secret("GROQ_API_KEY", GROQ_API_KEY)

    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ.setdefault("OPENAI_BASE_URL", "https://api.agentplatform.ru/v1")
    if not os.environ.get("OPENAI_MODEL"):
        os.environ.setdefault("OPENAI_MODEL", "openai/gpt-5.5")

    try:
        from dotenv import load_dotenv

        dataset_root = code_src or (Path(_expected_dataset_root()) if _expected_dataset_root() else None)
        for base in (dataset_root, repo, RUNTIME):
            if base is None:
                continue
            env_file = Path(base) / ".env"
            if env_file.is_file():
                load_dotenv(env_file, override=False)
                _log(f"  .env загружен: {env_file}")
    except ImportError:
        pass

    if os.environ.get("HF_TOKEN") and os.environ.get("OPENAI_API_KEY"):
        _log("  ключи из ячейки / .env")
        return

    aliases = {
        "HF_TOKEN": ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HF", "huggingface"],
        "OPENAI_API_KEY": ["OPENAI_API_KEY", "OPENAI_KEY", "openai"],
        "OPENAI_BASE_URL": ["OPENAI_BASE_URL"],
        "OPENAI_MODEL": ["OPENAI_MODEL"],
        "GROQ_API_KEY": ["GROQ_API_KEY", "GROQ", "groq"],
    }
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for env_key, names in aliases.items():
            if os.environ.get(env_key):
                continue
            for secret_name in names:
                try:
                    os.environ[env_key] = client.get_secret(secret_name)
                    _log(f"  secret OK: {secret_name} → {env_key}")
                    break
                except Exception:
                    continue
    except ImportError:
        pass

    missing = [k for k in ("HF_TOKEN", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        raise ValueError(
            "Не заданы: " + ", ".join(missing) + ".\n"
            "1) Kaggle: Add-ons → Secrets → HF_TOKEN и OPENAI_API_KEY.\n"
            "2) Вариант B — в начале ячейки:\n"
            "   HF_TOKEN = \"hf_...\"\n"
            "   OPENAI_API_KEY = \"sk_...\""
        )


def _write_child_script(repo: Path, use_kaggle_entry: bool) -> None:
    if use_kaggle_entry:
        body = '''
import os
import sys
from pathlib import Path

repo = Path(os.environ["SPEECHLAB_REPO"])
work = Path(os.environ["SPEECHLAB_WORK"])
out_root = Path(os.environ["SPEECHLAB_OUTPUT_ROOT"])
sys.path.insert(0, str(repo))
os.chdir(repo)

hf = work / "hf_cache"
hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(hf))
os.environ.setdefault("TORCH_HOME", str(work / "torch_cache"))
os.environ.setdefault("SPEECHLAB_WHISPER_MODEL", os.environ.get("SPEECHLAB_WHISPER_MODEL", "medium"))
os.environ.setdefault("SPEECHLAB_COMPUTE_TYPE", os.environ.get("SPEECHLAB_COMPUTE_TYPE", "float16"))

from kaggle_entry import configure_kaggle, run_kaggle

configure_kaggle(
    repo_root=repo,
    work_dir=work,
    output_root=out_root,
    whisper_model=os.environ["SPEECHLAB_WHISPER_MODEL"],
    compute_type=os.environ["SPEECHLAB_COMPUTE_TYPE"],
)
out = run_kaggle(
    os.environ["SPEECHLAB_PROJECT"],
    os.environ["SPEECHLAB_VIDEO"],
    os.environ["SPEECHLAB_SOURCE"],
    os.environ["SPEECHLAB_TARGET"],
    skip_demucs=os.environ.get("SPEECHLAB_SKIP_DEMUCS") == "1",
    whisper_model=os.environ["SPEECHLAB_WHISPER_MODEL"],
    compute_type=os.environ["SPEECHLAB_COMPUTE_TYPE"],
)
print("RESULT:", out)
'''
    else:
        body = '''
import os
import sys
from pathlib import Path

repo = Path(os.environ["SPEECHLAB_REPO"])
sys.path.insert(0, str(repo))
os.chdir(repo)
os.environ.setdefault("SPEECHLAB_WHISPER_MODEL", "medium")
os.environ.setdefault("SPEECHLAB_COMPUTE_TYPE", "float16")

import main

main.ROOT = Path(os.environ["SPEECHLAB_OUTPUT_ROOT"])
out = main.run(
    os.environ["SPEECHLAB_PROJECT"],
    os.environ["SPEECHLAB_VIDEO"],
    os.environ["SPEECHLAB_SOURCE"],
    os.environ["SPEECHLAB_TARGET"],
)
print("RESULT:", out)
'''
    CHILD_SCRIPT.write_text(body.strip() + "\n", encoding="utf-8")


def run_pipeline_subprocess(repo: Path, code_src: Path) -> Path:
    ke = repo / "kaggle_entry.py"
    use_ke = ke.is_file() and "def run_kaggle" in ke.read_text(encoding="utf-8")
    if not use_ke:
        _log("  (нет run_kaggle в kaggle_entry → main.run, медленнее по VRAM)")

    _write_child_script(repo, use_ke)
    env = os.environ.copy()
    video_path = _resolve_video_path(code_src)
    voice_cache = WORK / "speechlab_voice_bank"
    env.update({
        "SPEECHLAB_REPO": str(repo),
        "SPEECHLAB_WORK": str(WORK),
        "SPEECHLAB_OUTPUT_ROOT": str(OUTPUT_ROOT),
        "SPEECHLAB_PROJECT": PROJECT_NAME,
        "SPEECHLAB_VIDEO": str(video_path),
        "SPEECHLAB_SOURCE": SOURCE_LANG,
        "SPEECHLAB_TARGET": TARGET_LANG,
        "SPEECHLAB_SKIP_DEMUCS": "1" if SKIP_DEMUCS else "0",
        "SPEECHLAB_WHISPER_MODEL": WHISPER_MODEL,
        "SPEECHLAB_COMPUTE_TYPE": COMPUTE_TYPE,
        "SPEECHLAB_VOICE_CACHE": str(voice_cache),
        "SPEECHLAB_TRANSLATE_BATCH_SIZE": os.environ.get("SPEECHLAB_TRANSLATE_BATCH_SIZE", "12"),
    })
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (WORK / "hf_cache").mkdir(parents=True, exist_ok=True)

    _log("=== Пайплайн (новый процесс Python) ===")
    r = subprocess.run(
        [sys.executable, str(CHILD_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
    )
    if r.stdout:
        print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="")
    if r.returncode != 0:
        tail_out = (r.stdout or "")[-3000:]
        tail_err = (r.stderr or "")[-3000:]
        raise RuntimeError(
            "Пайплайн завершился с ошибкой.\n"
            f"--- stdout tail ---\n{tail_out}\n"
            f"--- stderr tail ---\n{tail_err}"
        )

    for line in (r.stdout or "").splitlines():
        if line.startswith("RESULT:"):
            return Path(line.split(":", 1)[1].strip())
    return OUTPUT_ROOT / PROJECT_NAME / f"{PROJECT_NAME}_dubbed.mp4"


# ============================= MAIN ===========================================
_log("=== SpeechLab one-cell ===")
if _dataset_slug():
    _log(f"Датасет: {DATASET_NAME} (user={KAGGLE_USER or 'авто'})")
else:
    _log("Датасет: авто-поиск (DATASET_NAME пуст)")

if USE_GIT:
    if not GIT_URL:
        raise ValueError("Укажите GIT_URL")
    src = WORK / "SpeechLab_git"
    if not src.is_dir():
        subprocess.check_call(["git", "clone", "--depth", "1", GIT_URL, str(src)])
    code_src = src
else:
    code_src = _find_code_source()
_log(f"Код: {code_src}")

_log("=== Установка зависимостей ===")
install_deps()

_log("=== Подготовка runtime ===")
repo = _sync_runtime(code_src)
patch_runtime(repo, code_src)
load_secrets(repo, code_src)

subprocess.run(["ffmpeg", "-version"], capture_output=True)

result = run_pipeline_subprocess(repo, code_src)
_log(f"=== Готово ===\nСкачайте: {result}")
