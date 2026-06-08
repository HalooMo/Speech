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
PROJECT_NAME = "kaggle_dub"
VIDEO_PATH = "/kaggle/input/datasets/salimkazhaerov/speechlab-coded/vocals.wav"
SOURCE_LANG = "en"
TARGET_LANG = "ru"
CODE_DATASET_PATH = "/kaggle/input/speechlab-codedd"
USE_GIT = False
GIT_URL = ""
SKIP_DEMUCS = False
WHISPER_MODEL = "medium"
COMPUTE_TYPE = "float16"

# --- Секреты (вариант B): вставьте свои ключи между кавычками ---
HF_TOKEN = "hf_..."  # или Kaggle Secrets / .env
OPENAI_API_KEY = "sk_..."  # или Kaggle Secrets / config/.env
OPENAI_BASE_URL = "https://api.agentplatform.ru/v1"
OPENAI_MODEL = "openai/gpt-5.5"
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


def _find_code_source() -> Path:
    candidates: list[Path] = []
    if CODE_DATASET_PATH:
        candidates.append(Path(CODE_DATASET_PATH))
    candidates.extend([
        Path("/kaggle/input/speechlab-coded"),
        Path("/kaggle/input/datasets/salimkazhaerov/speechlab-coded"),
    ])
    for base in Path("/kaggle/input").glob("*"):
        candidates.append(base)
        candidates.append(base / "speechlab-coded")

    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if (p / "main.py").is_file() and (p / "test.py").is_file():
            return p.resolve()
    raise FileNotFoundError(
        "Не найден SpeechLab (main.py + test.py). Укажите CODE_DATASET_PATH."
    )


def _resolve_video_path() -> Path:
    p = Path(VIDEO_PATH)
    if p.is_file():
        return p.resolve()

    # Если остался шаблонный путь, попробуем подобрать первый медиафайл из /kaggle/input
    placeholder = "your-video-dataset" in str(p) or str(p).endswith("/video.mp4")
    if placeholder:
        exts = ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.wav", "*.mp3", "*.m4a")
        candidates: list[Path] = []
        root = Path("/kaggle/input")
        for ext in exts:
            candidates.extend(root.glob(f"**/{ext}"))
        if candidates:
            chosen = sorted(candidates, key=lambda x: x.as_posix())[0].resolve()
            _log(f"  VIDEO_PATH не задан, выбран файл: {chosen}")
            return chosen

    raise FileNotFoundError(
        f"VIDEO_PATH не найден: {VIDEO_PATH}\n"
        "Укажите корректный путь в начале ячейки, например:\n"
        "VIDEO_PATH = \"/kaggle/input/<dataset>/<file>.mp4\""
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
        _log(f"  {code_src.name}/kaggle_entry.py → runtime")

    for name in ("main.py", "test.py", "prompt.py"):
        src = code_src / name
        if src.is_file():
            shutil.copy2(src, repo / name)
            _log(f"  {code_src.name}/{name} → runtime")

    (repo / "tools").mkdir(parents=True, exist_ok=True)
    (repo / "config").mkdir(parents=True, exist_ok=True)

    tool_modules = ("llm.py", "get_param.py", "dubbing.py", "fit_audio.py")
    for name in tool_modules:
        src = code_src / name
        if src.is_file():
            shutil.copy2(src, repo / "tools" / name)
            _log(f"  {code_src.name}/{name} → runtime/tools/")

    src_env = code_src / "env_config.py"
    if src_env.is_file():
        shutil.copy2(src_env, repo / "config" / "env_config.py")
        _log(f"  {code_src.name}/env_config.py → runtime/config/")

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
    # Плейсхолдеры из шаблона не считаем ключами
    if v in ("hf_", "sk-", "hf_...", "sk-..."):
        return
    if v and not os.environ.get(name):
        os.environ[name] = v


def load_secrets(repo: Path | None = None) -> None:
    """HF / OpenAI: переменные ячейки → .env → Kaggle Secrets."""
    _set_secret("HF_TOKEN", HF_TOKEN)
    _set_secret("OPENAI_API_KEY", OPENAI_API_KEY)
    _set_secret("OPENAI_BASE_URL", OPENAI_BASE_URL)
    _set_secret("OPENAI_MODEL", OPENAI_MODEL)

    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ.setdefault("OPENAI_BASE_URL", "https://api.agentplatform.ru/v1")
    if not os.environ.get("OPENAI_MODEL"):
        os.environ.setdefault("OPENAI_MODEL", "openai/gpt-5.5")

    try:
        from dotenv import load_dotenv

        for base in (repo, RUNTIME, Path(CODE_DATASET_PATH) if CODE_DATASET_PATH else None):
            if base is None:
                continue
            for env_file in (Path(base) / ".env", Path(base) / "config" / ".env"):
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


def run_pipeline_subprocess(repo: Path) -> Path:
    ke = repo / "kaggle_entry.py"
    use_ke = ke.is_file() and "def run_kaggle" in ke.read_text(encoding="utf-8")
    if not use_ke:
        _log("  (нет run_kaggle в kaggle_entry → main.run, медленнее по VRAM)")

    _write_child_script(repo, use_ke)
    env = os.environ.copy()
    video_path = _resolve_video_path()
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
load_secrets(repo)

subprocess.run(["ffmpeg", "-version"], capture_output=True)

result = run_pipeline_subprocess(repo)
_log(f"=== Готово ===\nСкачайте: {result}")
