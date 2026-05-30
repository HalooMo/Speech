# SpeechLab на Kaggle

## Одна ячейка (рекомендуется)

Скопируйте **целиком** файл [`kaggle_one_cell.py`](../kaggle_one_cell.py) в одну ячейку ноутбука.

Тот же файл: [`kaggle_run_cell.py`](../kaggle_run_cell.py).

### Что делает ячейка сама

1. Ставит зависимости с пином `numpy<2.2` (без поломки scipy)
2. Находит код SpeechLab в `/kaggle/input/...` (или `CODE_DATASET_PATH`)
3. Копирует в `/kaggle/working/speechlab_runtime` и патчит старый dataset
4. Загружает Secrets (`HF_TOKEN`, `OPENAI_API_KEY`)
5. Гоняет полный пайплайн дубляжа → `{project}_dubbed.mp4`

### Настройки в начале файла

```python
PROJECT_NAME = "kaggle_dub"
VIDEO_PATH = "/kaggle/input/.../video.mp4"
SOURCE_LANG = "en"
TARGET_LANG = "ru"
CODE_DATASET_PATH = "/kaggle/input/datasets/salimkazhaerov/speechlab-coded"  # или авто-поиск
SKIP_DEMUCS = False
WHISPER_MODEL = "medium"
```

### Перед первым запуском

1. GPU (T4 / P100)
2. Dataset с кодом (`main.py`, `test.py`, …)
3. Dataset с видео
4. Secrets: `HF_TOKEN`, `OPENAI_API_KEY`
5. **Session → Restart session** (если раньше уже ломали numpy 2.4)

### Результат

`/kaggle/working/speechlab_projects/<PROJECT_NAME>/<PROJECT_NAME>_dubbed.mp4`

---

## Ограничения T4

| Параметр | Рекомендация |
|----------|----------------|
| Whisper | `medium`, `float16` |
| Demucs | `SKIP_DEMUCS = True` для быстрого теста |
| Видео | 1–3 минуты для первого прогона |

## Старый способ

`kaggle_entry.py` + отдельная установка — не нужен, если используете `kaggle_one_cell.py`.
