# SpeechLab — деплой на облачный сервер

Пошаговое руководство по развёртыванию HTTP API дубляжа на Linux-сервере с GPU.
Пример из продакшена: API на `app.app.vandum.ru`, сервер с Ubuntu, путь `/opt/speechlab`.

Архитектура:

```text
Клиент (curl / приложение)
        │  HTTPS :443
        ▼
   Nginx (TLS, лимит upload 500M)
        │  HTTP 127.0.0.1:8080
        ▼
   Gunicorn (workers=1) → Flask API
        │  subprocess
        ▼
   server/run_job.py → main.py (Demucs, WhisperX, LLM, Qwen TTS)
```

Одна задача GPU одновременно. Статусы задач хранятся в `server/data/jobs/*.json`.

---

## 1. Требования к серверу

| Ресурс | Минимум | Рекомендуется |
|--------|---------|---------------|
| ОС | Ubuntu 22.04 / 24.04 LTS | то же |
| GPU | NVIDIA с CUDA | T4 / A10 / L4, ≥16 GB VRAM |
| RAM | 16 GB | 32 GB |
| Диск | 40 GB свободно | 60+ GB (модели HF + проекты) |
| Сеть | публичный IP, домен | A-запись на IP сервера |

Нужны: **Python 3.10+**, **CUDA-драйвер**, **ffmpeg**, **sox**, **libsndfile**.

Проверка GPU:

```bash
nvidia-smi
```

---

## 2. Подключение и базовая подготовка

```bash
ssh root@ВАШ_IP
apt update && apt upgrade -y
apt install -y git curl wget build-essential \
  ffmpeg sox libsox-dev libsndfile1 \
  python3 python3-venv python3-pip \
  nginx certbot python3-certbot-nginx
```

Создать пользователя для сервиса (не root):

```bash
useradd -m -s /bin/bash speechlab
```

---

## 3. Код проекта

### Вариант A — git (предпочтительно)

```bash
mkdir -p /opt/speechlab
chown speechlab:speechlab /opt/speechlab
sudo -u speechlab git clone https://github.com/HalooMo/Speech.git /opt/speechlab
cd /opt/speechlab
```

### Вариант B — копирование с локальной машины

На Windows (PowerShell):

```powershell
scp -r C:\Users\Имя\Projects\Speech root@ВАШ_IP:/opt/speechlab
ssh root@ВАШ_IP "chown -R speechlab:speechlab /opt/speechlab"
```

---

## 4. Python-окружение и зависимости

```bash
cd /opt/speechlab
sudo -u speechlab python3 -m venv .venv
sudo -u speechlab .venv/bin/pip install --upgrade pip wheel
sudo -u speechlab .venv/bin/pip install -r requirements.txt -r requirements-server.txt
```

Установка PyTorch с CUDA (если `pip install torch` не подхватил GPU):

```bash
sudo -u speechlab .venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Проверка:

```bash
sudo -u speechlab .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
# True
```

---

## 5. Секреты и конфигурация

```bash
sudo -u speechlab cp config/.env.example config/.env
nano /opt/speechlab/config/.env
```

Обязательно заполнить:

```ini
# Hugging Face — pyannote, модели TTS
HF_TOKEN=hf_...

# LLM — разметка и перевод
OPENAI_API_KEY=sk_...
OPENAI_BASE_URL=https://api.agentplatform.ru/v1
OPENAI_MODEL=openai/gpt-5.5

# Продакшен API
SPEECHLAB_ENV=production
SPEECHLAB_API_KEY=длинный_случайный_ключ_32+_символов
SPEECHLAB_REQUIRE_API_KEY=1
SPEECHLAB_GUNICORN_BIND=127.0.0.1:8080
SPEECHLAB_MAX_UPLOAD_MB=500

# Пути (по умолчанию можно не менять)
SPEECHLAB_PROJECTS_ROOT=/opt/speechlab/data/projects
SPEECHLAB_UPLOAD_DIR=/opt/speechlab/server/uploads
SPEECHLAB_JOBS_DIR=/opt/speechlab/server/data/jobs
SPEECHLAB_LOGS_DIR=/opt/speechlab/server/data/logs
```

На Hugging Face принять условия модели:
https://huggingface.co/pyannote/speaker-diarization-community-1

Убрать Windows-переносы строк (если правили `.env` на Windows):

```bash
sed -i 's/\r$//' /opt/speechlab/config/.env
```

Создать каталоги:

```bash
sudo -u speechlab mkdir -p \
  /opt/speechlab/data/projects \
  /opt/speechlab/server/uploads/voice_samples \
  /opt/speechlab/server/data/jobs \
  /opt/speechlab/server/data/logs
```

---

## 6. Systemd — Gunicorn

```bash
cp /opt/speechlab/deploy/speechlab.service /etc/systemd/system/speechlab.service
systemctl daemon-reload
systemctl enable speechlab
systemctl start speechlab
systemctl status speechlab
```

Файл `deploy/speechlab.service` уже содержит:

- `User=speechlab`
- `WorkingDirectory=/opt/speechlab`
- `EnvironmentFile=/opt/speechlab/config/.env`
- `ExecStart=.../gunicorn -c deploy/gunicorn.conf.py wsgi:app`

Проверка API локально (с сервера):

```bash
curl -s http://127.0.0.1:8080/health
curl -s -H "X-API-Key: ВАШ_КЛЮЧ" http://127.0.0.1:8080/api/v1/jobs
```

---

## 7. Nginx + HTTPS (Let's Encrypt)

Подставить свой домен вместо `app.vandum.ru`:

```bash
cp /opt/speechlab/deploy/nginx.conf.example /etc/nginx/sites-available/speechlab
# при другом домене — заменить app.vandum.ru в server_name и путях ssl_certificate
ln -sf /etc/nginx/sites-available/speechlab /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
```

Получить сертификат:

```bash
certbot --nginx -d app.vandum.ru
systemctl reload nginx
```

В конфиге nginx важно:

- `client_max_body_size 500M;` — загрузка видео
- `proxy_set_header X-API-Key $http_x_api_key;` — проброс API-ключа
- `proxy_set_header Authorization $http_authorization;`

---

## 7.1. Перенос API на поддомен `app.vandum.ru`

Если API раньше был на `vandum.ru`, а нужен поддомен:

**1. DNS** (панель регистратора, TTL 300–600):

| Тип | Имя | Значение |
|-----|-----|----------|
| A | `app` | IP сервера (тот же, что у `vandum.ru`) |

Проверка (с любой машины):

```bash
dig +short app.vandum.ru
```

**2. Nginx на сервере:**

```bash
nano /etc/nginx/sites-available/speechlab
```

Заменить `server_name vandum.ru` → `server_name app.vandum.ru` (в блоках `:443` и `:80`).

**3. Сертификат для поддомена:**

```bash
nginx -t
certbot --nginx -d app.vandum.ru
systemctl reload nginx
```

**4. Редирект со старого URL (опционально):**

В тот же файл или отдельный `sites-available/vandum-redirect`:

```nginx
server {
    listen 443 ssl http2;
    server_name vandum.ru www.vandum.ru;
    ssl_certificate     /etc/letsencrypt/live/vandum.ru/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vandum.ru/privkey.pem;
    return 301 https://app.vandum.ru$request_uri;
}
```

**5. Проверка:**

```bash
curl -s https://app.vandum.ru/health
curl -s -H "X-API-Key: КЛЮЧ" https://app.vandum.ru/api/v1/jobs
```

Gunicorn и `config/.env` менять не нужно — API слушает `127.0.0.1:8080`, меняется только nginx/DNS.

**6. Клиенты** — заменить базовый URL на `https://app.vandum.ru` (см. `for_client.txt`).

---

## 8. Первый тест дубляжа

С локальной машины (Windows CMD):

```cmd
curl.exe -X POST "https://app.vandum.ru/api/v1/dub" ^
  -H "X-API-Key: ВАШ_КЛЮЧ" ^
  -F "project_name=demo" ^
  -F "source_language=en" ^
  -F "target_language=ru" ^
  -F "video=@C:\path\to\video.mp4"
```

Ответ `202` с полем `"id"`. Опрос статуса:

```cmd
curl.exe -H "X-API-Key: ВАШ_КЛЮЧ" https://app.vandum.ru/api/v1/jobs/JOB_ID
```

Скачивание готового MP4:

```cmd
curl.exe -H "X-API-Key: ВАШ_КЛЮЧ" ^
  -o "C:\Users\Имя\Downloads\demo_dubbed.mp4" ^
  "https://app.vandum.ru/api/v1/jobs/JOB_ID/download"
```

Подробнее для клиентов — файл `for_client.txt` в корне репозитория.

---

## 9. Обновление после `git pull`

```bash
systemctl stop speechlab
cd /opt/speechlab
sudo -u speechlab git pull
sudo -u speechlab .venv/bin/pip install -r requirements.txt -r requirements-server.txt
sudo -u speechlab .venv/bin/python -m py_compile main.py server/routes.py tools/dubbing.py
systemctl start speechlab
systemctl status speechlab
```

При конфликтах локальных правок на сервере:

```bash
git stash
git pull
# или откатить конкретные файлы:
git checkout -- server/config.py wsgi.py
```

---

## 10. Логи и диагностика

| Что | Команда |
|-----|---------|
| Статус сервиса | `systemctl status speechlab` |
| Логи Gunicorn | `journalctl -u speechlab -f` |
| Лог задачи | `tail -f /opt/speechlab/server/data/logs/job_JOBID.log` |
| Кто слушает 8080 | `ss -tlnp \| grep 8080` |
| Место на диске | `df -h` и `du -sh /opt/speechlab/* ~/.cache/huggingface` |
| API key в процессе | `tr '\0' '\n' < /proc/$(systemctl show speechlab -p MainPID --value)/environ \| grep SPEECHLAB_API_KEY` |

Список задач:

```bash
ls -la /opt/speechlab/server/data/jobs/
```

---

## 11. Типичные проблемы (из реального деплоя)

### 401 «Неверный API key»

1. Проверить ключ в `config/.env` — одна строка `SPEECHLAB_API_KEY=...`, без дубликатов из `.env.example`.
2. Убрать `\r`: `sed -i 's/\r$//' config/.env`.
3. Убедиться, что на 8080 слушает **systemd**, а не старый gunicorn:
   ```bash
   ss -tlnp | grep 8080
   pkill -f "gunicorn.*wsgi:app"   # если висит зомби
   systemctl restart speechlab
   ```
4. Проверить напрямую: `curl -H "X-API-Key: ..." http://127.0.0.1:8080/api/v1/jobs`.

### 415 Unsupported Media Type

Используйте `multipart/form-data` с `-F`, не `-d` для загрузки видео. Актуальный код `server/routes.py` это поддерживает.

### ModuleNotFoundError (soundfile, demucs, …)

```bash
sudo -u speechlab .venv/bin/pip install -r requirements.txt
apt install -y libsndfile1
```

### `demucs: not found` / `sox: not found`

```bash
sudo -u speechlab .venv/bin/pip install demucs
apt install -y sox libsox-dev ffmpeg
```

Пайплайн вызывает `python -m demucs`, PATH воркера включает `.venv/bin`.

### No space left on device

Модели Hugging Face занимают десятки GB.

```bash
df -h
du -sh /root/.cache/huggingface /home/speechlab/.cache/huggingface
```

Очистка кэша pip/HF, старых upload. Если диск расширен у провайдера, но раздел маленький:

```bash
growpart /dev/sda 1
resize2fs /dev/sda1
df -h
```

### 503 «Пайплайн уже выполняется»

На GPU одна задача. Дождаться `done` или проверить зависший job:

```bash
curl -H "X-API-Key: ..." http://127.0.0.1:8080/api/v1/jobs
```

### Задача в `error`

```bash
cat /opt/speechlab/server/data/jobs/JOB_ID.json
tail -100 /opt/speechlab/server/data/logs/job_JOB_ID.log
```

---

## 12. Безопасность

- `SPEECHLAB_API_KEY` — длинный случайный ключ, не коммитить в git.
- `config/.env` в `.gitignore`.
- Gunicorn слушает только `127.0.0.1:8080`, снаружи — только nginx с TLS.
- Firewall: открыть 22, 80, 443; закрыть 8080 снаружи.
- Пользователь `speechlab` без sudo.

```bash
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw enable
```

---

## 13. Структура на сервере

```text
/opt/speechlab/
├── .venv/                    # Python
├── config/.env               # секреты (не в git)
├── deploy/
│   ├── gunicorn.conf.py
│   ├── speechlab.service
│   └── nginx.conf.example
├── wsgi.py                   # точка входа Gunicorn
├── main.py                   # пайплайн
├── server/
│   ├── uploads/              # загруженные видео и voice samples
│   └── data/
│       ├── jobs/             # JSON статусы задач
│       └── logs/             # логи воркеров
└── data/projects/            # артефакты дубляжа + *_dubbed.mp4
```

---

## 14. Краткий чеклист

- [ ] GPU виден в `nvidia-smi`
- [ ] `ffmpeg`, `sox`, `libsndfile1` установлены
- [ ] venv + `pip install -r requirements.txt -r requirements-server.txt`
- [ ] `config/.env` заполнен, `SPEECHLAB_ENV=production`
- [ ] `HF_TOKEN` + доступ к pyannote на huggingface.co
- [ ] `systemctl status speechlab` — active
- [ ] `curl http://127.0.0.1:8080/health` — ok
- [ ] nginx + certbot, HTTPS работает
- [ ] `curl -H X-API-Key ... https://домен/health` — ok
- [ ] Тестовый POST `/api/v1/dub` → job → download MP4
