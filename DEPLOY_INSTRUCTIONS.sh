# ИНСТРУКЦИЯ: Деплой МедПлатформа на Timeweb Cloud
# ====================================================
# Аль-Раве Мустафа Исам Табит, РГСУ, спец. 2.3.5

# ШАГ 1 — Купить VPS на Timeweb Cloud
# ─────────────────────────────────────
# 1. Зайти на https://timeweb.cloud
# 2. Зарегистрироваться (по номеру телефона РФ)
# 3. Создать облачный сервер:
#    - ОС: Ubuntu 22.04 LTS
#    - CPU: 1 vCPU, RAM: 1 GB (тариф ~199 руб/мес)
#    - Регион: Россия (Москва)
# 4. Запомнить IP-адрес сервера (например: 185.xxx.xxx.xxx)
# 5. Пароль от root придёт на email

# ШАГ 2 — Подключиться к серверу
# ────────────────────────────────
# На Windows: скачать PuTTY или использовать Windows Terminal
# На Mac/Linux: открыть Terminal

ssh root@ВАШ_IP

# ШАГ 3 — Установить зависимости на сервере
# ──────────────────────────────────────────
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git nginx curl

# ШАГ 4 — Загрузить код на сервер
# ─────────────────────────────────
# Вариант А: через SCP (с вашего ПК, не с сервера)
# scp -r ./timeweb_deploy/ root@ВАШ_IP:/opt/medplatforma/

# Вариант Б: создать файлы вручную на сервере
mkdir -p /opt/medplatforma
cd /opt/medplatforma

# ШАГ 5 — Установить Python-зависимости
# ───────────────────────────────────────
cd /opt/medplatforma
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn[standard] pydantic python-multipart

# ШАГ 6 — Запустить FastAPI (тест)
# ──────────────────────────────────
cd /opt/medplatforma
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000

# Проверить в браузере: http://ВАШ_IP:8000/docs
# Если видите Swagger UI — сервер работает!
# Нажмите Ctrl+C для остановки

# ШАГ 7 — Запустить как системный сервис (автозапуск)
# ─────────────────────────────────────────────────────
cat > /etc/systemd/system/medplatforma.service << 'SERVICE'
[Unit]
Description=МедПлатформа FastAPI
After=network.target

[Service]
User=root
WorkingDirectory=/opt/medplatforma
ExecStart=/opt/medplatforma/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable medplatforma
systemctl start medplatforma
systemctl status medplatforma   # должно быть: active (running)

# ШАГ 8 — Настроить Nginx (порт 80)
# ───────────────────────────────────
cat > /etc/nginx/sites-available/medplatforma << 'NGINX'
server {
    listen 80;
    server_name ВАШ_IP;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 60s;
    }
}
NGINX

ln -s /etc/nginx/sites-available/medplatforma /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# Теперь API доступен по адресу: http://ВАШ_IP/docs

# ШАГ 9 — Открыть порты в файрволле Timeweb
# ────────────────────────────────────────────
# В панели Timeweb → Сервер → Файрволл → Добавить правило:
# Порт 80 (HTTP)  — разрешить входящий
# Порт 443 (HTTPS) — разрешить входящий (если нужен SSL)
# Порт 8000 — можно закрыть (Nginx проксирует)

# ШАГ 10 — Проверить что всё работает
# ─────────────────────────────────────
curl http://ВАШ_IP/health
# Ожидаемый ответ: {"status":"ok","moderator":true,"recommender":true}

curl http://ВАШ_IP/posts
# Ожидаемый ответ: {"posts":[...], "total":3}

# ====================================================
# НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ (запускать со своего ПК)
# ====================================================

# ШАГ 11 — Установить Locust на СВОЁМ компьютере
# ────────────────────────────────────────────────
# (не на сервере!)
pip install locust

# ШАГ 12 — Запустить нагрузочный тест
# ─────────────────────────────────────
locust -f locustfile.py --host=http://ВАШ_IP

# Открыть браузер: http://localhost:8089
# Ввести параметры:
#   Number of users: 50   → нажать Start
#   Spawn rate: 10
#   Подождать 5 минут → сделать скриншот
#
# Повторить для: 100, 200, 500, 1000, 1500 пользователей

# ШАГ 13 — Сохранить результаты
# ──────────────────────────────
# Locust автоматически сохраняет CSV:
locust -f locustfile.py --host=http://ВАШ_IP \
  --headless \
  --users 200 \
  --spawn-rate 10 \
  --run-time 5m \
  --csv=results_200rps

# Файлы: results_200rps_stats.csv, results_200rps_failures.csv
# Это РЕАЛЬНЫЕ данные для диссертации!
