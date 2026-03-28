
"""
МедПлатформа — FastAPI Backend
Социально-ориентированная веб-система (медицинский домен)
Автор: Аль-Раве Мустафа Исам Табит
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List
import time, hashlib, secrets, json
from datetime import datetime

from services.moderation_service import ContentModerator
from services.recommendation_service import HybridRecommender, generate_demo_data
from services.load_testing import LoadTester, AnomalyDetector

# ─── Приложение ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="МедПлатформа API",
    description="Q&A-платформа для пациентов и врачей. Диссертационный прототип.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─── In-memory база данных (для прототипа) ───────────────────────────────────
USERS_DB = {
    "user1": {"id": "user1", "name": "Д-р Петров А.С.", "role": "doctor",
              "password_hash": hashlib.sha256(b"doctor123").hexdigest(), "posts": 42, "likes": 287},
    "user2": {"id": "user2", "name": "Мария Иванова",   "role": "patient",
              "password_hash": hashlib.sha256(b"patient123").hexdigest(), "posts": 8,  "likes": 34},
    "admin": {"id": "admin", "name": "Администратор",   "role": "admin",
              "password_hash": hashlib.sha256(b"admin123").hexdigest(),   "posts": 0,  "likes": 0},
}

POSTS_DB = [
    {"id": "p1", "author_id": "user1", "author": "Д-р Петров А.С.", "role": "doctor",
     "title": "Рекомендации по реабилитации после инфаркта",
     "body": "После перенесённого инфаркта миокарда крайне важно соблюдать программу кардиореабилитации. Первые 6 недель — ограниченная физическая активность, диета с ограничением соли.",
     "tags": ["кардиология", "реабилитация"], "likes": 34, "status": "approved", "created_at": "2025-01-15T10:00:00"},
    {"id": "p2", "author_id": "user2", "author": "Мария Иванова", "role": "patient",
     "title": "Как правильно принимать метформин при диабете 2 типа?",
     "body": "Врач назначил метформин. Когда лучше принимать — до или после еды? Есть ли побочные эффекты на желудок?",
     "tags": ["диабет", "препараты"], "likes": 12, "status": "approved", "created_at": "2025-01-15T08:00:00"},
    {"id": "p3", "author_id": "user2", "author": "Надежда С.", "role": "patient",
     "title": "Боли в пояснице после долгого сидения",
     "body": "Работаю удалённо по 10 часов. Начались сильные боли в пояснице. Какого специалиста посетить?",
     "tags": ["неврология", "позвоночник"], "likes": 8, "status": "approved", "created_at": "2025-01-14T15:00:00"},
]

SESSIONS = {}   # token → user_id
AUDIT_LOG = []  # события безопасности

# ─── Инициализация ML-сервисов ────────────────────────────────────────────────
print("[INIT] Обучение модуля модерации...")
moderator = ContentModerator()
moderator.train()

print("[INIT] Обучение рекомендательной системы...")
items, interactions = generate_demo_data()
recommender = HybridRecommender(alpha=0.4)
recommender.fit(items, interactions)

print("[INIT] Готово. Сервер запущен.")

# ─── МОДЕЛИ ЗАПРОСОВ / ОТВЕТОВ ────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class PostCreate(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    body:  str = Field(..., min_length=10)
    tags:  List[str] = []

class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class LoadTestRequest(BaseModel):
    rps_levels: List[int] = [50, 100, 200, 500, 1000, 1500]

# ─── Авторизация ─────────────────────────────────────────────────────────────
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    token = credentials.credentials
    user_id = SESSIONS.get(token)
    if not user_id:
        return None
    return USERS_DB.get(user_id)

def require_user(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    return user

def require_admin(user=Depends(get_current_user)):
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Требуются права администратора")
    return user

# ═══════════════════════════════════════════════════════════════════════════════
# ЭНДПОИНТЫ
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Система"])
def root():
    return {
        "service": "МедПлатформа API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/health", tags=["Система"])
def health():
    return {"status": "ok", "moderator": moderator._trained, "recommender": recommender._fitted}

# ─── Авторизация ─────────────────────────────────────────────────────────────
@app.post("/auth/login", tags=["Авторизация"])
def login(req: LoginRequest):
    user = USERS_DB.get(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    if user["password_hash"] != hashlib.sha256(req.password.encode()).hexdigest():
        AUDIT_LOG.append({"event": "login_fail", "username": req.username, "ts": time.time()})
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    token = secrets.token_hex(32)
    SESSIONS[token] = user["id"]
    AUDIT_LOG.append({"event": "login_ok", "user_id": user["id"], "ts": time.time()})
    return {"token": token, "user": {k: v for k, v in user.items() if k != "password_hash"}}

@app.post("/auth/logout", tags=["Авторизация"])
def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials and credentials.credentials in SESSIONS:
        del SESSIONS[credentials.credentials]
    return {"status": "ok"}

# ─── Публикации ──────────────────────────────────────────────────────────────
@app.get("/posts", tags=["Публикации"])
def get_posts(skip: int = 0, limit: int = 20, tag: Optional[str] = None):
    posts = [p for p in POSTS_DB if p["status"] == "approved"]
    if tag:
        posts = [p for p in posts if tag in p.get("tags", [])]
    return {"posts": posts[skip:skip+limit], "total": len(posts)}

@app.get("/posts/{post_id}", tags=["Публикации"])
def get_post(post_id: str):
    post = next((p for p in POSTS_DB if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Публикация не найдена")
    return post

@app.post("/posts", tags=["Публикации"])
def create_post(req: PostCreate, user=Depends(require_user)):
    text = f"{req.title} {req.body}"
    mod_result = moderator.moderate(text)

    new_post = {
        "id": f"p{len(POSTS_DB)+1}",
        "author_id": user["id"],
        "author": user["name"],
        "role": user["role"],
        "title": req.title,
        "body": req.body,
        "tags": req.tags,
        "likes": 0,
        "status": mod_result.label,
        "moderation": {
            "label": mod_result.label,
            "confidence": mod_result.confidence,
            "level": mod_result.level,
            "reasons": mod_result.reasons,
        },
        "created_at": datetime.now().isoformat(),
    }
    POSTS_DB.append(new_post)
    return new_post

@app.post("/posts/{post_id}/like", tags=["Публикации"])
def like_post(post_id: str, user=Depends(require_user)):
    post = next((p for p in POSTS_DB if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Публикация не найдена")
    post["likes"] += 1
    return {"likes": post["likes"]}

# ─── Модерация ───────────────────────────────────────────────────────────────
@app.post("/moderation/check", tags=["Модерация"])
def check_content(req: ModerateRequest):
    start = time.time()
    result = moderator.moderate(req.text)
    latency_ms = round((time.time() - start) * 1000, 2)
    return {
        "label": result.label,
        "confidence": result.confidence,
        "level": result.level,
        "reasons": result.reasons,
        "scores": result.scores,
        "latency_ms": latency_ms,
    }

@app.get("/moderation/queue", tags=["Модерация"])
def get_moderation_queue(user=Depends(require_admin)):
    return {"queue": [p for p in POSTS_DB if p["status"] == "suspicious"]}

@app.post("/moderation/resolve/{post_id}", tags=["Модерация"])
def resolve_post(post_id: str, action: str, user=Depends(require_admin)):
    post = next((p for p in POSTS_DB if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Публикация не найдена")
    if action not in ("approve", "block"):
        raise HTTPException(status_code=400, detail="action должен быть 'approve' или 'block'")
    post["status"] = "approved" if action == "approve" else "blocked"
    AUDIT_LOG.append({"event": f"moderation_{action}", "post_id": post_id,
                      "admin": user["id"], "ts": time.time()})
    return {"status": post["status"]}

@app.get("/moderation/metrics", tags=["Модерация"])
def get_moderation_metrics():
    return moderator.evaluate()

# ─── Рекомендации ────────────────────────────────────────────────────────────
@app.get("/recommendations/{user_id}", tags=["Рекомендации"])
def get_recommendations(user_id: str, top_k: int = 10):
    liked = [p["id"] for p in POSTS_DB if p.get("author_id") == user_id]
    recs = recommender.recommend(user_id, liked_ids=liked or ["p1","p2"], top_k=top_k)
    # Обогащаем метаданными
    item_meta = {it["id"]: it for it in items}
    for r in recs:
        meta = item_meta.get(r["item_id"], {})
        r["title"] = meta.get("title", r["item_id"])
        r["tags"]  = meta.get("tags", [])
    return {"user_id": user_id, "recommendations": recs, "algorithm": "CBF+SVD hybrid (alpha=0.4)"}

@app.get("/recommendations/metrics/evaluate", tags=["Рекомендации"])
def get_recommendation_metrics():
    return recommender.evaluate_metrics(interactions[:10])

# ─── Нагрузочные тесты ───────────────────────────────────────────────────────
@app.post("/performance/load-test", tags=["Производительность"])
def run_load_test(req: LoadTestRequest):
    tester = LoadTester()
    results = tester.run(req.rps_levels)
    return {
        "results": [{"rps": r.rps, "mean_ms": r.mean_ms, "p95_ms": r.p95_ms,
                     "error_rate": r.error_rate, "cpu_pct": r.cpu_pct} for r in results],
        "comparison": tester.compare_architectures([200, 500, 1000]),
    }

# ─── Пользователи и аномалии ─────────────────────────────────────────────────
@app.get("/users", tags=["Пользователи"])
def get_users(user=Depends(require_admin)):
    return {"users": [{k: v for k, v in u.items() if k != "password_hash"}
                      for u in USERS_DB.values()]}

@app.post("/users/anomaly-detection", tags=["Пользователи"])
def detect_anomalies(user=Depends(require_admin)):
    import random
    rng = random.Random(7)
    activity = {f"u{i}": {"posts_per_day": max(0, rng.gauss(2,1)),
                           "likes_per_day": max(0, rng.gauss(10,3)),
                           "watch_time": max(0, rng.gauss(30,10)),
                           "reports_received": max(0, rng.gauss(0.1,0.1))}
                for i in range(1, 21)}
    activity["bot1"] = {"posts_per_day": 180, "likes_per_day": 500,
                        "watch_time": 0.5, "reports_received": 12}
    activity["bot2"] = {"posts_per_day": 95,  "likes_per_day": 300,
                        "watch_time": 1.0, "reports_received": 8}
    detector = AnomalyDetector(threshold_percentile=90)
    anomalous, errors = detector.detect_anomalous_users(activity)
    return {
        "anomalous_users": anomalous,
        "threshold": round(detector.threshold_, 4),
        "top_errors": sorted(
            [{"user_id": uid, "error": round(e, 4)} for uid, e in errors.items()],
            key=lambda x: x["error"], reverse=True
        )[:10],
    }

# ─── Аудит ───────────────────────────────────────────────────────────────────
@app.get("/admin/audit-log", tags=["Администрирование"])
def get_audit_log(user=Depends(require_admin)):
    return {"log": AUDIT_LOG[-50:], "total": len(AUDIT_LOG)}

@app.get("/admin/stats", tags=["Администрирование"])
def get_stats(user=Depends(require_admin)):
    return {
        "posts_total": len(POSTS_DB),
        "posts_approved": sum(1 for p in POSTS_DB if p["status"] == "approved"),
        "posts_blocked": sum(1 for p in POSTS_DB if p["status"] == "blocked"),
        "posts_suspicious": sum(1 for p in POSTS_DB if p["status"] == "suspicious"),
        "users_total": len(USERS_DB),
        "active_sessions": len(SESSIONS),
    }
