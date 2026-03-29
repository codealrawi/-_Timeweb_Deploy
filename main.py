"""
МедПлатформа — FastAPI Backend с PostgreSQL
Автор: Аль-Раве Мустафа Исам Табит · РГСУ · спец. 2.3.5
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import time, hashlib, secrets, os
from datetime import datetime, timedelta

from services.moderation_service import ContentModerator
from services.recommendation_service import HybridRecommender, generate_demo_data
from services.load_testing import LoadTester, AnomalyDetector
from database import create_pool, close_pool, get_pool, check_connection

# ─── Приложение ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="МедПлатформа API",
    description="Q&A-платформа для пациентов и врачей. Диссертационный прототип.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Жизненный цикл (startup / shutdown) ─────────────────────────────────────
@app.on_event("startup")
async def startup():
    await create_pool()
    # Инициализируем ML-сервисы
    app.state.moderator = ContentModerator()
    app.state.moderator.train()
    items, interactions = generate_demo_data()
    app.state.recommender = HybridRecommender(alpha=0.4)
    app.state.recommender.fit(items, interactions)
    app.state.items_meta = {it["id"]: it for it in items}

@app.on_event("shutdown")
async def shutdown():
    await close_pool()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─── Статический фронтенд ─────────────────────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

@app.get("/app", include_in_schema=False)
async def frontend():
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    from fastapi.responses import HTMLResponse
    return HTMLResponse("<h2>🏥 Фронтенд не найден</h2>", status_code=404)

# ─── Pydantic модели ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class PostCreate(BaseModel):
    title: str = Field(..., min_length=5, max_length=500)
    body:  str = Field(..., min_length=10)
    tags:  List[str] = []

class ModerationRequest(BaseModel):
    text: str = Field(..., min_length=1)

# ─── Авторизация ──────────────────────────────────────────────────────────────
# Простой in-memory кеш токенов (для прототипа)
_tokens: dict = {}

async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    pool = Depends(get_pool)
):
    if not creds:
        raise HTTPException(status_code=401, detail="Не авторизован")
    token = creds.credentials
    if token not in _tokens:
        raise HTTPException(status_code=401, detail="Токен недействителен")
    return _tokens[token]

# ─── Эндпоинты ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "МедПлатформа API",
        "version": "2.0.0",
        "status": "running",
        "database": "postgresql",
        "docs": "/docs",
        "app": "/app",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health(pool = Depends(get_pool)):
    db_ok = await check_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": db_ok,
        "moderator": True,
        "recommender": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/auth/login")
async def login(req: LoginRequest, pool = Depends(get_pool)):
    password_hash = hashlib.sha256(req.password.encode()).hexdigest()
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, name, role FROM users WHERE id=$1 AND password_hash=$2",
            req.username, password_hash
        )
    if not user:
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    token = secrets.token_hex(32)
    _tokens[token] = dict(user)
    return {"token": token, "user": dict(user)}

@app.get("/posts")
async def get_posts(
    limit: int = 20,
    offset: int = 0,
    pool = Depends(get_pool)
):
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                p.id, p.title, p.body, p.status,
                p.likes_count, p.created_at,
                u.id AS author_id, u.name AS author, u.role,
                array_agg(t.name) FILTER (WHERE t.name IS NOT NULL) AS tags
            FROM posts p
            JOIN users u ON u.id = p.author_id
            LEFT JOIN post_tags pt ON pt.post_id = p.id
            LEFT JOIN tags t ON t.id = pt.tag_id
            WHERE p.status = 'approved'
            GROUP BY p.id, u.id
            ORDER BY p.created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)

        total = await conn.fetchval(
            "SELECT COUNT(*) FROM posts WHERE status='approved'"
        )
    return {
        "posts": [dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.post("/posts", status_code=201)
async def create_post(
    post: PostCreate,
    user = Depends(get_current_user),
    pool = Depends(get_pool)
):
    # Модерация
    mod = app.state.moderator.moderate(post.title + " " + post.body)
    status_val = "approved" if mod.label == "approved" else \
                 "suspicious" if mod.label == "suspicious" else "blocked"

    post_id = f"p{int(time.time() * 1000)}"

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("""
                INSERT INTO posts (id, author_id, title, body, status, mod_level, mod_conf)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, post_id, user["id"], post.title, post.body,
                status_val, mod.level, float(mod.confidence))

            # Теги
            for tag_name in post.tags[:10]:
                tag_name = tag_name.strip().lower()
                if not tag_name:
                    continue
                tag_id = await conn.fetchval(
                    "INSERT INTO tags(name) VALUES($1) ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name RETURNING id",
                    tag_name
                )
                await conn.execute(
                    "INSERT INTO post_tags(post_id, tag_id) VALUES($1,$2) ON CONFLICT DO NOTHING",
                    post_id, tag_id
                )

            # Лог модерации
            await conn.execute("""
                INSERT INTO moderation_log (post_id, input_text, label, confidence, level, reasons)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, post_id, post.title + " " + post.body,
                mod.label, float(mod.confidence), mod.level,
                list(mod.reasons) if hasattr(mod, 'reasons') else [])

            # Обновляем счётчик постов пользователя
            await conn.execute(
                "UPDATE users SET posts_count = posts_count + 1 WHERE id=$1",
                user["id"]
            )

    return {
        "id": post_id,
        "status": status_val,
        "moderation": {
            "label": mod.label,
            "confidence": mod.confidence,
            "level": mod.level
        }
    }

@app.post("/posts/{post_id}/like")
async def like_post(
    post_id: str,
    user = Depends(get_current_user),
    pool = Depends(get_pool)
):
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT 1 FROM likes WHERE user_id=$1 AND post_id=$2",
            user["id"], post_id
        )
        if existing:
            await conn.execute(
                "DELETE FROM likes WHERE user_id=$1 AND post_id=$2",
                user["id"], post_id
            )
            await conn.execute(
                "UPDATE posts SET likes_count = likes_count - 1 WHERE id=$1",
                post_id
            )
            return {"liked": False}
        else:
            await conn.execute(
                "INSERT INTO likes(user_id, post_id) VALUES($1,$2) ON CONFLICT DO NOTHING",
                user["id"], post_id
            )
            await conn.execute(
                "UPDATE posts SET likes_count = likes_count + 1 WHERE id=$1",
                post_id
            )
            return {"liked": True}

@app.post("/moderation/check")
async def check_moderation(req: ModerationRequest):
    mod = app.state.moderator.moderate(req.text)
    return {
        "label":      mod.label,
        "confidence": mod.confidence,
        "level":      mod.level,
    }

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    top_k: int = 5,
    pool = Depends(get_pool)
):
    # Получаем просмотренные посты пользователя
    async with pool.acquire() as conn:
        liked = await conn.fetch(
            "SELECT post_id FROM likes WHERE user_id=$1 LIMIT 20",
            user_id
        )
    liked_ids = [r["post_id"] for r in liked]

    recs = app.state.recommender.recommend(user_id, liked_ids=liked_ids, top_k=top_k)
    return {"user_id": user_id, "recommendations": recs}

@app.get("/users")
async def get_users(pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, role, posts_count, likes_count, is_anomalous FROM users ORDER BY posts_count DESC"
        )
    return {"users": [dict(r) for r in rows]}

@app.get("/stats")
async def get_stats(pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM posts WHERE status='approved') AS posts,
                (SELECT COUNT(*) FROM users)                          AS users,
                (SELECT COUNT(*) FROM likes)                          AS likes,
                (SELECT COUNT(*) FROM tags)                           AS tags,
                (SELECT ROUND(
                    COUNT(*) FILTER (WHERE status='approved')::numeric /
                    NULLIF(COUNT(*),0) * 100, 1
                ) FROM posts)                                          AS moderation_pct
        """)
    return dict(stats)
