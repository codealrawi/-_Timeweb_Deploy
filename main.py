"""
МедПлатформа API v3.0
Автор: Аль-Раве Мустафа Исам Табит · РГСУ · спец. 2.3.5

Особенности:
  - PostgreSQL через asyncpg (если настроены переменные окружения)
  - Если БД недоступна — автоматически работает с in-memory данными
  - Приложение НИКОГДА не падает из-за БД
"""

import os, time, hashlib, secrets, logging
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import database as db
from services.moderation_service import ContentModerator
from services.recommendation_service import HybridRecommender, generate_demo_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Приложение ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="МедПлатформа API",
    version="3.0.0",
    description="Q&A платформа для пациентов и врачей · Диссертационный прототип",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ── In-memory хранилище (fallback) ────────────────────────────────────────────
_h = lambda p: hashlib.sha256(p.encode()).hexdigest()

MEM_USERS = {
    "user1": {"id":"user1","name":"Д-р Петров А.С.","role":"doctor",
              "ph":_h("doctor123"),"posts":42,"likes":287},
    "user2": {"id":"user2","name":"Мария Иванова","role":"patient",
              "ph":_h("patient123"),"posts":8,"likes":34},
    "admin": {"id":"admin","name":"Администратор","role":"admin",
              "ph":_h("admin123"),"posts":0,"likes":0},
}
MEM_POSTS = [
    {"id":"p1","author":"Д-р Петров А.С.","role":"doctor",
     "title":"Реабилитация после инфаркта миокарда",
     "body":"После инфаркта важна кардиореабилитация. Первые 6 недель — ограниченная физическая активность, диета с ограничением соли.",
     "tags":["кардиология","реабилитация"],"likes_count":34,"status":"approved"},
    {"id":"p2","author":"Мария Иванова","role":"patient",
     "title":"Как принимать метформин при диабете 2 типа?",
     "body":"Врач назначил метформин 500мг. Когда лучше принимать — до или после еды? Есть ли побочные эффекты?",
     "tags":["диабет","препараты"],"likes_count":12,"status":"approved"},
    {"id":"p3","author":"Д-р Семёнова Е.В.","role":"doctor",
     "title":"Профилактика ОРВИ у детей",
     "body":"Закаливание, проветривание, промывание носа физраствором и вакцинация от гриппа.",
     "tags":["педиатрия","ОРВИ"],"likes_count":56,"status":"approved"},
]
_TOKENS: dict = {}

# ── Жизненный цикл ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    # ML-сервисы (обязательно)
    app.state.mod = ContentModerator()
    app.state.mod.train()
    items, ints = generate_demo_data()
    app.state.rec = HybridRecommender(alpha=0.4)
    app.state.rec.fit(items, ints)
    logger.info("[INIT] ML-сервисы готовы")

    # БД (опционально — не падаем если недоступна)
    ok = await db.init_pool()
    app.state.use_db = ok
    logger.info(f"[INIT] БД: {'PostgreSQL ✓' if ok else 'in-memory (БД недоступна)'}")

@app.on_event("shutdown")
async def shutdown():
    await db.close_pool()

# ── Статический фронтенд ─────────────────────────────────────────────────────
_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

@app.get("/app", include_in_schema=False)
async def frontend():
    idx = os.path.join(_static, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    from fastapi.responses import HTMLResponse
    return HTMLResponse("<h2>🏥 МедПлатформа — фронтенд не найден</h2>", 404)

# ── Модели ────────────────────────────────────────────────────────────────────
class LoginReq(BaseModel):
    username: str
    password: str

class PostReq(BaseModel):
    title: str = Field(..., min_length=5, max_length=500)
    body:  str = Field(..., min_length=10)
    tags:  List[str] = []

class ModReq(BaseModel):
    text: str

# ── Авторизация ───────────────────────────────────────────────────────────────
def get_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not creds or creds.credentials not in _TOKENS:
        raise HTTPException(401, "Не авторизован")
    return _TOKENS[creds.credentials]

# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service":   "МедПлатформа API",
        "version":   "3.0.0",
        "status":    "running",
        "database":  "postgresql" if app.state.use_db else "in-memory",
        "docs":      "/docs",
        "app":       "/app",
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/health")
async def health():
    db_status = await db.db_ok() if app.state.use_db else False
    return {
        "status":        "ok",
        "database":      db_status,
        "database_mode": "postgresql" if app.state.use_db else "in-memory",
        "moderator":     True,
        "recommender":   True,
        "timestamp":     datetime.utcnow().isoformat(),
    }

@app.post("/auth/login")
async def login(req: LoginReq):
    ph = hashlib.sha256(req.password.encode()).hexdigest()

    if app.state.use_db and db.pool:
        row = await db.pool.fetchrow(
            "SELECT id, name, role FROM users WHERE id=$1 AND password_hash=$2",
            req.username, ph,
        )
        if not row:
            raise HTTPException(401, "Неверный логин или пароль")
        user = dict(row)
    else:
        u = MEM_USERS.get(req.username)
        if not u or u["ph"] != ph:
            raise HTTPException(401, "Неверный логин или пароль")
        user = {"id": u["id"], "name": u["name"], "role": u["role"]}

    tok = secrets.token_hex(32)
    _TOKENS[tok] = user
    return {"token": tok, "user": user}

@app.get("/posts")
async def get_posts(limit: int = 20, offset: int = 0):
    if app.state.use_db and db.pool:
        rows = await db.pool.fetch("""
            SELECT p.id, p.title, p.body, p.status, p.likes_count,
                   u.name AS author, u.role,
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
        total = await db.pool.fetchval(
            "SELECT COUNT(*) FROM posts WHERE status='approved'")
        return {"posts": [dict(r) for r in rows], "total": total}
    else:
        sl = MEM_POSTS[offset:offset + limit]
        return {"posts": sl, "total": len(MEM_POSTS)}

@app.post("/posts", status_code=201)
async def create_post(req: PostReq, user=Depends(get_user)):
    mod = app.state.mod.moderate(req.title + " " + req.body)
    status = ("approved"   if mod.label == "approved"   else
              "suspicious" if mod.label == "suspicious" else "blocked")
    pid = f"p{int(time.time()*1000)}"

    if app.state.use_db and db.pool:
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO posts(id,author_id,title,body,status,mod_level,mod_conf)"
                    " VALUES($1,$2,$3,$4,$5,$6,$7)",
                    pid, user["id"], req.title, req.body,
                    status, mod.level, float(mod.confidence),
                )
                for tag in req.tags[:10]:
                    tag = tag.strip().lower()
                    if tag:
                        tid = await conn.fetchval(
                            "INSERT INTO tags(name) VALUES($1)"
                            " ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name RETURNING id",
                            tag,
                        )
                        await conn.execute(
                            "INSERT INTO post_tags(post_id,tag_id) VALUES($1,$2)"
                            " ON CONFLICT DO NOTHING",
                            pid, tid,
                        )
    else:
        MEM_POSTS.insert(0, {
            "id": pid, "author": user["name"], "role": user["role"],
            "title": req.title, "body": req.body,
            "tags": req.tags, "likes_count": 0, "status": status,
        })

    return {
        "id": pid, "status": status,
        "moderation": {"label": mod.label, "confidence": mod.confidence, "level": mod.level},
    }

@app.post("/posts/{post_id}/like")
async def like_post(post_id: str, user=Depends(get_user)):
    if app.state.use_db and db.pool:
        exists = await db.pool.fetchrow(
            "SELECT 1 FROM likes WHERE user_id=$1 AND post_id=$2",
            user["id"], post_id,
        )
        if exists:
            await db.pool.execute(
                "DELETE FROM likes WHERE user_id=$1 AND post_id=$2",
                user["id"], post_id,
            )
            await db.pool.execute(
                "UPDATE posts SET likes_count=likes_count-1 WHERE id=$1", post_id)
            return {"liked": False}
        else:
            await db.pool.execute(
                "INSERT INTO likes(user_id,post_id) VALUES($1,$2) ON CONFLICT DO NOTHING",
                user["id"], post_id,
            )
            await db.pool.execute(
                "UPDATE posts SET likes_count=likes_count+1 WHERE id=$1", post_id)
            return {"liked": True}
    return {"liked": True, "note": "in-memory mode"}

@app.post("/moderation/check")
async def moderation_check(req: ModReq):
    mod = app.state.mod.moderate(req.text)
    return {"label": mod.label, "confidence": mod.confidence, "level": mod.level}

@app.get("/recommendations/{user_id}")
async def recommendations(user_id: str, top_k: int = 5):
    liked = []
    if app.state.use_db and db.pool:
        rows = await db.pool.fetch(
            "SELECT post_id FROM likes WHERE user_id=$1 LIMIT 20", user_id)
        liked = [r["post_id"] for r in rows]
    recs = app.state.rec.recommend(user_id, liked_ids=liked, top_k=top_k)
    return {"user_id": user_id, "recommendations": recs}

@app.get("/users")
async def get_users():
    if app.state.use_db and db.pool:
        rows = await db.pool.fetch(
            "SELECT id,name,role,posts_count,likes_count,is_anomalous"
            " FROM users ORDER BY posts_count DESC")
        return {"users": [dict(r) for r in rows]}
    return {"users": [
        {"id": u["id"], "name": u["name"], "role": u["role"],
         "posts_count": u["posts"], "likes_count": u["likes"], "is_anomalous": False}
        for u in MEM_USERS.values()
    ]}

@app.get("/stats")
async def stats():
    if app.state.use_db and db.pool:
        row = await db.pool.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM posts WHERE status='approved') AS posts,
                (SELECT COUNT(*) FROM users)                          AS users,
                (SELECT COUNT(*) FROM likes)                          AS likes,
                (SELECT ROUND(COUNT(*) FILTER (WHERE status='approved')::numeric
                              / NULLIF(COUNT(*),0)*100,1) FROM posts) AS moderation_pct
        """)
        return dict(row)
    return {
        "posts": len(MEM_POSTS), "users": len(MEM_USERS),
        "likes": 0, "moderation_pct": 100.0,
    }
