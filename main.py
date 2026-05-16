"""
МедПлатформа API v3.1 — расширенная роль-модель + персональные рекомендации
- Роли: admin (управляет всем), doctor/patient (обычные пользователи)
- Реальные рекомендации на основе тегов постов и лайков
- 4 архитектуры для load-test (только админ)
"""
import os, time, hashlib, secrets, logging, asyncio
from datetime import datetime
from typing import List, Optional, Dict, Set
from collections import Counter, defaultdict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database as db
from services.moderation_service import ContentModerator
from services.recommendation_service import HybridRecommender, generate_demo_data
from services.architectures import get_architecture, list_architectures
from services.anomaly_service import (
    AccountAnomalyDetector,
    UserActivity,
    compute_features,
    generate_synthetic_activities,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("medplatforma")

app = FastAPI(title="МедПлатформа API", version="3.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ── In-memory хранилище ───────────────────────────────────────────────────────
_h = lambda p: hashlib.sha256(p.encode()).hexdigest()

MEM_USERS: Dict[str, dict] = {
    "user1": {"id":"user1","name":"Д-р Петров А.С.","role":"doctor",
              "ph":_h("doctor123"),"posts":0,"likes":0},
    "user2": {"id":"user2","name":"Мария Иванова","role":"patient",
              "ph":_h("patient123"),"posts":0,"likes":0},
    "user3": {"id":"user3","name":"Д-р Семёнова Е.В.","role":"doctor",
              "ph":_h("doctor789"),"posts":0,"likes":0},
    "user4": {"id":"user4","name":"Алексей Котов","role":"patient",
              "ph":_h("patient789"),"posts":0,"likes":0},
    "admin": {"id":"admin","name":"Администратор","role":"admin",
              "ph":_h("admin123"),"posts":0,"likes":0},
}
MEM_POSTS: List[dict] = [
    # --- Кардиология ---
    {"id":"p1","author_id":"user1","author":"Д-р Петров А.С.","role":"doctor",
     "title":"Реабилитация после инфаркта миокарда",
     "body":"После инфаркта важна кардиореабилитация. Первые 6 недель — ограниченная физическая активность.",
     "tags":["кардиология","реабилитация"],"likes_count":34,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},
    {"id":"p2","author_id":"user3","author":"Д-р Семёнова Е.В.","role":"doctor",
     "title":"Аритмия: когда стоит беспокоиться",
     "body":"Экстрасистолы у здоровых людей бывают, но при частоте более 6/мин и длительности — нужна консультация кардиолога.",
     "tags":["кардиология","аритмия"],"likes_count":21,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},
    {"id":"p3","author_id":"user4","author":"Алексей Котов","role":"patient",
     "title":"Высокое давление по утрам",
     "body":"Каждое утро АД 145/95. Принимаю эналаприл, но эффекта нет. Что делать?",
     "tags":["кардиология","гипертония"],"likes_count":8,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},

    # --- Диабет / эндокринология ---
    {"id":"p4","author_id":"user2","author":"Мария Иванова","role":"patient",
     "title":"Как принимать метформин при диабете 2 типа?",
     "body":"Врач назначил метформин 500мг. Когда лучше принимать — до или после еды? Есть ли побочные эффекты?",
     "tags":["диабет","препараты"],"likes_count":12,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},
    {"id":"p5","author_id":"user1","author":"Д-р Петров А.С.","role":"doctor",
     "title":"Целевой HbA1c при диабете 2 типа",
     "body":"Согласно клиническим рекомендациям, для большинства пациентов цель HbA1c менее 7%. У пожилых — допустимо 7.5-8%.",
     "tags":["диабет","эндокринология"],"likes_count":47,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},
    {"id":"p6","author_id":"user3","author":"Д-р Семёнова Е.В.","role":"doctor",
     "title":"Гипотиреоз: симптомы и лечение",
     "body":"Усталость, зябкость, прибавка веса — повод сдать ТТГ. При повышенном ТТГ назначается L-тироксин.",
     "tags":["эндокринология","гипотиреоз"],"likes_count":33,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},

    # --- Педиатрия ---
    {"id":"p7","author_id":"user3","author":"Д-р Семёнова Е.В.","role":"doctor",
     "title":"Профилактика ОРВИ у детей",
     "body":"Закаливание, проветривание, промывание носа физраствором и вакцинация от гриппа.",
     "tags":["педиатрия","ОРВИ","профилактика"],"likes_count":56,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},
    {"id":"p8","author_id":"user3","author":"Д-р Семёнова Е.В.","role":"doctor",
     "title":"Когда можно сбивать температуру у ребёнка",
     "body":"До 38.5°C при хорошей переносимости — не сбиваем. Парацетамол или ибупрофен по весу ребёнка.",
     "tags":["педиатрия","температура"],"likes_count":42,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},

    # --- Неврология ---
    {"id":"p9","author_id":"user1","author":"Д-р Петров А.С.","role":"doctor",
     "title":"Мигрень: триггеры и профилактика",
     "body":"Сон, питание, стресс — основные триггеры. При частых приступах назначается профилактическая терапия.",
     "tags":["неврология","мигрень"],"likes_count":29,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},
    {"id":"p10","author_id":"user4","author":"Алексей Котов","role":"patient",
     "title":"Боли в пояснице после тренировки",
     "body":"После становой тяги болит поясница 3 дня. Когда стоит делать МРТ?",
     "tags":["неврология","боль"],"likes_count":15,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},

    # --- Препараты ---
    {"id":"p11","author_id":"user2","author":"Мария Иванова","role":"patient",
     "title":"Сочетание препаратов: метформин и витамин B12",
     "body":"Прочитала что метформин снижает B12. Нужно ли пить добавки?",
     "tags":["диабет","препараты","витамины"],"likes_count":9,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},
    {"id":"p12","author_id":"user1","author":"Д-р Петров А.С.","role":"doctor",
     "title":"Антибиотики при пневмонии",
     "body":"Согласно клиническим протоколам — амоксициллин/клавуланат как первая линия. Курс 7-10 дней.",
     "tags":["препараты","антибиотики","пульмонология"],"likes_count":38,"status":"approved",
     "created_at": datetime.utcnow().isoformat()},
]
# user_id → set(post_id) — отслеживание лайков для in-memory
MEM_LIKES: Dict[str, Set[str]] = defaultdict(set)
_TOKENS: dict = {}
_NEXT_USER_ID = [10]   # счётчик id для новых пользователей
_NEXT_POST_ID = [100]  # счётчик id для новых постов

# ── Жизненный цикл ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    app.state.mod = ContentModerator()
    app.state.mod.train()
    items, ints = generate_demo_data()
    app.state.rec = HybridRecommender(alpha=0.4)
    app.state.rec.fit(items, ints)
    logger.info("[INIT] ML-сервисы готовы")
    app.state.use_db = await db.init_pool()
    logger.info("[INIT] БД: %s",
                "postgresql" if app.state.use_db else "in-memory (БД недоступна)")

@app.on_event("shutdown")
async def shutdown():
    await db.close_pool()

# ── Статика ───────────────────────────────────────────────────────────────────
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app", include_in_schema=False)
async def app_index():
    idx = "static/index.html"
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

class PostUpdateReq(BaseModel):
    title: Optional[str] = Field(None, min_length=5, max_length=500)
    body:  Optional[str] = Field(None, min_length=10)
    tags:  Optional[List[str]] = None
    status: Optional[str] = None

class ModReq(BaseModel):
    text: str

class UserCreateReq(BaseModel):
    username: str = Field(..., min_length=2, max_length=30)
    password: str = Field(..., min_length=4, max_length=100)
    name: str = Field(..., min_length=2, max_length=100)
    role: str = Field(..., pattern="^(doctor|patient|admin)$")

class UserUpdateReq(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = Field(None, pattern="^(doctor|patient|admin)$")
    password: Optional[str] = None

class AnomalyDetectReq(BaseModel):
    synthetic: bool = False
    n_synthetic_legit: int = Field(45, ge=0, le=200)
    n_synthetic_anomalies: int = Field(5, ge=0, le=50)
    update_db: bool = True

# ── Авторизация и роли ────────────────────────────────────────────────────────
def get_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not creds or creds.credentials not in _TOKENS:
        raise HTTPException(401, "Не авторизован")
    return _TOKENS[creds.credentials]

def get_user_optional(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not creds or creds.credentials not in _TOKENS:
        return None
    return _TOKENS[creds.credentials]

def require_admin(user=Depends(get_user)):
    if user.get("role") != "admin":
        raise HTTPException(403, "Доступ только для администратора")
    return user

# ── Базовые эндпоинты ─────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service":   "МедПлатформа API",
        "version":   "3.1.0",
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

# ── Посты ─────────────────────────────────────────────────────────────────────
@app.get("/posts")
async def get_posts(limit: int = 20, offset: int = 0,
                    user=Depends(get_user_optional)):
    user_id = user["id"] if user else None

    if app.state.use_db and db.pool:
        rows = await db.pool.fetch("""
            SELECT p.id, p.author_id, p.title, p.body, p.status, p.likes_count,
                   u.name AS author, u.role,
                   array_agg(t.name) FILTER (WHERE t.name IS NOT NULL) AS tags,
                   EXISTS(SELECT 1 FROM likes l WHERE l.post_id=p.id AND l.user_id=$3) AS liked_by_me
            FROM posts p
            JOIN users u ON u.id = p.author_id
            LEFT JOIN post_tags pt ON pt.post_id = p.id
            LEFT JOIN tags t ON t.id = pt.tag_id
            WHERE p.status = 'approved'
            GROUP BY p.id, u.id
            ORDER BY p.created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset, user_id or "")
        total = await db.pool.fetchval(
            "SELECT COUNT(*) FROM posts WHERE status='approved'")
        return {"posts": [dict(r) for r in rows], "total": total}

    # in-memory: с фильтром approved + флаг liked_by_me
    filtered = [p for p in MEM_POSTS if p.get("status") == "approved"]
    sl = filtered[offset:offset+limit]
    out = []
    for p in sl:
        d = dict(p)
        d["liked_by_me"] = bool(user_id) and p["id"] in MEM_LIKES.get(user_id, set())
        out.append(d)
    return {"posts": out, "total": len(filtered)}

@app.post("/posts", status_code=201)
async def create_post(req: PostReq, user=Depends(get_user)):
    full_text = f"{req.title}\n{req.body}"
    mod = app.state.mod.moderate(full_text)
    status = "approved" if mod.label == "approved" else mod.label

    pid = f"p{_NEXT_POST_ID[0]}"
    _NEXT_POST_ID[0] += 1

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
            "id": pid, "author_id": user["id"], "author": user["name"],
            "role": user["role"], "title": req.title, "body": req.body,
            "tags": [t.strip().lower() for t in req.tags if t.strip()][:10],
            "likes_count": 0, "status": status,
            "created_at": datetime.utcnow().isoformat(),
        })
        if user["id"] in MEM_USERS:
            MEM_USERS[user["id"]]["posts"] += 1

    return {
        "id": pid, "status": status,
        "moderation": {"label": mod.label, "confidence": mod.confidence, "level": mod.level},
    }

@app.put("/posts/{post_id}")
async def update_post(post_id: str, req: PostUpdateReq, user=Depends(get_user)):
    """Редактирование поста — только автор или админ."""
    if app.state.use_db and db.pool:
        post = await db.pool.fetchrow("SELECT author_id FROM posts WHERE id=$1", post_id)
        if not post:
            raise HTTPException(404, "Пост не найден")
        if post["author_id"] != user["id"] and user["role"] != "admin":
            raise HTTPException(403, "Можно редактировать только свои посты")
        fields, vals = [], []
        if req.title is not None:
            fields.append(f"title=${len(vals)+1}"); vals.append(req.title)
        if req.body is not None:
            fields.append(f"body=${len(vals)+1}"); vals.append(req.body)
        if req.status is not None and user["role"] == "admin":
            fields.append(f"status=${len(vals)+1}"); vals.append(req.status)
        if not fields:
            return {"updated": False}
        vals.append(post_id)
        await db.pool.execute(
            f"UPDATE posts SET {', '.join(fields)} WHERE id=${len(vals)}", *vals
        )
        return {"updated": True}

    # in-memory
    for p in MEM_POSTS:
        if p["id"] == post_id:
            if p["author_id"] != user["id"] and user["role"] != "admin":
                raise HTTPException(403, "Можно редактировать только свои посты")
            if req.title is not None: p["title"] = req.title
            if req.body  is not None: p["body"]  = req.body
            if req.tags  is not None:
                p["tags"] = [t.strip().lower() for t in req.tags if t.strip()][:10]
            if req.status is not None and user["role"] == "admin":
                p["status"] = req.status
            return {"updated": True}
    raise HTTPException(404, "Пост не найден")

@app.delete("/posts/{post_id}")
async def delete_post(post_id: str, user=Depends(get_user)):
    """Удалить пост — только автор или админ."""
    if app.state.use_db and db.pool:
        post = await db.pool.fetchrow("SELECT author_id FROM posts WHERE id=$1", post_id)
        if not post:
            raise HTTPException(404, "Пост не найден")
        if post["author_id"] != user["id"] and user["role"] != "admin":
            raise HTTPException(403, "Можно удалять только свои посты")
        await db.pool.execute("DELETE FROM posts WHERE id=$1", post_id)
        return {"deleted": True}

    # in-memory
    for i, p in enumerate(MEM_POSTS):
        if p["id"] == post_id:
            if p["author_id"] != user["id"] and user["role"] != "admin":
                raise HTTPException(403, "Можно удалять только свои посты")
            MEM_POSTS.pop(i)
            # удалить связанные лайки
            for likes_set in MEM_LIKES.values():
                likes_set.discard(post_id)
            return {"deleted": True}
    raise HTTPException(404, "Пост не найден")

@app.post("/posts/{post_id}/like")
async def like_post(post_id: str, user=Depends(get_user)):
    """Toggle лайк (поставил/убрал)."""
    if app.state.use_db and db.pool:
        exists = await db.pool.fetchrow(
            "SELECT 1 FROM likes WHERE user_id=$1 AND post_id=$2",
            user["id"], post_id,
        )
        if exists:
            await db.pool.execute(
                "DELETE FROM likes WHERE user_id=$1 AND post_id=$2",
                user["id"], post_id)
            await db.pool.execute(
                "UPDATE posts SET likes_count=likes_count-1 WHERE id=$1", post_id)
            return {"liked": False}
        else:
            await db.pool.execute(
                "INSERT INTO likes(user_id,post_id) VALUES($1,$2) ON CONFLICT DO NOTHING",
                user["id"], post_id)
            await db.pool.execute(
                "UPDATE posts SET likes_count=likes_count+1 WHERE id=$1", post_id)
            return {"liked": True}

    # in-memory
    target = next((p for p in MEM_POSTS if p["id"] == post_id), None)
    if not target:
        raise HTTPException(404, "Пост не найден")
    user_likes = MEM_LIKES[user["id"]]
    if post_id in user_likes:
        user_likes.discard(post_id)
        target["likes_count"] = max(0, target["likes_count"] - 1)
        return {"liked": False, "likes_count": target["likes_count"]}
    else:
        user_likes.add(post_id)
        target["likes_count"] = target["likes_count"] + 1
        return {"liked": True, "likes_count": target["likes_count"]}

# ── Модерация ─────────────────────────────────────────────────────────────────
@app.post("/moderation/check")
async def moderation_check(req: ModReq, _=Depends(require_admin)):
    mod = app.state.mod.moderate(req.text)
    return {"label": mod.label, "confidence": mod.confidence, "level": mod.level}

# ── Рекомендации (РЕАЛЬНЫЕ, на основе тегов и лайков) ─────────────────────────
@app.get("/recommendations/{user_id}")
async def recommendations(user_id: str, top_k: int = 5, user=Depends(get_user_optional)):
    """
    Персональные рекомендации на основе:
    1) Тегов постов которые написал сам пользователь
    2) Тегов постов которые он лайкнул
    Возвращаем посты других авторов с пересекающимися тегами.
    """
    if app.state.use_db and db.pool:
        # Соберём интересы — теги из своих постов и лайкнутых постов
        interest_tags = await db.pool.fetch("""
            SELECT DISTINCT t.name
            FROM tags t
            JOIN post_tags pt ON pt.tag_id = t.id
            JOIN posts p ON p.id = pt.post_id
            WHERE p.author_id = $1
               OR p.id IN (SELECT post_id FROM likes WHERE user_id = $1)
        """, user_id)
        tag_set = [r["name"] for r in interest_tags]
        if not tag_set:
            # cold start — самые популярные посты других авторов
            rows = await db.pool.fetch("""
                SELECT p.id, p.title, u.name AS author, p.likes_count,
                       array_agg(t.name) FILTER (WHERE t.name IS NOT NULL) AS tags
                FROM posts p
                JOIN users u ON u.id = p.author_id
                LEFT JOIN post_tags pt ON pt.post_id = p.id
                LEFT JOIN tags t ON t.id = pt.tag_id
                WHERE p.status='approved' AND p.author_id != $1
                GROUP BY p.id, u.id
                ORDER BY p.likes_count DESC
                LIMIT $2
            """, user_id, top_k)
            return {
                "user_id": user_id,
                "based_on": [],
                "recommendations": [
                    {"id": r["id"], "title": r["title"], "author": r["author"],
                     "likes_count": r["likes_count"], "tags": r["tags"] or [],
                     "score": 0.0, "reason": "Популярное"} for r in rows
                ],
            }

        # Подбор по пересечению тегов
        rows = await db.pool.fetch("""
            SELECT p.id, p.title, u.name AS author, p.likes_count,
                   array_agg(t.name) FILTER (WHERE t.name IS NOT NULL) AS tags,
                   COUNT(DISTINCT t.name) FILTER (WHERE t.name = ANY($2::text[])) AS overlap
            FROM posts p
            JOIN users u ON u.id = p.author_id
            LEFT JOIN post_tags pt ON pt.post_id = p.id
            LEFT JOIN tags t ON t.id = pt.tag_id
            WHERE p.status='approved'
              AND p.author_id != $1
              AND p.id NOT IN (SELECT post_id FROM likes WHERE user_id = $1)
            GROUP BY p.id, u.id
            HAVING COUNT(DISTINCT t.name) FILTER (WHERE t.name = ANY($2::text[])) > 0
            ORDER BY overlap DESC, p.likes_count DESC
            LIMIT $3
        """, user_id, tag_set, top_k)
        return {
            "user_id": user_id,
            "based_on": tag_set,
            "recommendations": [
                {"id": r["id"], "title": r["title"], "author": r["author"],
                 "likes_count": r["likes_count"], "tags": r["tags"] or [],
                 "score": float(r["overlap"]) / max(1, len(tag_set)),
                 "reason": f"Совпадают теги: {r['overlap']}/{len(tag_set)}"} for r in rows
            ],
        }

    # in-memory режим
    own_posts = [p for p in MEM_POSTS if p.get("author_id") == user_id]
    liked_post_ids = MEM_LIKES.get(user_id, set())
    liked_posts = [p for p in MEM_POSTS if p["id"] in liked_post_ids]

    # Собираем интересы из собственных постов + лайкнутых
    interest_counter = Counter()
    for p in own_posts + liked_posts:
        for tag in p.get("tags", []):
            interest_counter[tag] += 1

    if not interest_counter:
        # Cold start: топ постов других авторов по лайкам
        candidates = sorted(
            [p for p in MEM_POSTS
             if p.get("status") == "approved" and p.get("author_id") != user_id],
            key=lambda p: -p.get("likes_count", 0),
        )[:top_k]
        return {
            "user_id": user_id, "based_on": [],
            "recommendations": [
                {"id": p["id"], "title": p["title"], "author": p["author"],
                 "likes_count": p["likes_count"], "tags": p.get("tags", []),
                 "score": 0.0, "reason": "Популярное у других"}
                for p in candidates
            ],
        }

    # Скоринг по пересечению тегов
    interest_tags = list(interest_counter.keys())
    scored = []
    for p in MEM_POSTS:
        if p.get("status") != "approved": continue
        if p.get("author_id") == user_id: continue
        if p["id"] in liked_post_ids: continue
        ptags = set(p.get("tags", []))
        overlap = ptags.intersection(interest_tags)
        if overlap:
            score = sum(interest_counter[t] for t in overlap) / sum(interest_counter.values())
            scored.append((score, len(overlap), p))

    scored.sort(key=lambda x: (-x[0], -x[1], -x[2].get("likes_count", 0)))
    top = scored[:top_k]

    # Fallback: если по тегам ничего не нашлось — показываем популярные посты других авторов
    if not top:
        candidates = sorted(
            [p for p in MEM_POSTS
             if p.get("status") == "approved"
             and p.get("author_id") != user_id
             and p["id"] not in liked_post_ids],
            key=lambda p: -p.get("likes_count", 0),
        )[:top_k]
        return {
            "user_id": user_id, "based_on": interest_tags,
            "recommendations": [
                {"id": p["id"], "title": p["title"], "author": p["author"],
                 "likes_count": p["likes_count"], "tags": p.get("tags", []),
                 "score": 0.0, "reason": "Другие популярные публикации"}
                for p in candidates
            ],
        }

    return {
        "user_id": user_id,
        "based_on": interest_tags,
        "recommendations": [
            {"id": p["id"], "title": p["title"], "author": p["author"],
             "likes_count": p["likes_count"], "tags": p.get("tags", []),
             "score": round(score, 3),
             "reason": f"Совпадают теги: {n_match}/{len(interest_tags)}"}
            for score, n_match, p in top
        ],
    }

# ── Пользователи ──────────────────────────────────────────────────────────────
@app.get("/users")
async def get_users(_=Depends(require_admin)):
    if app.state.use_db and db.pool:
        rows = await db.pool.fetch(
            "SELECT id,name,role,posts_count,likes_count,is_anomalous"
            " FROM users ORDER BY posts_count DESC")
        return {"users": [dict(r) for r in rows]}
    # in-memory: пересчитываем посты/лайки на лету
    users_out = []
    for u in MEM_USERS.values():
        actual_posts = sum(1 for p in MEM_POSTS if p.get("author_id") == u["id"])
        likes_received = sum(p.get("likes_count", 0) for p in MEM_POSTS if p.get("author_id") == u["id"])
        users_out.append({
            "id": u["id"], "name": u["name"], "role": u["role"],
            "posts_count": actual_posts, "likes_count": likes_received,
            "is_anomalous": bool(u.get("is_anomalous", False)),
        })
    return {"users": users_out}

@app.post("/users", status_code=201)
async def create_user(req: UserCreateReq, _=Depends(require_admin)):
    """Только админ."""
    if app.state.use_db and db.pool:
        exists = await db.pool.fetchrow("SELECT 1 FROM users WHERE id=$1", req.username)
        if exists:
            raise HTTPException(400, "Пользователь с таким username уже существует")
        await db.pool.execute(
            "INSERT INTO users(id,name,role,password_hash) VALUES($1,$2,$3,$4)",
            req.username, req.name, req.role, _h(req.password),
        )
        return {"id": req.username, "name": req.name, "role": req.role}

    if req.username in MEM_USERS:
        raise HTTPException(400, "Пользователь с таким username уже существует")
    MEM_USERS[req.username] = {
        "id": req.username, "name": req.name, "role": req.role,
        "ph": _h(req.password), "posts": 0, "likes": 0,
    }
    return {"id": req.username, "name": req.name, "role": req.role}

@app.put("/users/{user_id}")
async def update_user(user_id: str, req: UserUpdateReq, _=Depends(require_admin)):
    """Только админ."""
    if app.state.use_db and db.pool:
        fields, vals = [], []
        if req.name is not None:
            fields.append(f"name=${len(vals)+1}"); vals.append(req.name)
        if req.role is not None:
            fields.append(f"role=${len(vals)+1}"); vals.append(req.role)
        if req.password is not None:
            fields.append(f"password_hash=${len(vals)+1}"); vals.append(_h(req.password))
        if not fields:
            return {"updated": False}
        vals.append(user_id)
        result = await db.pool.execute(
            f"UPDATE users SET {', '.join(fields)} WHERE id=${len(vals)}", *vals)
        return {"updated": result.endswith("1")}

    if user_id not in MEM_USERS:
        raise HTTPException(404, "Пользователь не найден")
    u = MEM_USERS[user_id]
    if req.name is not None: u["name"] = req.name
    if req.role is not None: u["role"] = req.role
    if req.password is not None: u["ph"] = _h(req.password)
    return {"updated": True}

@app.delete("/users/{user_id}")
async def delete_user(user_id: str, admin=Depends(require_admin)):
    """Только админ. Запрещено удалять самого себя."""
    if user_id == admin["id"]:
        raise HTTPException(400, "Нельзя удалить самого себя")
    if app.state.use_db and db.pool:
        result = await db.pool.execute("DELETE FROM users WHERE id=$1", user_id)
        return {"deleted": result.endswith("1")}

    if user_id not in MEM_USERS:
        raise HTTPException(404, "Пользователь не найден")
    del MEM_USERS[user_id]
    # Удалить связанные посты и лайки
    MEM_POSTS[:] = [p for p in MEM_POSTS if p.get("author_id") != user_id]
    MEM_LIKES.pop(user_id, None)
    for likes_set in MEM_LIKES.values():
        # лайки от удалённого юзера могли остаться у других — в данном случае не актуально
        pass
    return {"deleted": True}

# ── Обнаружение аномальных аккаунтов (SVD) ────────────────────────────────────
async def _gather_user_activities() -> List[UserActivity]:
    """
    Собирает данные всех аккаунтов системы для вычисления поведенческих
    признаков. Работает и с PostgreSQL, и с in-memory хранилищем.
    """
    activities: List[UserActivity] = []

    if app.state.use_db and db.pool:
        users_rows = await db.pool.fetch(
            "SELECT id, name, role, created_at FROM users ORDER BY id")
        for u in users_rows:
            posts_rows = await db.pool.fetch(
                "SELECT title, body, created_at, likes_count "
                "FROM posts WHERE author_id=$1", u["id"])
            likes_given_row = await db.pool.fetchrow(
                "SELECT COUNT(*) AS c FROM likes WHERE user_id=$1", u["id"])
            likes_given = int(likes_given_row["c"]) if likes_given_row else 0
            activities.append(UserActivity(
                user_id=u["id"], name=u["name"], role=u["role"],
                created_at=u["created_at"],
                posts=[dict(p) for p in posts_rows],
                likes_given=likes_given,
            ))
    else:
        for uid, u in MEM_USERS.items():
            user_posts = [p for p in MEM_POSTS if p.get("author_id") == uid]
            likes_given = len(MEM_LIKES.get(uid, set())) if isinstance(MEM_LIKES.get(uid), set) else 0
            activities.append(UserActivity(
                user_id=uid, name=u["name"], role=u["role"],
                posts=user_posts, likes_given=likes_given,
            ))
    return activities


@app.post("/users/detect-anomalies")
async def detect_anomalies(req: AnomalyDetectReq, _=Depends(require_admin)):
    """
    Запускает SVD-детектор аномальных аккаунтов.

    Реализует метод из статьи «Метод обнаружения аномальных аккаунтов
    в медицинских социальных платформах на основе сингулярного разложения
    матрицы поведения» (Аль-Раве М.И.Т., Макаров А.В.).

    Параметры:
        synthetic — добавить к реальным аккаунтам синтетические для
                    демонстрации работы алгоритма (если в системе мало данных).
        n_synthetic_legit — число «нормальных» синтетических аккаунтов.
        n_synthetic_anomalies — число «аномальных» синтетических аккаунтов.
        update_db — обновить поле is_anomalous в БД по результатам анализа
                    (только для реальных аккаунтов, синтетика игнорируется).
    """
    activities = await _gather_user_activities()

    if req.synthetic:
        synth = generate_synthetic_activities(
            n_legit=req.n_synthetic_legit,
            n_anomalies=req.n_synthetic_anomalies,
        )
        activities = activities + synth

    detector = AccountAnomalyDetector()
    result = detector.run(activities)

    # Обновление флага is_anomalous в БД (только для реальных аккаунтов)
    if req.update_db:
        real_results = [u for u in result["users"] if not u["is_synthetic"]]
        if app.state.use_db and db.pool:
            for u in real_results:
                await db.pool.execute(
                    "UPDATE users SET is_anomalous=$1 WHERE id=$2",
                    bool(u["is_anomalous"]), u["id"],
                )
        else:
            for u in real_results:
                if u["id"] in MEM_USERS:
                    MEM_USERS[u["id"]]["is_anomalous"] = bool(u["is_anomalous"])

    return result


@app.get("/users/{user_id}/behavioral-features")
async def get_user_features(user_id: str, _=Depends(require_admin)):
    """
    Возвращает 9 поведенческих признаков одного аккаунта без запуска SVD.
    Полезно для углублённого анализа конкретного пользователя.
    """
    activities = await _gather_user_activities()
    activity = next((a for a in activities if a.user_id == user_id), None)
    if activity is None:
        raise HTTPException(404, "Пользователь не найден")
    return {
        "id":          activity.user_id,
        "name":        activity.name,
        "role":        activity.role,
        "n_posts":     len(activity.posts),
        "likes_given": activity.likes_given,
        "features":    compute_features(activity),
    }


# ── Статистика ────────────────────────────────────────────────────────────────
@app.get("/stats")
async def stats(_=Depends(require_admin)):
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
    total_likes = sum(len(s) for s in MEM_LIKES.values())
    return {
        "posts": len([p for p in MEM_POSTS if p.get("status") == "approved"]),
        "users": len(MEM_USERS), "likes": total_likes, "moderation_pct": 100.0,
    }

# ── Архитектуры (только админ) ────────────────────────────────────────────────
@app.get("/arch/list")
async def arch_list():
    return {"architectures": list_architectures()}

@app.get("/arch/info")
async def arch_info():
    from services.architectures import (
        MonolithApp, ThreeTierApp, MicroservicesApp, ServerlessApp
    )
    return {
        "monolith":      {"thread_pool": MonolithApp.THREAD_POOL,
                          "bottleneck": "общий lock БД"},
        "three_tier":    {"frontend_pool": ThreeTierApp.FRONTEND_POOL,
                          "backend_pool":  ThreeTierApp.BACKEND_POOL,
                          "db_pool":       ThreeTierApp.DB_POOL},
        "microservices": {"gateway_pool": MicroservicesApp.GATEWAY_BASE},
        "serverless":    {"cold_start_ms": ServerlessApp.COLD_START_LATENCY*1000},
    }

@app.get("/arch/{name}/loadtest")
async def arch_loadtest(name: str, n: int = 50, _=Depends(require_admin)):
    """Реальный concurrent load-test. Только админ."""
    arch = get_architecture(name)
    if not arch:
        raise HTTPException(404, f"Архитектура '{name}' не найдена")
    n = max(1, min(n, 2000))

    async def one_request():
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(arch.handle_request(), timeout=30.0)
            return (time.perf_counter() - t0), None
        except asyncio.TimeoutError:
            return 30.0, "timeout"
        except Exception as e:
            return time.perf_counter() - t0, str(e)[:80]

    t_start = time.perf_counter()
    results = await asyncio.gather(*[one_request() for _ in range(n)])
    total = time.perf_counter() - t_start

    times_ok = sorted([r[0] * 1000 for r in results if r[1] is None])
    errors = [r[1] for r in results if r[1] is not None]

    if not times_ok:
        return {"arch": name, "n": n, "rps": 0, "rt_avg": 0, "rt_p50": 0,
                "rt_p95": 0, "rt_p99": 0, "rt_min": 0, "rt_max": 0,
                "error_rate": 100.0, "duration_ms": round(total * 1000, 1)}

    def pct(p):
        i = int(len(times_ok) * p)
        return times_ok[min(i, len(times_ok)-1)]

    return {
        "arch": name, "n": n,
        "rps": round(n / total, 1) if total else 0,
        "rt_avg": round(sum(times_ok) / len(times_ok), 1),
        "rt_min": round(times_ok[0], 1),
        "rt_p50": round(pct(0.50), 1),
        "rt_p95": round(pct(0.95), 1),
        "rt_p99": round(pct(0.99), 1),
        "rt_max": round(times_ok[-1], 1),
        "error_rate": round(len(errors) / n * 100, 2),
        "duration_ms": round(total * 1000, 1),
    }
