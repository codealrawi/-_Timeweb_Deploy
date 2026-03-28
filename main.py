"""
МедПлатформа — FastAPI + PostgreSQL
Автор: Аль-Раве Мустафа Исам Табит, РГСУ, спец. 2.3.5
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import time, hashlib, secrets, os
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from database import (
    get_db, init_db, engine,
    User, Post, PostTag, Comment, Like, ModerationLog, Recommendation
)
from services.moderation_service import ContentModerator
from services.recommendation_service import HybridRecommender, generate_demo_data

# ─── Приложение ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="МедПлатформа API",
    description="Q&A-платформа для пациентов и врачей · PostgreSQL edition",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─── ML-сервисы ──────────────────────────────────────────────────────────────
moderator = ContentModerator()
moderator.train()

items, interactions = generate_demo_data()
recommender = HybridRecommender(alpha=0.4)
recommender.fit(items, interactions)

TOKENS: dict[str, str] = {}   # token → user_id (in-memory, заменить на Redis в проде)

# ─── Статический фронтенд ────────────────────────────────────────────────────
_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

@app.get("/app", include_in_schema=False)
async def frontend():
    f = os.path.join(_static, "index.html")
    return FileResponse(f) if os.path.exists(f) else {"error": "frontend not found"}

# ─── Lifecycle ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await init_db()
    print("✓ БД подключена и инициализирована")

@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()

# ─── Схемы ───────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class PostCreate(BaseModel):
    title: str
    body: str
    tags: List[str] = []

class ModerateRequest(BaseModel):
    text: str

class CommentCreate(BaseModel):
    body: str

# ─── Авторизация ─────────────────────────────────────────────────────────────
def get_token(creds: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    return creds.credentials if creds else None

async def get_current_user(
    token: str = Depends(get_token),
    db: AsyncSession = Depends(get_db)
) -> User:
    if not token or token not in TOKENS:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    user = await db.get(User, TOKENS[token])
    if not user:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    return user

# ─── ЭНДПОИНТЫ ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "МедПлатформа API", "version": "2.0.0", "db": "PostgreSQL", "docs": "/docs"}

@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(select(func.now()))
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "moderator": True,
        "recommender": True,
        "timestamp": datetime.utcnow().isoformat(),
    }

# ── Аутентификация ────────────────────────────────────────────────────────────
@app.post("/auth/login")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == req.username))
    user = result.scalar_one_or_none()
    if not user or user.password_hash != hashlib.sha256(req.password.encode()).hexdigest():
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    token = secrets.token_hex(32)
    TOKENS[token] = user.id
    return {"token": token, "user_id": user.id, "name": user.name, "role": user.role}

@app.post("/auth/logout")
async def logout(token: str = Depends(get_token)):
    if token in TOKENS:
        del TOKENS[token]
    return {"status": "ok"}

# ── Пользователи ─────────────────────────────────────────────────────────────
@app.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return {"users": [
        {"id": u.id, "name": u.name, "role": u.role,
         "posts": u.posts_count, "likes": u.likes_count,
         "anomalous": u.is_anomalous}
        for u in users
    ]}

@app.get("/users/{user_id}")
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return {"id": user.id, "name": user.name, "role": user.role,
            "posts": user.posts_count, "likes": user.likes_count}

# ── Публикации ────────────────────────────────────────────────────────────────
@app.get("/posts")
async def get_posts(
    limit: int = 20, offset: int = 0,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    q = select(Post).offset(offset).limit(limit).order_by(Post.created_at.desc())
    if status:
        q = q.where(Post.status == status)
    result = await db.execute(q)
    posts = result.scalars().all()

    total = await db.scalar(select(func.count(Post.id)))
    return {
        "posts": [_post_to_dict(p) for p in posts],
        "total": total,
        "limit": limit,
        "offset": offset,
    }

@app.get("/posts/{post_id}")
async def get_post(post_id: str, db: AsyncSession = Depends(get_db)):
    post = await db.get(Post, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Публикация не найдена")
    return _post_to_dict(post)

@app.post("/posts", status_code=201)
async def create_post(
    req: PostCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Модерация
    mod = moderator.moderate(f"{req.title} {req.body}")

    import uuid
    post_id = f"p_{uuid.uuid4().hex[:8]}"
    post = Post(
        id=post_id,
        author_id=current_user.id,
        title=req.title,
        body=req.body,
        status=mod.label,
        mod_conf=mod.confidence,
        mod_level=mod.level,
    )
    db.add(post)

    for tag in req.tags:
        db.add(PostTag(post_id=post_id, tag=tag.strip()))

    # Лог модерации
    db.add(ModerationLog(
        post_id=post_id,
        text_hash=hashlib.sha256(f"{req.title} {req.body}".encode()).hexdigest()[:64],
        label=mod.label, confidence=mod.confidence, level=mod.level,
        reasons=", ".join(getattr(mod, "reasons", []))
    ))

    current_user.posts_count += 1
    await db.commit()
    await db.refresh(post)
    return {"post": _post_to_dict(post), "moderation": {"label": mod.label, "confidence": mod.confidence}}

@app.post("/posts/{post_id}/like")
async def like_post(
    post_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    post = await db.get(Post, post_id)
    if not post:
        raise HTTPException(status_code=404)
    existing = await db.execute(
        select(Like).where(Like.user_id == current_user.id, Like.post_id == post_id)
    )
    if existing.scalar():
        await db.execute(delete(Like).where(Like.user_id == current_user.id, Like.post_id == post_id))
        post.likes = max(0, post.likes - 1)
        liked = False
    else:
        db.add(Like(user_id=current_user.id, post_id=post_id))
        post.likes += 1
        liked = True
    await db.commit()
    return {"liked": liked, "likes": post.likes}

# ── Комментарии ───────────────────────────────────────────────────────────────
@app.post("/posts/{post_id}/comments", status_code=201)
async def add_comment(
    post_id: str,
    req: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    post = await db.get(Post, post_id)
    if not post:
        raise HTTPException(status_code=404)
    comment = Comment(post_id=post_id, author_id=current_user.id, body=req.body)
    db.add(comment)
    await db.commit()
    return {"id": comment.id, "body": comment.body, "author": current_user.name}

@app.get("/posts/{post_id}/comments")
async def get_comments(post_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Comment).where(Comment.post_id == post_id).order_by(Comment.created_at)
    )
    comments = result.scalars().all()
    return {"comments": [
        {"id": c.id, "body": c.body, "author_id": c.author_id, "created_at": c.created_at.isoformat()}
        for c in comments
    ]}

# ── Модерация ─────────────────────────────────────────────────────────────────
@app.post("/moderation/check")
async def check_moderation(req: ModerateRequest):
    result = moderator.moderate(req.text)
    return {
        "text":       req.text[:100],
        "label":      result.label,
        "confidence": result.confidence,
        "level":      result.level,
    }

@app.get("/moderation/stats")
async def moderation_stats(db: AsyncSession = Depends(get_db)):
    total   = await db.scalar(select(func.count(ModerationLog.id)))
    blocked = await db.scalar(select(func.count(ModerationLog.id)).where(ModerationLog.label == "blocked"))
    return {
        "total": total,
        "blocked": blocked,
        "approved": (total or 0) - (blocked or 0),
        "metrics": moderator.evaluate(),
    }

# ── Рекомендации ──────────────────────────────────────────────────────────────
@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    top_k: int = 5,
    db: AsyncSession = Depends(get_db)
):
    # Получаем ID постов которые пользователь уже видел (liked)
    result = await db.execute(select(Like.post_id).where(Like.user_id == user_id))
    liked_ids = [r[0] for r in result.all()]

    recs = recommender.recommend(user_id, liked_ids=liked_ids, top_k=top_k)

    # Обновляем таблицу рекомендаций
    for rec in recs:
        existing = await db.execute(
            select(Recommendation).where(
                Recommendation.user_id == user_id,
                Recommendation.post_id == rec["item_id"]
            )
        )
        if not existing.scalar():
            db.add(Recommendation(
                user_id=user_id, post_id=rec["item_id"],
                score=rec["score"],
                cbf_score=rec.get("cbf_score", 0),
                svd_score=rec.get("svd_score", 0),
            ))
    try:
        await db.commit()
    except Exception:
        await db.rollback()

    return {"user_id": user_id, "recommendations": recs, "algorithm": "CBF+SVD hybrid α=0.4"}

# ── Статистика ────────────────────────────────────────────────────────────────
@app.get("/stats")
async def platform_stats(db: AsyncSession = Depends(get_db)):
    posts_total = await db.scalar(select(func.count(Post.id)))
    users_total = await db.scalar(select(func.count(User.id)))
    approved    = await db.scalar(select(func.count(Post.id)).where(Post.status == "approved"))
    blocked     = await db.scalar(select(func.count(Post.id)).where(Post.status == "blocked"))
    comments    = await db.scalar(select(func.count(Comment.id)))
    return {
        "posts":     posts_total,
        "users":     users_total,
        "approved":  approved,
        "blocked":   blocked,
        "comments":  comments,
        "mod_rate":  round((approved or 0) / max(posts_total or 1, 1) * 100, 1),
    }

# ─── Хелпер ──────────────────────────────────────────────────────────────────
def _post_to_dict(p: Post) -> dict:
    return {
        "id":         p.id,
        "author_id":  p.author_id,
        "author":     p.author.name if p.author else p.author_id,
        "role":       p.author.role if p.author else "unknown",
        "title":      p.title,
        "body":       p.body,
        "status":     p.status,
        "mod_conf":   p.mod_conf,
        "likes":      p.likes,
        "tags":       [t.tag for t in p.tags],
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }
