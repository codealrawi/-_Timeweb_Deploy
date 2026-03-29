"""
database.py — SQLAlchemy async модели для МедПлатформа
PostgreSQL через asyncpg
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Integer, Boolean, DateTime, ForeignKey, Float, func
from typing import Optional, List
from datetime import datetime
import os

# ─── URL из переменной окружения ─────────────────────────────────────────────
# Formат: postgresql+asyncpg://gen_user:ПАРОЛЬ@HOST:5432/default_db?ssl=require
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://gen_user:ПАРОЛЬ@2becf44e38c66a65b2f1dfc5.twc1.net:5432/default_db?ssl=require"
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ─── Базовый класс ────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

# ─── МОДЕЛИ ──────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id:            Mapped[str]           = mapped_column(String(50), primary_key=True)
    name:          Mapped[str]           = mapped_column(String(200))
    role:          Mapped[str]           = mapped_column(String(20))  # doctor / patient / admin
    password_hash: Mapped[str]           = mapped_column(String(200))
    posts_count:   Mapped[int]           = mapped_column(Integer, default=0)
    likes_count:   Mapped[int]           = mapped_column(Integer, default=0)
    is_anomalous:  Mapped[bool]          = mapped_column(Boolean, default=False)
    created_at:    Mapped[datetime]      = mapped_column(DateTime, server_default=func.now())

    posts:    "List[Post]"    = relationship("Post", back_populates="author", lazy="selectin")
    comments: "List[Comment]" = relationship("Comment", back_populates="author", lazy="selectin")


class Post(Base):
    __tablename__ = "posts"
    id:         Mapped[str]       = mapped_column(String(50), primary_key=True)
    author_id:  Mapped[str]       = mapped_column(String(50), ForeignKey("users.id"))
    title:      Mapped[str]       = mapped_column(String(500))
    body:       Mapped[str]       = mapped_column(Text)
    status:     Mapped[str]       = mapped_column(String(30), default="pending")  # approved/blocked/pending
    mod_conf:   Mapped[float]     = mapped_column(Float, default=0.0)
    mod_level:  Mapped[int]       = mapped_column(Integer, default=1)
    likes:      Mapped[int]       = mapped_column(Integer, default=0)
    created_at: Mapped[datetime]  = mapped_column(DateTime, server_default=func.now())

    author:   "User"         = relationship("User", back_populates="posts")
    tags:     "List[PostTag]"= relationship("PostTag", back_populates="post", lazy="selectin", cascade="all, delete-orphan")
    comments: "List[Comment]"= relationship("Comment", back_populates="post", lazy="selectin")


class PostTag(Base):
    __tablename__ = "post_tags"
    id:      Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(String(50), ForeignKey("posts.id"))
    tag:     Mapped[str] = mapped_column(String(100))
    post: "Post" = relationship("Post", back_populates="tags")


class Comment(Base):
    __tablename__ = "comments"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id:    Mapped[str]      = mapped_column(String(50), ForeignKey("posts.id"))
    author_id:  Mapped[str]      = mapped_column(String(50), ForeignKey("users.id"))
    body:       Mapped[str]      = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    post:   "Post" = relationship("Post", back_populates="comments")
    author: "User" = relationship("User", back_populates="comments")


class Like(Base):
    __tablename__ = "likes"
    user_id: Mapped[str] = mapped_column(String(50), ForeignKey("users.id"), primary_key=True)
    post_id: Mapped[str] = mapped_column(String(50), ForeignKey("posts.id"), primary_key=True)


class ModerationLog(Base):
    __tablename__ = "moderation_log"
    id:         Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id:    Mapped[str]      = mapped_column(String(50), ForeignKey("posts.id"))
    text_hash:  Mapped[str]      = mapped_column(String(64))
    label:      Mapped[str]      = mapped_column(String(30))
    confidence: Mapped[float]    = mapped_column(Float)
    level:      Mapped[int]      = mapped_column(Integer)
    reasons:    Mapped[str]      = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Recommendation(Base):
    __tablename__ = "recommendations"
    user_id:    Mapped[str]      = mapped_column(String(50), ForeignKey("users.id"), primary_key=True)
    post_id:    Mapped[str]      = mapped_column(String(50), ForeignKey("posts.id"), primary_key=True)
    score:      Mapped[float]    = mapped_column(Float)
    cbf_score:  Mapped[float]    = mapped_column(Float, default=0.0)
    svd_score:  Mapped[float]    = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ─── Вспомогательные функции ─────────────────────────────────────────────────
async def get_db():
    """Dependency для FastAPI — даёт сессию БД"""
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    """Создаёт таблицы и заполняет начальными данными"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_initial_data()

async def seed_initial_data():
    """Начальные данные: пользователи и демо-посты"""
    import hashlib
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        # Проверяем — если уже есть пользователи, не добавляем
        result = await session.execute(select(User).limit(1))
        if result.scalar():
            return

        # Пользователи
        users = [
            User(id="user1", name="Д-р Петров А.С.",  role="doctor",
                 password_hash=hashlib.sha256(b"doctor123").hexdigest(), posts_count=42, likes_count=287),
            User(id="user2", name="Мария Иванова",    role="patient",
                 password_hash=hashlib.sha256(b"patient123").hexdigest(), posts_count=8,  likes_count=34),
            User(id="admin", name="Администратор",    role="admin",
                 password_hash=hashlib.sha256(b"admin123").hexdigest(), posts_count=0, likes_count=0),
        ]
        session.add_all(users)

        # Демо-посты
        posts = [
            Post(id="p1", author_id="user1", status="approved", mod_conf=0.98, likes=34,
                 title="Реабилитация после инфаркта",
                 body="После перенесённого инфаркта важно соблюдать программу кардиореабилитации. Первые 6 недель — ограниченная активность, диета."),
            Post(id="p2", author_id="user2", status="approved", mod_conf=0.96, likes=12,
                 title="Как принимать метформин?",
                 body="Врач назначил метформин 500мг при диабете 2 типа. Когда лучше принимать — до или после еды?"),
            Post(id="p3", author_id="user1", status="approved", mod_conf=0.99, likes=56,
                 title="Профилактика ОРВИ у детей",
                 body="Закаливание, проветривание, промывание носа физраствором и вакцинация от гриппа."),
        ]
        session.add_all(posts)

        # Теги
        tags = [
            PostTag(post_id="p1", tag="кардиология"), PostTag(post_id="p1", tag="реабилитация"),
            PostTag(post_id="p2", tag="диабет"),       PostTag(post_id="p2", tag="препараты"),
            PostTag(post_id="p3", tag="педиатрия"),    PostTag(post_id="p3", tag="ОРВИ"),
        ]
        session.add_all(tags)
        await session.commit()
        print("✓ БД инициализирована начальными данными")
