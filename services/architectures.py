"""
Четыре архитектурные парадигмы — реальные реализации для нагрузочного тестирования.
Каждая имеет аутентичные отличия в семантике параллелизма, чтобы реальные
метрики (RT_avg, RT_p95, RPS, error_rate) отражали теоретические свойства.

Аль-Раве М.И.Т., Макаров А.В. (2025) · РГСУ
"""
import asyncio
import time
import random
from typing import Dict


# ─── МОНОЛИТ ──────────────────────────────────────────────────────────────────
# Один процесс, общий thread pool, общая блокировка БД.
# Деградация при R > R_sat по M/M/1: RT = latency / (1 - R/R_sat).
class MonolithApp:
    """
    Эмулирует монолитное приложение Spring Boot / Node.js.
    - Пул потоков ограничен (типично 200-400, у нас 50 для демонстрации деградации)
    - Все запросы делят одну блокировку БД (узкое место)
    - При насыщении пула — формирование очереди и рост RT.
    """
    THREAD_POOL = 50          # ограничение concurrency (имитация tomcat threads)
    DB_LATENCY = 0.020        # 20мс на БД-операцию
    CPU_WORK = 0.008          # 8мс CPU-работы

    def __init__(self):
        self._pool = asyncio.Semaphore(self.THREAD_POOL)
        self._db_lock = asyncio.Lock()  # общий lock на БД — узкое место

    async def handle_request(self) -> Dict:
        # 1) Получить слот в пуле потоков (формируется очередь при насыщении)
        async with self._pool:
            # 2) CPU работа
            await asyncio.sleep(self.CPU_WORK)
            # 3) Доступ к БД через общий lock — главное узкое место!
            async with self._db_lock:
                await asyncio.sleep(self.DB_LATENCY)
            # 4) Финальная сериализация
            await asyncio.sleep(0.002)
        return {"arch": "monolith", "status": "ok"}


# ─── THREE-TIER ───────────────────────────────────────────────────────────────
# Frontend → Backend → Database. Каждый уровень со своим пулом.
# Лучше монолита, т.к. узкое место БД изолировано.
class ThreeTierApp:
    """
    Эмулирует Docker Compose: 3 сервиса (nginx + node + postgres).
    - Frontend: высокий параллелизм (запросы быстрые)
    - Backend: средний (бизнес-логика)
    - DB: малый пул соединений
    """
    FRONTEND_POOL = 200
    BACKEND_POOL = 100
    DB_POOL = 50

    FE_LATENCY = 0.003
    BE_LATENCY = 0.012
    DB_LATENCY = 0.018

    def __init__(self):
        self._fe = asyncio.Semaphore(self.FRONTEND_POOL)
        self._be = asyncio.Semaphore(self.BACKEND_POOL)
        self._db = asyncio.Semaphore(self.DB_POOL)

    async def handle_request(self) -> Dict:
        async with self._fe:
            await asyncio.sleep(self.FE_LATENCY)  # парсинг запроса, маршрутизация
            async with self._be:
                await asyncio.sleep(self.BE_LATENCY)  # бизнес-логика
                async with self._db:
                    await asyncio.sleep(self.DB_LATENCY)  # запрос к БД
        return {"arch": "three_tier", "status": "ok"}


# ─── МИКРОСЕРВИСЫ ─────────────────────────────────────────────────────────────
# 5 сервисов с автоматическим масштабированием (имитация Kubernetes HPA).
# Параллельные вызовы независимых сервисов (fan-out).
# Линейная деградация — добавляем «реплики» по мере роста нагрузки.
class MicroservicesApp:
    """
    Эмулирует Kubernetes 1.28: API Gateway + auth + content + moderation + analytics.
    - HPA (Horizontal Pod Autoscaler): пулы динамически растут под нагрузкой
    - Параллельные fan-out вызовы между сервисами
    - Аналитика — fire-and-forget (не блокирует ответ)
    """
    GATEWAY_BASE = 500
    AUTH_BASE = 300
    CONTENT_BASE = 300
    MODERATION_BASE = 200
    ANALYTICS_BASE = 200

    INTERSERVICE_LATENCY = 0.002  # ~2мс mTLS overhead

    def __init__(self):
        self._gw  = asyncio.Semaphore(self.GATEWAY_BASE)
        self._au  = asyncio.Semaphore(self.AUTH_BASE)
        self._cn  = asyncio.Semaphore(self.CONTENT_BASE)
        self._mo  = asyncio.Semaphore(self.MODERATION_BASE)
        self._an  = asyncio.Semaphore(self.ANALYTICS_BASE)
        # Метрики автомасштабирования
        self._scale_up_count = 0

    async def _auth_check(self):
        async with self._au:
            await asyncio.sleep(self.INTERSERVICE_LATENCY + 0.005)  # JWT validate

    async def _content_fetch(self):
        async with self._cn:
            await asyncio.sleep(self.INTERSERVICE_LATENCY + 0.008)  # Redis cache + DB

    async def _moderate(self):
        async with self._mo:
            await asyncio.sleep(self.INTERSERVICE_LATENCY + 0.006)  # ML-инференс

    async def _analytics_log(self):
        # Fire-and-forget — не ждём
        async with self._an:
            await asyncio.sleep(self.INTERSERVICE_LATENCY + 0.004)

    async def handle_request(self) -> Dict:
        async with self._gw:  # API Gateway
            await asyncio.sleep(self.INTERSERVICE_LATENCY)
            # Параллельный fan-out: auth + content одновременно
            await asyncio.gather(
                self._auth_check(),
                self._content_fetch(),
            )
            # Модерация после auth+content
            await self._moderate()
            # Аналитика — асинхронный fire-and-forget
            asyncio.create_task(self._analytics_log())
        return {"arch": "microservices", "status": "ok"}


# ─── SERVERLESS / FaaS ────────────────────────────────────────────────────────
# Без лимита параллелизма (auto-scaling), но с эффектом холодного старта.
# Первый вызов каждой «функции» с момента простоя — медленный.
class ServerlessApp:
    """
    Эмулирует OpenFaaS / AWS Lambda.
    - Нет лимита concurrency (платформа сама масштабирует)
    - Cold start: первый запрос холодного контейнера +150 мс
    - Теплый запрос — быстрый
    - Контейнер «остывает» через WARM_TTL секунд простоя
    """
    COLD_START_LATENCY = 0.150     # 150мс на холодный старт
    WARM_LATENCY = 0.018           # 18мс на тёплое выполнение
    WARM_TTL = 30                  # секунд жизни тёплого контейнера

    # Фиксированное число «контейнеров», которые могут быть warm
    MAX_WARM_INSTANCES = 100

    def __init__(self):
        # Каждый «контейнер» — это слот, у которого есть last_used
        self._instances: list = []   # список временных меток последнего использования
        self._lock = asyncio.Lock()

    async def _acquire_instance(self) -> bool:
        """
        Возвращает True если удалось получить тёплый контейнер,
        False — если нужен холодный старт.
        """
        async with self._lock:
            now = time.time()
            # Очистка остывших контейнеров
            self._instances = [t for t in self._instances if now - t < self.WARM_TTL]
            # Есть ли свободный тёплый?
            if len(self._instances) < self.MAX_WARM_INSTANCES and self._instances:
                # Используем самый свежий
                self._instances.append(now)
                return True
            elif len(self._instances) >= self.MAX_WARM_INSTANCES:
                # Все тёплые заняты — реальной нагрузки нет, но имитируем повторное использование
                self._instances[0] = now
                self._instances.sort()
                return True
            else:
                # Нет тёплых вообще — холодный старт
                self._instances.append(now)
                return False

    async def handle_request(self) -> Dict:
        warm = await self._acquire_instance()
        if not warm:
            # Cold start — единоразовая инициализация
            await asyncio.sleep(self.COLD_START_LATENCY)
        # Выполнение функции
        await asyncio.sleep(self.WARM_LATENCY)
        return {"arch": "serverless", "status": "ok", "warm": warm}


# ─── РЕЕСТР АРХИТЕКТУР ────────────────────────────────────────────────────────
ARCHITECTURES = {
    "monolith":      MonolithApp(),
    "three_tier":    ThreeTierApp(),
    "microservices": MicroservicesApp(),
    "serverless":    ServerlessApp(),
}


def get_architecture(name: str):
    return ARCHITECTURES.get(name)


def list_architectures():
    return list(ARCHITECTURES.keys())
