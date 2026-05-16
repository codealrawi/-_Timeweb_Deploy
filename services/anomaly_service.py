"""
================================================================================
Account Anomaly Detection Service — SVD on Behavioral Matrix
================================================================================

Реализация метода обнаружения аномальных аккаунтов из статьи:
«Метод обнаружения аномальных аккаунтов в медицинских социальных платформах
 на основе сингулярного разложения матрицы поведения»

Алгоритм (точно по разделу «Материалы и методы» статьи):
  1. Построение поведенческой матрицы X ∈ ℝ^{m×9} (m аккаунтов × 9 признаков).
  2. Z-score нормализация (формула 1).
  3. Усечённое SVD X̃ = U·Σ·V^T (формула 2).
  4. Выбор k по критерию объяснённой дисперсии ≥ 90 % (формула 4).
  5. Низкоранговая реконструкция X̂_k = U_k·Σ_k·V_k^T (формула 3).
  6. Норма остатка e_i = ‖x_i − x̂_{k,i}‖_2 (формула 5).
  7. Порог τ = P_95({e_i}) (формула 6).
  8. Классификация: аккаунт аномальный ⇔ e_i > τ.

Авторы статьи: Аль-Раве М.И.Т., Макаров А.В.
"""

from __future__ import annotations

import math
import re
import logging
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("anomaly_service")

# ──────────────────────────────────────────────────────────────────────────────
# Константы метода (точно соответствуют статье)
# ──────────────────────────────────────────────────────────────────────────────
EXPLAINED_VARIANCE_THRESHOLD = 0.90   # ≥ 90 % суммарной энергии для выбора k
PERCENTILE_THRESHOLD = 95.0           # τ = 95-й процентиль распределения ошибок

# Имена девяти поведенческих признаков
FEATURE_KEYS = [
    "p1_post_freq",
    "p2_like_freq",
    "p3_medical_lexicon",
    "p4_external_links",
    "p5_time_entropy",
    "p6_avg_msg_length",
    "p7_uppercase_share",
    "p8_interval_variance",
    "p9_repeated_ngrams",
]

FEATURE_LABELS = [
    "p1 — частота публикаций",
    "p2 — частота лайков",
    "p3 — доля медицинской лексики",
    "p4 — доля внешних ссылок",
    "p5 — энтропия временных интервалов",
    "p6 — средняя длина сообщения",
    "p7 — доля заглавных букв",
    "p8 — дисперсия интервалов",
    "p9 — доля повторяющихся n-грамм",
]

# Медицинский словарь (расширенная версия)
MEDICAL_TERMS = {
    "диагноз", "симптом", "симптомы", "лечение", "терапия", "препарат", "препараты",
    "врач", "пациент", "пациенты", "анализ", "анализы", "операция", "хирург",
    "реабилитация", "антибиотик", "антибиотики", "дозировка", "диабет",
    "кардиология", "кардиолог", "кардиореабилитация", "педиатрия", "педиатр",
    "невролог", "неврология", "онколог", "онкология", "терапевт", "хирургия",
    "давление", "температура", "кашель", "боль", "болезнь", "гипертония",
    "аллергия", "инъекция", "процедура", "диагностика", "мрт", "узи", "кт", "экг",
    "госпитализация", "вакцина", "вакцинация", "профилактика", "инфаркт",
    "миокард", "аритмия", "экстрасистолы", "эналаприл", "метформин",
    "эндокринология", "гипотиреоз", "ттг", "тироксин", "орви", "грипп",
    "парацетамол", "ибупрофен", "мигрень", "пневмония", "хба1с", "клиника",
    "патология", "синдром", "лимфоузел", "кровь", "сердце", "лёгкие", "печень",
    "почки", "артерия", "вена", "холестерин", "глюкоза", "инсулин", "гормон",
    "обследование", "узи", "ктг", "вес", "рост", "артрит", "артроз", "остеохондроз",
    "грыжа", "межпозвонк", "позвоночник", "сосуд", "тромбоз", "анемия", "лейкоцит",
    "тромбоцит", "эритроцит", "гемоглобин", "соэ", "коагулограмма", "биохимия",
}

# Регулярка для внешних ссылок
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w-]+\b", re.UNICODE)


# ──────────────────────────────────────────────────────────────────────────────
# Вычисление поведенческих признаков по данным одного аккаунта
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class UserActivity:
    """Сырые данные одного аккаунта для вычисления признаков."""
    user_id: str
    name: str
    role: str
    created_at: Optional[datetime] = None
    posts: List[Dict] = field(default_factory=list)        # [{body, title, created_at, likes_count}, ...]
    likes_given: int = 0                                     # сколько лайков поставил пользователь
    is_synthetic: bool = False


def _parse_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def compute_features(activity: UserActivity, now: Optional[datetime] = None) -> Dict[str, float]:
    """
    Вычисляет 9 поведенческих признаков (p1..p9) для одного аккаунта.

    Все признаки нормированы к содержательным масштабам:
      p1 — посты/сутки;            p2 — лайки/сутки;
      p3 ∈ [0,1] — доля мед. лексики;     p4 ∈ [0,1] — доля постов со ссылками;
      p5 ≥ 0 — энтропия по 24 часам (макс ≈ ln 24 ≈ 3.18);
      p6 ≥ 0 — средняя длина сообщения (символов);
      p7 ∈ [0,1] — доля заглавных букв;
      p8 ≥ 0 — дисперсия интервалов (часы^2);
      p9 ∈ [0,1] — доля повторяющихся биграмм.
    """
    now = now or datetime.now(timezone.utc)
    posts = activity.posts
    n_posts = len(posts)

    # ── Период активности (сутки) ──
    times = [_parse_dt(p.get("created_at")) for p in posts]
    times = [t for t in times if t is not None]
    times.sort()
    if times:
        period_days = max((times[-1] - times[0]).total_seconds() / 86400, 1.0)
    elif activity.created_at:
        period_days = max((now - activity.created_at).total_seconds() / 86400, 1.0)
    else:
        period_days = 30.0  # дефолт

    # p1: посты/сутки
    p1 = n_posts / period_days

    # p2: лайки/сутки (лайков ПОСТАВЛЕНО пользователем)
    p2 = activity.likes_given / period_days

    # Собираем тексты всех постов
    texts: List[str] = []
    for p in posts:
        body = (p.get("body") or "") + " " + (p.get("title") or "")
        texts.append(body)
    full_text = " ".join(texts)
    full_text_lower = full_text.lower()
    words = WORD_RE.findall(full_text_lower)
    total_words = len(words)

    # p3: доля медицинской лексики
    if total_words > 0:
        med_count = sum(1 for w in words if w in MEDICAL_TERMS)
        p3 = med_count / total_words
    else:
        p3 = 0.0

    # p4: доля постов с внешними ссылками
    if n_posts > 0:
        with_links = sum(1 for t in texts if URL_RE.search(t))
        p4 = with_links / n_posts
    else:
        p4 = 0.0

    # p5: энтропия временных интервалов (часы суток)
    if len(times) >= 2:
        hours = [t.hour for t in times]
        counts = np.bincount(hours, minlength=24).astype(float)
        probs = counts / counts.sum()
        # Энтропия Шеннона
        with np.errstate(divide="ignore", invalid="ignore"):
            p5 = float(-np.sum([pp * math.log(pp) for pp in probs if pp > 0]))
    else:
        p5 = 0.0

    # p6: средняя длина сообщения
    if n_posts > 0:
        p6 = float(np.mean([len(t) for t in texts]))
    else:
        p6 = 0.0

    # p7: доля заглавных букв
    if full_text:
        upper = sum(1 for c in full_text if c.isupper())
        letters = sum(1 for c in full_text if c.isalpha())
        p7 = upper / letters if letters > 0 else 0.0
    else:
        p7 = 0.0

    # p8: дисперсия межпубликационных интервалов (часы)
    if len(times) >= 3:
        intervals = [
            (times[i + 1] - times[i]).total_seconds() / 3600
            for i in range(len(times) - 1)
        ]
        p8 = float(np.var(intervals))
    else:
        p8 = 0.0

    # p9: доля повторяющихся биграмм
    bigrams: List[Tuple[str, str]] = []
    for t in texts:
        toks = WORD_RE.findall(t.lower())
        bigrams.extend((toks[i], toks[i + 1]) for i in range(len(toks) - 1))
    if bigrams:
        counter = Counter(bigrams)
        repeated = sum(c for c in counter.values() if c > 1)
        p9 = repeated / len(bigrams)
    else:
        p9 = 0.0

    return {
        "p1_post_freq":         round(p1, 4),
        "p2_like_freq":         round(p2, 4),
        "p3_medical_lexicon":   round(p3, 4),
        "p4_external_links":    round(p4, 4),
        "p5_time_entropy":      round(p5, 4),
        "p6_avg_msg_length":    round(p6, 2),
        "p7_uppercase_share":   round(p7, 4),
        "p8_interval_variance": round(p8, 4),
        "p9_repeated_ngrams":   round(p9, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Основной детектор (SVD)
# ──────────────────────────────────────────────────────────────────────────────
class AccountAnomalyDetector:
    """
    Метод обнаружения аномальных аккаунтов на основе SVD поведенческой матрицы.

    Использование:
        detector = AccountAnomalyDetector()
        result = detector.run(activities)   # список UserActivity
        # result содержит: per-user features/errors/flags, loadings V_k, k, τ.
    """

    def __init__(
        self,
        explained_variance_threshold: float = EXPLAINED_VARIANCE_THRESHOLD,
        percentile: float = PERCENTILE_THRESHOLD,
    ):
        self.explained_variance_threshold = explained_variance_threshold
        self.percentile = percentile

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, activities: List[UserActivity]) -> Dict:
        """
        Запускает полный пайплайн на списке UserActivity.

        Возвращает словарь с:
            n_total, n_anomalies, k_components, threshold,
            explained_variance, feature_names, loadings, users.
        """
        if len(activities) < 3:
            return self._too_few(activities)

        # 1. Вычисление признаков
        feat_dicts = [compute_features(a) for a in activities]
        X = np.array([[d[k] for k in FEATURE_KEYS] for d in feat_dicts], dtype=float)
        m, n = X.shape

        # 2. z-score нормализация (формула 1)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma < 1e-12] = 1.0     # защита от деления на ноль
        X_norm = (X - mu) / sigma

        # 3. SVD (формула 2)
        try:
            U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.warning("SVD не сошлось — возвращаем нейтральный результат")
            return self._too_few(activities)

        # 4. Выбор k по объяснённой дисперсии (формула 4)
        total_energy = float(np.sum(S ** 2))
        if total_energy < 1e-12:
            return self._too_few(activities)
        explained = (S ** 2) / total_energy
        cumulative = np.cumsum(explained)
        k = int(np.searchsorted(cumulative, self.explained_variance_threshold) + 1)
        k = max(1, min(k, n - 1, len(S)))

        # 5. Низкоранговая реконструкция X̂_k (формула 3)
        # X_hat = U[:,:k] · diag(S[:k]) · Vt[:k,:]
        X_hat = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

        # 6. Норма остатка реконструкции (формула 5)
        errors = np.linalg.norm(X_norm - X_hat, axis=1)

        # 7. Порог (формула 6)
        threshold = float(np.percentile(errors, self.percentile))

        # 8. Классификация
        flags = errors > threshold

        # ── Сборка результата ──
        loadings = Vt[:k, :].T            # shape (n_features, k)
        loadings_rows = []
        for j in range(n):
            row = {"feature": FEATURE_LABELS[j], "key": FEATURE_KEYS[j]}
            for p in range(k):
                row[f"pc{p+1}"] = round(float(loadings[j, p]), 4)
            loadings_rows.append(row)

        users_out = []
        for i, act in enumerate(activities):
            users_out.append({
                "id":                  act.user_id,
                "name":                act.name,
                "role":                act.role,
                "is_synthetic":        act.is_synthetic,
                "features":            feat_dicts[i],
                "reconstruction_error": round(float(errors[i]), 4),
                "is_anomalous":        bool(flags[i]),
            })

        return {
            "n_total":                int(m),
            "n_anomalies":            int(flags.sum()),
            "k_components":           int(k),
            "threshold":              round(threshold, 4),
            "percentile":             self.percentile,
            "explained_variance_ratio": [round(float(e), 4) for e in explained[:k]],
            "explained_variance_total": round(float(cumulative[k-1]), 4),
            "singular_values":        [round(float(s), 4) for s in S[:k]],
            "feature_names":          FEATURE_LABELS,
            "feature_keys":           FEATURE_KEYS,
            "loadings":               loadings_rows,
            "users":                  users_out,
        }

    @staticmethod
    def _too_few(activities: List[UserActivity]) -> Dict:
        """Запасной путь при m < 3 или вырожденной выборке."""
        return {
            "n_total":                len(activities),
            "n_anomalies":            0,
            "k_components":           0,
            "threshold":              0.0,
            "percentile":             PERCENTILE_THRESHOLD,
            "explained_variance_ratio": [],
            "explained_variance_total": 0.0,
            "singular_values":        [],
            "feature_names":          FEATURE_LABELS,
            "feature_keys":           FEATURE_KEYS,
            "loadings":               [],
            "users": [{
                "id": a.user_id, "name": a.name, "role": a.role,
                "is_synthetic": a.is_synthetic,
                "features": compute_features(a),
                "reconstruction_error": 0.0,
                "is_anomalous": False,
            } for a in activities],
            "warning": "Слишком мало аккаунтов для SVD-анализа (требуется ≥ 3). "
                       "Используйте synthetic=true для демонстрации алгоритма.",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Генератор синтетических аккаунтов для демонстрации
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_activities(
    n_legit: int = 45,
    n_anomalies: int = 5,
    seed: int = 42,
) -> List[UserActivity]:
    """
    Генерирует список UserActivity, имитирующих три типа легитимных
    пользователей и четыре типа аномальных аккаунтов (боты, спам,
    накрутчики, агрессивные маркетологи). Используется для демонстрации
    работы алгоритма, когда в системе мало реальных аккаунтов.

    Поведение синтетических активностей детерминировано (seed = 42),
    поэтому результаты воспроизводимы между запусками.
    """
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    activities: List[UserActivity] = []

    # Богатый словарь для легитимных пользователей (>200 слов) — низкий p9
    legit_vocab = [
        "доктор", "лечился", "болело", "горло", "выписали", "помогло", "приём",
        "консультация", "обследование", "получил", "результат", "пришёл",
        "обратился", "клиника", "регистратура", "очередь", "талон", "запись",
        "анализы", "сдавал", "поликлиника", "стационар", "выписка", "санаторий",
        "лекарство", "капельница", "укол", "таблетки", "капли", "мазь", "ингаляция",
        "режим", "питание", "диета", "вес", "сон", "усталость", "слабость",
        "температура", "озноб", "потливость", "пульс", "давление", "кашель", "насморк",
        "горло", "грудь", "живот", "спина", "ноги", "руки", "колено", "плечо",
        "голова", "шея", "глаза", "уши", "память", "концентрация", "настроение",
        "первый", "второй", "вчера", "сегодня", "недавно", "давно", "целый", "месяц",
        "неделя", "год", "утром", "днём", "вечером", "ночью", "часто", "редко",
        "сильно", "слегка", "терпимо", "мучительно", "беспокоит", "тревожит",
        "посоветовали", "решили", "поставили", "назначили", "отменили", "заменили",
        "спросил", "ответил", "обсудили", "разъяснил", "пояснил", "уточнил",
        "благодарю", "спасибо", "интересно", "понятно", "сложно", "странно",
        "семья", "жена", "муж", "сын", "дочь", "ребёнок", "мама", "папа",
        "погода", "холодно", "тепло", "сыро", "сухо", "влажно", "душно",
        "работа", "отпуск", "стресс", "переутомление", "выходной", "праздник",
        "город", "район", "поликлиника", "филиал", "адрес", "транспорт",
    ]
    # Маленький стабильный пул для бота-накрутчика — высокий p9
    template_padder_phrase = (
        "прекрасный специалист очень рекомендую обратиться "
        "по любым вопросам здоровья замечательный врач благодарю "
        "за профессионализм и внимание к пациенту"
    ).split()
    # Минимальный пул для гиперактивного бота
    bot_minimal = "новость информация ссылка переход переходи срочно акция".split()
    # Спам-лексика
    spam_vocab = "скидка бесплатно купить акция промокод переходи".split()

    # ── Конфигурация легитимных пользователей ──
    legit_profiles = [
        {"type": "moderate_patient",
         "post_per_day": (1.0, 0.5), "len_words": (25, 8), "med_freq": 0.20,
         "ext_link_p": 0.05, "upper_p": 0.04, "vocab": legit_vocab,
         "circadian": True, "likes_per_day": (4, 2)},
        {"type": "active_doctor",
         "post_per_day": (3.5, 1.0), "len_words": (35, 10), "med_freq": 0.35,
         "ext_link_p": 0.10, "upper_p": 0.05, "vocab": legit_vocab,
         "circadian": True, "likes_per_day": (8, 3)},
        {"type": "lurker",
         "post_per_day": (0.3, 0.2), "len_words": (15, 5), "med_freq": 0.12,
         "ext_link_p": 0.03, "upper_p": 0.03, "vocab": legit_vocab,
         "circadian": True, "likes_per_day": (1, 1)},
    ]

    # ── Конфигурация аномальных аккаунтов ──
    anomaly_profiles = [
        {"type": "hyperactive_bot",
         "post_per_day": (38, 6), "len_words": (10, 3), "med_freq": 0.02,
         "ext_link_p": 0.20, "upper_p": 0.08, "vocab": bot_minimal,
         "circadian": False, "likes_per_day": (1, 1)},
        {"type": "spam_account",
         "post_per_day": (9, 2), "len_words": (22, 5), "med_freq": 0.04,
         "ext_link_p": 0.70, "upper_p": 0.10, "vocab": spam_vocab + legit_vocab[:20],
         "circadian": True, "likes_per_day": (1, 1)},
        {"type": "template_padder",
         "post_per_day": (5, 1.5), "len_words": (18, 3), "med_freq": 0.06,
         "ext_link_p": 0.10, "upper_p": 0.04, "vocab": template_padder_phrase,
         "circadian": True, "likes_per_day": (1, 1)},
        {"type": "aggressive_marketing",
         "post_per_day": (6, 2), "len_words": (30, 8), "med_freq": 0.06,
         "ext_link_p": 0.30, "upper_p": 0.55, "vocab": spam_vocab + legit_vocab[:30],
         "circadian": True, "likes_per_day": (2, 1)},
    ]

    def _build_synthetic(profile: dict, idx: int, is_anom: bool):
        from datetime import timedelta

        # Имитация активности за 30 дней
        ppd_mean, ppd_std = profile["post_per_day"]
        ppd = max(0.05, rng.normal(ppd_mean, ppd_std))
        n_posts = max(3, int(ppd * 30))
        n_posts = min(n_posts, 400)

        # ── Времена постов ──
        start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        post_times = []
        if profile["circadian"]:
            # Циркадное распределение — пики в 9-21
            for _ in range(n_posts):
                day_offset = int(rng.integers(0, 30))
                hour = int(np.clip(rng.normal(14, 4), 6, 23))
                minute = int(rng.integers(0, 60))
                post_times.append(start - timedelta(days=day_offset) +
                                  timedelta(hours=hour - 12, minutes=minute))
        else:
            # Бот — фиксированный интервал, минимальная дисперсия
            interval = 30 * 24 / n_posts
            for kk in range(n_posts):
                jitter = rng.normal(0, 0.05)
                post_times.append(start - timedelta(hours=interval * kk + jitter))
        post_times.sort()

        # ── Тексты постов ──
        med_words = ["диагноз", "симптом", "лечение", "терапия", "пациент",
                     "врач", "препарат", "диабет", "давление", "кардиология",
                     "анализ", "реабилитация", "профилактика", "грипп",
                     "вакцинация", "профилактика", "невролог", "терапевт"]
        vocab = profile["vocab"]
        lw_mean, lw_std = profile["len_words"]

        posts = []
        for t in post_times:
            n_words = max(3, int(rng.normal(lw_mean, lw_std)))
            words: List[str] = []

            if profile["type"] == "template_padder":
                # Почти одинаковый текст с мелким вариативным префиксом/суффиксом
                base = list(profile["vocab"])
                # Иногда добавляем 1–3 случайных слова в начале/конце для имитации
                # лёгкой персонализации.
                noise_count = int(rng.integers(0, 3))
                noise = [str(rng.choice(legit_vocab)) for _ in range(noise_count)]
                words = noise + base + (noise[:1] if noise else [])
            elif profile["type"] == "hyperactive_bot":
                # Очень короткие шаблонные посты
                for _ in range(n_words):
                    words.append(str(rng.choice(vocab)))
            else:
                # Нормальный или спам-микс
                for _ in range(n_words):
                    u = rng.random()
                    if u < profile["med_freq"]:
                        words.append(str(rng.choice(med_words)))
                    else:
                        words.append(str(rng.choice(vocab)))

            text = " ".join(words)
            # Внешние ссылки
            if rng.random() < profile["ext_link_p"]:
                text += " https://example.com/promo"
            # Caps lock?
            if profile["type"] == "aggressive_marketing" and rng.random() < profile["upper_p"]:
                text = text.upper()
            posts.append({
                "title": text[:60],
                "body":  text,
                "created_at":  t.isoformat(),
                "likes_count": int(max(0, rng.normal(3, 2))),
            })

        # ── Лайки, поставленные пользователем (lp = likes per day) ──
        lpd_mean, lpd_std = profile["likes_per_day"]
        likes_given = int(max(0, rng.normal(lpd_mean * 30, lpd_std * 30)))

        uid_hash = hashlib.md5(f"syn-{profile['type']}-{idx}".encode()).hexdigest()[:8]
        return UserActivity(
            user_id=f"syn_{uid_hash}",
            name=f"[{'BOT' if is_anom else 'SYN'}] {profile['type']}_{idx}",
            role="patient",
            posts=posts,
            likes_given=likes_given,
            is_synthetic=True,
        )

    for i in range(n_legit):
        profile = legit_profiles[i % len(legit_profiles)]
        activities.append(_build_synthetic(profile, i, is_anom=False))

    for i in range(n_anomalies):
        profile = anomaly_profiles[i % len(anomaly_profiles)]
        activities.append(_build_synthetic(profile, i, is_anom=True))

    return activities
