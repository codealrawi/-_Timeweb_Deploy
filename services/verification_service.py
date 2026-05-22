"""
Verification Service — система верификации врачей и пациентов.

Patient: email → identity document → автоматическая верификация.
Doctor:  email → diploma + license + certificates → UNDER_REVIEW → модератор approve/reject.
"""

import time
import random
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("verification")

# ── Статусы верификации ──
STATUSES = ("UNVERIFIED", "PENDING", "UNDER_REVIEW", "VERIFIED", "REJECTED", "SUSPENDED")

# ── Типы документов ──
PATIENT_DOC_TYPES = ("passport", "national_id")
DOCTOR_DOC_TYPES  = ("diploma", "medical_license", "certificate", "specialization")

# ── In-memory хранилище ──
_VERIFICATIONS: Dict[str, dict] = {}
_EMAIL_CODES:   Dict[str, dict] = {}   # user_id → {code, email, expires_at}
_COOLDOWNS:     Dict[str, float] = {}   # user_id → last_send_timestamp


def _now():
    return datetime.now(timezone.utc)


def _ts():
    return _now().strftime("%Y-%m-%d %H:%M:%S")


def _get_or_create(user_id: str, role: str) -> dict:
    """Получить или создать запись верификации."""
    if user_id not in _VERIFICATIONS:
        _VERIFICATIONS[user_id] = {
            "user_id":          user_id,
            "role":             role,
            "status":           "UNVERIFIED",
            "email":            None,
            "email_verified":   False,
            "documents":        [],
            "ai_score":         None,
            "ai_details":       None,
            "reviewer_id":      None,
            "reviewer_notes":   None,
            "reviewed_at":      None,
            "created_at":       _ts(),
            "updated_at":       _ts(),
            "audit_log":        [],
        }
    return _VERIFICATIONS[user_id]


def _log(v: dict, action: str, detail: str = ""):
    v["audit_log"].append({
        "timestamp": _ts(),
        "action":    action,
        "detail":    detail,
    })
    v["updated_at"] = _ts()


# ══════════════════════════════════════════════════════════════
# EMAIL VERIFICATION
# ══════════════════════════════════════════════════════════════

def send_email_code(user_id: str, role: str, email: str) -> dict:
    """Генерирует 6-значный код и отправляет на email (SMTP или демо-режим)."""
    # Anti-spam cooldown (30 секунд)
    last = _COOLDOWNS.get(user_id, 0)
    if time.time() - last < 30:
        wait = int(30 - (time.time() - last))
        return {"ok": False, "error": f"Подождите {wait} сек. перед повторной отправкой"}

    v = _get_or_create(user_id, role)
    code = f"{random.randint(100000, 999999)}"
    expires = _now() + timedelta(minutes=10)

    _EMAIL_CODES[user_id] = {
        "code":       code,
        "email":      email,
        "expires_at": expires,
    }
    _COOLDOWNS[user_id] = time.time()
    v["email"] = email
    _log(v, "EMAIL_CODE_SENT", f"code sent to {email}")

    # ── Попытка отправки через SMTP ──
    email_sent = _try_send_smtp(email, code)

    logger.info(f"[VERIFY] Email code for {user_id}: {code} (smtp={'OK' if email_sent else 'DEMO'})")

    result = {
        "ok":      True,
        "message": f"Код отправлен на {email}",
        "expires_in_sec": 600,
        "email_sent": email_sent,
    }
    # Если SMTP не настроен — возвращаем код для демо-режима
    if not email_sent:
        result["_demo_code"] = code
        result["_demo_mode"] = True
    return result


def _try_send_smtp(to_email: str, code: str) -> bool:
    """
    Отправляет код по email через SMTP.

    Поддерживает ДВА режима:
      - Порт 465 → SMTP_SSL (для Timeweb Cloud и большинства хостингов)
      - Порт 587 → STARTTLS (для локальной разработки)

    Настраивается через переменные окружения:
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
    """
    import os
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "465"))
    user = os.getenv("SMTP_USER", "")
    pwd  = os.getenv("SMTP_PASS", "")
    from_addr = os.getenv("SMTP_FROM", user)

    if not host or not user or not pwd:
        logger.info("[SMTP] Not configured (SMTP_HOST/USER/PASS missing)")
        return False

    try:
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(
            f"Ваш код верификации МедПлатформы: {code}\n\n"
            f"Код действителен 10 минут.\n"
            f"Если вы не запрашивали код — проигнорируйте это письмо.",
            "plain", "utf-8"
        )
        msg["Subject"] = f"МедПлатформа — код верификации: {code}"
        msg["From"]    = from_addr
        msg["To"]      = to_email

        if port == 465:
            # SSL — для Timeweb Cloud, Yandex, Gmail
            logger.info(f"[SMTP] Connecting SSL to {host}:{port}...")
            with smtplib.SMTP_SSL(host, port, timeout=15) as s:
                s.login(user, pwd)
                s.sendmail(from_addr, [to_email], msg.as_string())
        else:
            # STARTTLS — порт 587 (может быть заблокирован на хостинге)
            logger.info(f"[SMTP] Connecting STARTTLS to {host}:{port}...")
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.starttls()
                s.login(user, pwd)
                s.sendmail(from_addr, [to_email], msg.as_string())

        logger.info(f"[SMTP] ✓ Code sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"[SMTP] ✗ Failed to send to {to_email}: {type(e).__name__}: {e}")
        return False


def verify_email(user_id: str, role: str, code: str) -> dict:
    """Проверяет email-код."""
    v = _get_or_create(user_id, role)
    stored = _EMAIL_CODES.get(user_id)

    if not stored:
        return {"ok": False, "error": "Код не найден — запросите новый"}
    if _now() > stored["expires_at"]:
        del _EMAIL_CODES[user_id]
        return {"ok": False, "error": "Код истёк — запросите новый"}
    if stored["code"] != code.strip():
        return {"ok": False, "error": "Неверный код"}

    v["email_verified"] = True
    if v["status"] == "UNVERIFIED":
        v["status"] = "PENDING"
    del _EMAIL_CODES[user_id]
    _log(v, "EMAIL_VERIFIED", f"email {stored['email']} verified")

    return {"ok": True, "message": "Email подтверждён", "status": v["status"]}


# ══════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD + AI CHECK
# ══════════════════════════════════════════════════════════════

def upload_document(user_id: str, role: str, doc_type: str, filename: str,
                    file_size: int = 0) -> dict:
    """Загружает документ и запускает симуляцию AI-проверки."""
    v = _get_or_create(user_id, role)

    # Валидация типа
    allowed = DOCTOR_DOC_TYPES if role == "doctor" else PATIENT_DOC_TYPES
    if doc_type not in allowed:
        return {"ok": False, "error": f"Тип '{doc_type}' недопустим для роли {role}"}

    # Валидация формата
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "png", "jpg", "jpeg"):
        return {"ok": False, "error": f"Формат .{ext} не поддерживается (pdf, png, jpg, jpeg)"}

    # Проверка дубликатов
    for d in v["documents"]:
        if d["doc_type"] == doc_type:
            v["documents"].remove(d)
            _log(v, "DOC_REPLACED", f"replaced {doc_type}")
            break

    # ── Симуляция AI-проверки документа ──
    time.sleep(0.05)  # имитация задержки OCR
    ai_score = round(random.uniform(0.75, 0.99), 3)
    ai_checks = {
        "format_valid":     True,
        "ocr_readable":     ai_score > 0.5,
        "name_match":       ai_score > 0.6,
        "expiry_valid":     ai_score > 0.4,
        "fraud_score":      round(1 - ai_score, 3),
    }

    doc = {
        "doc_type":     doc_type,
        "filename":     filename,
        "file_size":    file_size,
        "uploaded_at":  _ts(),
        "ai_score":     ai_score,
        "ai_checks":    ai_checks,
    }
    v["documents"].append(doc)
    _log(v, "DOC_UPLOADED", f"{doc_type}: {filename} (AI score: {ai_score})")

    return {
        "ok":        True,
        "document":  doc,
        "message":   f"Документ '{doc_type}' загружен, AI score: {ai_score}",
    }


def submit_for_review(user_id: str, role: str) -> dict:
    """
    Завершить загрузку документов и отправить на проверку.
    Patient → автоматическая верификация (если всё ОК).
    Doctor  → переход в UNDER_REVIEW для модератора.
    """
    v = _get_or_create(user_id, role)

    if not v["email_verified"]:
        return {"ok": False, "error": "Сначала подтвердите email"}

    doc_types_uploaded = {d["doc_type"] for d in v["documents"]}

    if role == "patient":
        required = {"passport"}
        missing = required - doc_types_uploaded
        if missing:
            return {"ok": False, "error": f"Не загружены документы: {', '.join(missing)}"}

        # Автоматическая верификация пациента
        avg_score = sum(d["ai_score"] for d in v["documents"]) / len(v["documents"])
        v["ai_score"] = round(avg_score, 3)

        if avg_score >= 0.7:
            v["status"] = "VERIFIED"
            _log(v, "AUTO_VERIFIED", f"Patient verified (AI avg: {avg_score:.3f})")
            return {"ok": True, "status": "VERIFIED",
                    "message": "Аккаунт верифицирован автоматически"}
        else:
            v["status"] = "UNDER_REVIEW"
            _log(v, "SENT_TO_REVIEW", f"Low AI score ({avg_score:.3f}), needs manual review")
            return {"ok": True, "status": "UNDER_REVIEW",
                    "message": "AI-проверка требует ручного обзора"}

    elif role == "doctor":
        required = {"diploma", "medical_license"}
        missing = required - doc_types_uploaded
        if missing:
            return {"ok": False, "error": f"Не загружены обязательные документы: {', '.join(missing)}"}

        avg_score = sum(d["ai_score"] for d in v["documents"]) / len(v["documents"])
        v["ai_score"] = round(avg_score, 3)
        v["ai_details"] = {
            "avg_score": round(avg_score, 3),
            "docs_count": len(v["documents"]),
            "all_checks_passed": all(
                d["ai_checks"]["ocr_readable"] and d["ai_checks"]["name_match"]
                for d in v["documents"]
            ),
        }
        v["status"] = "UNDER_REVIEW"
        _log(v, "SENT_TO_REVIEW", f"Doctor sent for moderator review (AI avg: {avg_score:.3f})")
        return {"ok": True, "status": "UNDER_REVIEW",
                "message": "Документы отправлены на проверку модератору"}

    return {"ok": False, "error": "Неизвестная роль"}


# ══════════════════════════════════════════════════════════════
# MODERATOR ACTIONS
# ══════════════════════════════════════════════════════════════

def get_pending_verifications() -> List[dict]:
    """Возвращает список пользователей со статусом UNDER_REVIEW."""
    return [
        {**v, "audit_log": v["audit_log"][-5:]}  # последние 5 записей
        for v in _VERIFICATIONS.values()
        if v["status"] == "UNDER_REVIEW"
    ]


def approve_verification(user_id: str, reviewer_id: str, notes: str = "") -> dict:
    """Модератор одобряет верификацию."""
    v = _VERIFICATIONS.get(user_id)
    if not v:
        return {"ok": False, "error": "Запись верификации не найдена"}
    if v["status"] != "UNDER_REVIEW":
        return {"ok": False, "error": f"Статус '{v['status']}' — нельзя одобрить"}

    v["status"]         = "VERIFIED"
    v["reviewer_id"]    = reviewer_id
    v["reviewer_notes"] = notes
    v["reviewed_at"]    = _ts()
    _log(v, "APPROVED", f"by {reviewer_id}: {notes or 'no notes'}")

    return {"ok": True, "status": "VERIFIED", "message": "Верификация одобрена"}


def reject_verification(user_id: str, reviewer_id: str, notes: str = "") -> dict:
    """Модератор отклоняет верификацию."""
    v = _VERIFICATIONS.get(user_id)
    if not v:
        return {"ok": False, "error": "Запись верификации не найдена"}
    if v["status"] not in ("UNDER_REVIEW", "PENDING"):
        return {"ok": False, "error": f"Статус '{v['status']}' — нельзя отклонить"}

    v["status"]         = "REJECTED"
    v["reviewer_id"]    = reviewer_id
    v["reviewer_notes"] = notes
    v["reviewed_at"]    = _ts()
    _log(v, "REJECTED", f"by {reviewer_id}: {notes or 'no notes'}")

    return {"ok": True, "status": "REJECTED", "message": "Верификация отклонена"}


def get_status(user_id: str, role: str) -> dict:
    """Получить полный статус верификации пользователя."""
    v = _get_or_create(user_id, role)
    doc_types = {d["doc_type"] for d in v["documents"]}
    required = {"diploma", "medical_license"} if role == "doctor" else {"passport"}

    return {
        "user_id":          user_id,
        "role":             role,
        "status":           v["status"],
        "email":            v["email"],
        "email_verified":   v["email_verified"],
        "documents":        v["documents"],
        "docs_uploaded":    sorted(doc_types),
        "docs_required":    sorted(required),
        "docs_missing":     sorted(required - doc_types),
        "ai_score":         v["ai_score"],
        "ai_details":       v.get("ai_details"),
        "reviewer_id":      v["reviewer_id"],
        "reviewer_notes":   v["reviewer_notes"],
        "reviewed_at":      v["reviewed_at"],
        "audit_log":        v["audit_log"][-10:],
        "created_at":       v["created_at"],
        "updated_at":       v["updated_at"],
    }


def get_stats() -> dict:
    """Статистика верификации для админа."""
    by_status = defaultdict(int)
    by_role = defaultdict(lambda: defaultdict(int))
    for v in _VERIFICATIONS.values():
        by_status[v["status"]] += 1
        by_role[v["role"]][v["status"]] += 1
    return {
        "total":      len(_VERIFICATIONS),
        "by_status":  dict(by_status),
        "by_role":    {r: dict(s) for r, s in by_role.items()},
    }
