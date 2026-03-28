"""
Нагрузочный тест МедПлатформа — запускать с ЛОКАЛЬНОГО ПК
против реального сервера Timeweb

Установка:  pip install locust
Запуск:     locust -f locustfile.py --host=http://ВАШ_IP
Браузер:    http://localhost:8089

Параметры для диссертации:
  Users: 50 / 100 / 200 / 500 / 1000 / 1500
  Spawn rate: 10 users/sec
  Duration: 5 минут на каждый уровень
"""
from locust import HttpUser, task, between
import random

SAMPLE_TEXTS = [
    "Как правильно принимать антибиотики при бронхите?",
    "Врач назначил метформин при диабете 2 типа. Когда принимать?",
    "Рекомендации по реабилитации после операции на колене",
    "Симптомы дефицита витамина D — что сдать?",
    "Гарантированное излечение рака за 3 дня без врачей!",
]

class MedPlatformUser(HttpUser):
    # Имитация реального пользователя: пауза 1–3 сек между запросами
    wait_time = between(1, 3)

    def on_start(self):
        """Авторизация при старте"""
        resp = self.client.post("/auth/login",
            json={"username": "user1", "password": "doctor123"},
            catch_response=True)
        if resp.status_code == 200:
            self.token = resp.json().get("token", "")
        else:
            self.token = ""

    def auth_headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(6)   # 60% трафика — просмотр ленты
    def view_feed(self):
        self.client.get("/posts", headers=self.auth_headers(), name="/posts [GET]")

    @task(2)   # 20% — создание публикации с модерацией
    def create_post(self):
        text = random.choice(SAMPLE_TEXTS)
        self.client.post("/posts",
            json={"title": text, "body": text + " Прошу совета.", "tags": ["вопрос"]},
            headers=self.auth_headers(),
            name="/posts [POST]")

    @task(2)   # 20% — получение рекомендаций
    def get_recommendations(self):
        self.client.get("/recommendations/user1",
            headers=self.auth_headers(),
            name="/recommendations [GET]")

    @task(1)   # доп. — проверка модерации
    def check_moderation(self):
        text = random.choice(SAMPLE_TEXTS)
        self.client.post("/moderation/check",
            json={"text": text},
            headers=self.auth_headers(),
            name="/moderation/check [POST]")
