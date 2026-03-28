"""
Тесты прототипа МедПлатформа
pytest tests/test_services.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from services.moderation_service import ContentModerator, TFIDFVectorizer, LogisticRegressionSGD
from services.recommendation_service import (
    HybridRecommender, ContentBasedFilter, SVDRecommender, generate_demo_data
)
from services.load_testing import LoadTester, AnomalyDetector


# ─── Модерация ────────────────────────────────────────────────────────────────
class TestTFIDFVectorizer:
    def test_fit_builds_vocabulary(self):
        v = TFIDFVectorizer(max_features=50)
        corpus = ["лечение диабета антибиотиками", "врач пациент больница"]
        v.fit(corpus)
        assert len(v.vocabulary_) > 0
        assert v._fitted

    def test_transform_returns_sparse_vectors(self):
        v = TFIDFVectorizer(max_features=50)
        corpus = ["лечение диабета", "пациент врач"]
        v.fit(corpus)
        vecs = v.transform(["лечение"])
        assert isinstance(vecs, list)
        assert isinstance(vecs[0], dict)

    def test_idf_higher_for_rare_terms(self):
        v = TFIDFVectorizer(max_features=100)
        corpus = ["антибиотик антибиотик антибиотик", "редкий"] * 5 + ["редкий"]
        v.fit(corpus)
        # редкий термин должен иметь более высокий IDF
        idf_common = v.idf_.get("антибиотик", 0)
        idf_rare   = v.idf_.get("редкий", 0)
        assert idf_rare >= idf_common


class TestLogisticRegression:
    def test_fit_predict(self):
        clf = LogisticRegressionSGD(lr=0.1, epochs=10)
        X = [{0: 1.0, 1: 0.5}, {2: 1.0, 3: 0.8}, {0: 0.9, 1: 0.4}, {2: 0.8}]
        y = [0, 1, 0, 1]
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert len(proba) == 4
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_sigmoid_bounds(self):
        clf = LogisticRegressionSGD()
        assert 0.0 < clf._sigmoid(0)   < 1.0
        assert 0.0 < clf._sigmoid(100) < 1.0
        assert 0.0 < clf._sigmoid(-100)< 1.0
        assert clf._sigmoid(100) > 0.99
        assert clf._sigmoid(-100) < 0.01


class TestContentModerator:
    @pytest.fixture(scope="class")
    def trained_moderator(self):
        m = ContentModerator()
        m.train()
        return m

    def test_approves_medical_content(self, trained_moderator):
        result = trained_moderator.moderate("Как правильно принимать антибиотики при ангине?")
        assert result.label == "approved"
        assert result.confidence > 0.5

    def test_blocks_disinformation(self, trained_moderator):
        result = trained_moderator.moderate(
            "Врачи скрывают это средство! Гарантированное излечение рака за 3 дня!"
        )
        assert result.label == "blocked"

    def test_level1_fast_path(self, trained_moderator):
        result = trained_moderator.moderate(
            "Врач назначил метформин при диабете второго типа"
        )
        assert result.level in (1, 2)

    def test_evaluate_metrics(self, trained_moderator):
        metrics = trained_moderator.evaluate()
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0


# ─── Рекомендации ─────────────────────────────────────────────────────────────
class TestContentBasedFilter:
    @pytest.fixture(scope="class")
    def fitted_cbf(self):
        items, _ = generate_demo_data()
        cbf = ContentBasedFilter()
        cbf.fit(items)
        return cbf, items

    def test_recommend_returns_list(self, fitted_cbf):
        cbf, items = fitted_cbf
        recs = cbf.recommend(items[0]["id"], top_k=5)
        assert isinstance(recs, list)
        assert len(recs) <= 5

    def test_scores_in_valid_range(self, fitted_cbf):
        cbf, items = fitted_cbf
        recs = cbf.recommend(items[0]["id"], top_k=10)
        for _, score in recs:
            assert 0.0 <= score <= 1.001   # float precision

    def test_recommend_for_user(self, fitted_cbf):
        cbf, items = fitted_cbf
        liked = [items[0]["id"], items[1]["id"]]
        recs = cbf.recommend_for_user(liked, top_k=5)
        rec_ids = [r[0] for r in recs]
        # не рекомендует уже просмотренные
        for iid in liked:
            assert iid not in rec_ids


class TestSVDRecommender:
    @pytest.fixture(scope="class")
    def trained_svd(self):
        _, interactions = generate_demo_data()
        svd = SVDRecommender(n_factors=5, epochs=10)
        svd.fit(interactions)
        return svd, interactions

    def test_predict_returns_float(self, trained_svd):
        svd, interactions = trained_svd
        uid = interactions[0]["user_id"]
        iid = interactions[0]["item_id"]
        pred = svd.predict(uid, iid)
        assert isinstance(pred, float)

    def test_recommend_excludes_seen(self, trained_svd):
        svd, interactions = trained_svd
        uid = interactions[0]["user_id"]
        seen = [interactions[0]["item_id"]]
        recs = svd.recommend(uid, seen, top_k=5)
        for iid, _ in recs:
            assert iid not in seen


class TestHybridRecommender:
    @pytest.fixture(scope="class")
    def fitted_hybrid(self):
        items, interactions = generate_demo_data()
        h = HybridRecommender(alpha=0.4)
        h.fit(items, interactions)
        return h, items, interactions

    def test_recommend_top_k(self, fitted_hybrid):
        h, items, interactions = fitted_hybrid
        recs = h.recommend("u1", liked_ids=["p1", "p3"], top_k=5)
        assert len(recs) <= 5
        assert all("score" in r for r in recs)

    def test_scores_normalized(self, fitted_hybrid):
        h, items, interactions = fitted_hybrid
        recs = h.recommend("u1", liked_ids=["p1"], top_k=10)
        for r in recs:
            assert 0.0 <= r["score"] <= 1.001

    def test_sorted_descending(self, fitted_hybrid):
        h, items, interactions = fitted_hybrid
        recs = h.recommend("u1", liked_ids=["p1", "p2"], top_k=8)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)


# ─── Нагрузочное тестирование и аномалии ─────────────────────────────────────
class TestLoadTester:
    def test_run_returns_results(self):
        tester = LoadTester()
        results = tester.run([50, 200])
        assert len(results) == 2
        assert results[0].rps == 50
        assert results[0].mean_ms > 0
        assert 0.0 <= results[0].error_rate <= 100.0

    def test_microservice_faster_than_monolith(self):
        tester = LoadTester()
        comparison = tester.compare_architectures([200])
        c = comparison[0]
        assert c["micro_mean_ms"] < c["mono_mean_ms"]

    def test_error_rate_increases_with_load(self):
        tester = LoadTester()
        results = tester.run([50, 1500])
        assert results[-1].error_rate >= results[0].error_rate


class TestAnomalyDetector:
    @pytest.fixture(scope="class")
    def detector_with_data(self):
        import random
        rng = random.Random(42)
        activity = {f"u{i}": {
            "posts_per_day": max(0, rng.gauss(2, 1)),
            "likes_per_day": max(0, rng.gauss(10, 3)),
            "watch_time": max(0, rng.gauss(30, 10)),
            "reports_received": max(0, rng.gauss(0.1, 0.1)),
        } for i in range(1, 16)}
        activity["bot1"] = {"posts_per_day": 180, "likes_per_day": 500,
                            "watch_time": 0.5, "reports_received": 12}
        return activity

    def test_detects_bots(self, detector_with_data):
        det = AnomalyDetector(threshold_percentile=85)
        anomalous, errors = det.detect_anomalous_users(detector_with_data)
        assert "bot1" in anomalous

    def test_threshold_positive(self, detector_with_data):
        det = AnomalyDetector(threshold_percentile=90)
        det.detect_anomalous_users(detector_with_data)
        assert det.threshold_ > 0

    def test_errors_dict_complete(self, detector_with_data):
        det = AnomalyDetector(threshold_percentile=90)
        _, errors = det.detect_anomalous_users(detector_with_data)
        for uid in detector_with_data:
            assert uid in errors
