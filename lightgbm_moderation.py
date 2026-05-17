# lightgbm_moderation.py
# LightGBM классификатор медицинского контента
# Автор: Аль-Раве Мустафа · РГСУ · спец. 2.3.5

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json, time, joblib

# ── 1. Обучающие данные ───────────────────────────────────────────
TRAIN_DATA = [
    # (текст, метка)  0=approved, 1=suspicious, 2=blocked
    ("Врач назначил метформин 500мг при диабете 2 типа. Принимаю после еды.", 0),
    ("После инфаркта важна кардиореабилитация. Консультация кардиолога обязательна.", 0),
    ("Реабилитация после инсульта включает физиотерапию и ЛФК по протоколу.", 0),
    ("Вакцинация против гриппа согласно рекомендациям ВОЗ снижает риск осложнений.", 0),
    ("Гемоглобин 110 г/л — анемия лёгкой степени, назначены препараты железа.", 0),
    ("МРТ головного мозга показало норму. Невролог направил на ЭЭГ.", 0),
    ("Антибиотикотерапия пневмонии: амоксициллин по назначению пульмонолога.", 0),
    ("Гастроскопия выявила гастрит. Гастроэнтеролог назначил ингибиторы протонной помпы.", 0),
    ("Холестерин 6.2 ммоль/л. Кардиолог назначил статины и диету.", 0),
    ("Перед применением проконсультируйтесь с врачом. Информация ознакомительная.", 0),

    ("Биорезонансная терапия помогает лучше антибиотиков, читал на форуме.", 1),
    ("Сам поставил себе диагноз диабет по симптомам из интернета.", 1),
    ("Гомеопатия вылечивает лучше традиционной медицины по отзывам.", 1),
    ("Врачи не предупреждают о побочных эффектах препаратов.", 1),
    ("Отказался от химиотерапии, перехожу на народные средства.", 1),
    ("БАД заменит лекарства, принимаю вместо назначенных таблеток.", 1),
    ("Сосед вылечился травами вместо врача, мне тоже советует.", 1),
    ("Превышение дозировки не страшно, прочитал в блоге.", 1),

    ("Врачи скрывают правду! Фармацевтическая мафия обманывает пациентов!", 2),
    ("Чудо-средство навсегда вылечит рак за 3 дня без врачей! 100% гарантия!", 2),
    ("Минздрав замалчивает! Вакцины с чипами 5G убивают детей!", 2),
    ("Купи наш БАД без рецепта! Доставка по России. Промокод HEALTH30%!", 2),
    ("Больницы заражают! Скорую не вызывай, лечись дома природными методами!", 2),
    ("Пиши в личку — расскажу секретный метод лечения онкологии пока не удалили!", 2),
    ("Без операции и таблеток! Природные средства победят онкологию навсегда!", 2),
    ("ВОЗ лжёт! Правительство запрещает истинные методы лечения!", 2),
    ("Telegram @healthsecret — покупай лекарства без рецепта со скидкой 50%!", 2),
    ("Навсегда избавим от диабета за 7 дней! Звони сейчас пока не поздно!", 2),
]

texts = [d[0] for d in TRAIN_DATA]
labels = [d[1] for d in TRAIN_DATA]

# ── 2. TF-IDF векторизация ────────────────────────────────────────
print("🔧 Векторизация текста (TF-IDF)...")
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 4),
    max_features=5000,
    min_df=1
)
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ── 3. Обучение LightGBM ──────────────────────────────────────────
print("🚀 Обучение LightGBM...")
t0 = time.time()

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_test,  label=y_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
)

elapsed = time.time() - t0
print(f"✅ Обучение завершено за {elapsed:.2f}с")

# ── 4. Оценка модели ──────────────────────────────────────────────
print("\n📊 Оценка на тестовой выборке:")
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print("\n" + classification_report(
    y_test, y_pred,
    target_names=['approved', 'suspicious', 'blocked']
))

# ── 5. Важность признаков ─────────────────────────────────────────
importance = model.feature_importance(importance_type='gain')
feat_names  = vectorizer.get_feature_names_out()
top_idx     = np.argsort(importance)[::-1][:10]
print("🔑 Топ-10 важных n-gram признаков:")
for i in top_idx:
    print(f"   '{feat_names[i]}': {importance[i]:.2f}")

# ── 6. Тест на новых примерах ─────────────────────────────────────
LABELS = {0: '✅ approved', 1: '⚠️ suspicious', 2: '🚫 blocked'}
test_cases = [
    "Врач назначил инсулин при диабете 1 типа, принимаю согласно протоколу эндокринологии",
    "Врачи скрывают лечение! Большая фарма обманывает! Вылечись дома без таблеток!",
    "Биорезонанс лучше антибиотиков — читал на форуме, сам поставил диагноз",
    "Купи БАД без рецепта, доставка по России, промокод HEALTH",
]

print("\n🔬 Тест на новых примерах:")
for txt in test_cases:
    vec = vectorizer.transform([txt]).toarray()
    proba = model.predict(vec)[0]
    pred  = np.argmax(proba)
    print(f"   {LABELS[pred]} (conf: {proba[pred]*100:.1f}%) — {txt[:60]}...")

# ── 7. Сохранение модели ──────────────────────────────────────────
model.save_model('lgbm_moderation.txt')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

results = {
    'accuracy':
