"""
Zomato IPO Sentiment Analysis
Feature Engineering + Model Building Pipeline
=============================================
Outputs (saved to /home/claude/outputs/):
  - featured_zomato.csv
  - featured_ipo_tweets.csv
  - featured_yt_comments.csv
  - featured_yt_videos.csv
  - model_results_summary.csv
  - confusion_matrix_*.png
  - roc_curve.png
  - feature_importance.png
  - tfidf_vectorizer.pkl
  - best_model.pkl
  - label_encoder.pkl
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.utils import resample
import pickle
warnings.filterwarnings('ignore')

OUT = '/home/claude/outputs'
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("ZOMATO IPO SENTIMENT ANALYSIS PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1] Loading datasets...")
BASE = '/mnt/user-data/uploads'
zomato    = pd.read_csv(f'{BASE}/zomato_preprocessed.csv')
ipo       = pd.read_csv(f'{BASE}/zomato_ipo_preprocessed.csv')
yt_comm   = pd.read_csv(f'{BASE}/youtube_comments_preprocessed.csv')
yt_vid    = pd.read_csv(f'{BASE}/youtube_zomato_preprocessed.csv')

print(f"  zomato news    : {zomato.shape}")
print(f"  IPO tweets     : {ipo.shape}")
print(f"  YT comments    : {yt_comm.shape}")
print(f"  YT videos      : {yt_vid.shape}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING — ZOMATO NEWS
# ─────────────────────────────────────────────
print("\n[2] Feature engineering: Zomato news...")

z = zomato.copy()

# Text-based features
z['avg_word_len'] = z['text_clean'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) and str(x).strip() else 0
)
z['unique_word_ratio'] = z['text_clean'].apply(
    lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1)
)
z['exclamation_count'] = z['text'].apply(lambda x: str(x).count('!'))
z['question_count']    = z['text'].apply(lambda x: str(x).count('?'))
z['upper_word_count']  = z['text'].apply(
    lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
)

# Finance keywords
bullish_kw = ['profit', 'growth', 'revenue', 'strong', 'positive', 'rally',
              'surge', 'gain', 'buy', 'invest', 'bull', 'rise', 'up', 'high',
              'record', 'beat', 'outperform', 'expand', 'opportunity']
bearish_kw = ['loss', 'decline', 'fall', 'risk', 'concern', 'crash', 'sell',
              'down', 'weak', 'negative', 'drop', 'underperform', 'overvalued',
              'expensive', 'burn', 'unprofitable', 'worry']

z['bullish_keyword_count'] = z['text_clean'].apply(
    lambda x: sum(1 for w in str(x).lower().split() if w in bullish_kw)
)
z['bearish_keyword_count'] = z['text_clean'].apply(
    lambda x: sum(1 for w in str(x).lower().split() if w in bearish_kw)
)
z['sentiment_keyword_ratio'] = (
    (z['bullish_keyword_count'] - z['bearish_keyword_count']) /
    (z['bullish_keyword_count'] + z['bearish_keyword_count'] + 1)
)

# Source dummies
z = pd.get_dummies(z, columns=['source'], prefix='src', drop_first=False)

# Target encoding (only labeled rows)
le_z = LabelEncoder()
labeled = z[z['sentiment'] != 'unknown'].copy()
labeled['target'] = le_z.fit_transform(labeled['sentiment'])  # Neg=0,Neu=1,Pos=2

z.to_csv(f'{OUT}/featured_zomato.csv', index=False)
labeled.to_csv(f'{OUT}/featured_zomato_labeled.csv', index=False)
print(f"  Features added. Labeled rows: {len(labeled)}")
print(f"  Class distribution: {labeled['sentiment'].value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING — IPO TWEETS
# ─────────────────────────────────────────────
print("\n[3] Feature engineering: IPO tweets...")

t = ipo.copy()
t['tweet_clean'] = t['cleaned_tweet'].fillna(t['tweet'])

# Text features
t['avg_word_len'] = t['tweet_clean'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).strip() else 0
)
t['unique_word_ratio'] = t['tweet_clean'].apply(
    lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1)
)
t['exclamation_count'] = t['tweet'].apply(lambda x: str(x).count('!'))
t['question_count']    = t['tweet'].apply(lambda x: str(x).count('?'))
t['url_count']   = t['tweet'].apply(lambda x: str(x).count('http'))
t['mention_count'] = t['tweet'].apply(lambda x: str(x).count('@'))
t['hashtag_count'] = t['tweet'].apply(lambda x: str(x).count('#'))

# Finance keywords
t['bullish_keyword_count'] = t['tweet_clean'].apply(
    lambda x: sum(1 for w in str(x).lower().split() if w in bullish_kw)
)
t['bearish_keyword_count'] = t['tweet_clean'].apply(
    lambda x: sum(1 for w in str(x).lower().split() if w in bearish_kw)
)
t['sentiment_keyword_ratio'] = (
    (t['bullish_keyword_count'] - t['bearish_keyword_count']) /
    (t['bullish_keyword_count'] + t['bearish_keyword_count'] + 1)
)

# Engagement normalisation
t['log_likes']    = np.log1p(t['likes_count'])
t['log_retweets'] = np.log1p(t['retweets_count'])
t['log_replies']  = np.log1p(t['replies_count'])
t['viral_score']  = t['log_likes'] + 2 * t['log_retweets'] + t['log_replies']

# Day-of-week encoding
dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
           'Friday':4,'Saturday':5,'Sunday':6}
t['dow_num'] = t['day_of_week'].map(dow_map).fillna(0).astype(int)
t['is_weekend'] = (t['dow_num'] >= 5).astype(int)

# Hour buckets
t['hour_bucket'] = pd.cut(t['hour'], bins=[0,6,12,18,24], labels=[0,1,2,3], right=False).astype(int)

t.to_csv(f'{OUT}/featured_ipo_tweets.csv', index=False)
print(f"  Features added. Rows: {len(t)}")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING — YT COMMENTS
# ─────────────────────────────────────────────
print("\n[4] Feature engineering: YouTube comments...")

c = yt_comm.copy()

c['avg_word_len'] = c['text_clean'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).strip() else 0
)
c['unique_word_ratio'] = c['text_clean'].apply(
    lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1)
)
c['exclamation_count'] = c['text'].apply(lambda x: str(x).count('!'))
c['question_count']    = c['text'].apply(lambda x: str(x).count('?'))

c['bullish_keyword_count'] = c['text_clean'].apply(
    lambda x: sum(1 for w in str(x).lower().split() if w in bullish_kw)
)
c['bearish_keyword_count'] = c['text_clean'].apply(
    lambda x: sum(1 for w in str(x).lower().split() if w in bearish_kw)
)
c['sentiment_keyword_ratio'] = (
    (c['bullish_keyword_count'] - c['bearish_keyword_count']) /
    (c['bullish_keyword_count'] + c['bearish_keyword_count'] + 1)
)
c['is_weekend']  = (c['day_of_week'] >= 5).astype(int)
c['hour_bucket'] = pd.cut(c['hour'], bins=[0,6,12,18,24], labels=[0,1,2,3], right=False).astype(int)

# Channel frequency tier
vals = c['channel_freq']; c['channel_tier'] = pd.cut(vals, bins=[-1, vals.quantile(0.33), vals.quantile(0.67), float('inf')], labels=['low','mid','high'], duplicates='drop')
c = pd.get_dummies(c, columns=['channel_tier'], drop_first=False)

# Engagement ratio
c['like_per_reply'] = c['like_count'] / (c['reply_count'] + 1)

c.to_csv(f'{OUT}/featured_yt_comments.csv', index=False)
print(f"  Features added. Rows: {len(c)}")

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING — YT VIDEOS
# ─────────────────────────────────────────────
print("\n[5] Feature engineering: YouTube videos...")

v = yt_vid.copy()

v['avg_word_len'] = v['title_clean'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).strip() else 0
)
v['unique_word_ratio'] = v['title_clean'].apply(
    lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1)
)
v['title_has_ipo']      = v['title'].str.lower().str.contains('ipo').astype(int)
v['title_has_invest']   = v['title'].str.lower().str.contains('invest').astype(int)
v['title_has_price']    = v['title'].str.lower().str.contains('price|target').astype(int)
v['title_has_positive'] = v['title'].str.lower().str.contains('buy|subscribe|growth').astype(int)
v['title_has_negative'] = v['title'].str.lower().str.contains('avoid|sell|loss|crash').astype(int)
v['is_weekend'] = v['day_of_week'].isin(['Saturday','Sunday']).astype(int)
v = pd.get_dummies(v, columns=['search_query_cat'], prefix='sq', drop_first=False)

v.to_csv(f'{OUT}/featured_yt_videos.csv', index=False)
print(f"  Features added. Rows: {len(v)}")

# ─────────────────────────────────────────────
# 6. MODEL BUILDING — ZOMATO NEWS SENTIMENT
# ─────────────────────────────────────────────
print("\n[6] Model building on Zomato news (labeled subset)...")

df_model = labeled.copy()

# ── Handle class imbalance via oversampling ──
classes = df_model['target'].unique()
max_count = df_model['target'].value_counts().max()
balanced_parts = []
for cls in classes:
    subset = df_model[df_model['target'] == cls]
    upsampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
    balanced_parts.append(upsampled)
df_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  After balancing: {df_balanced['target'].value_counts().to_dict()}")

X_text = df_balanced['text_clean'].fillna('')
y      = df_balanced['target']

# Numeric features to append
num_cols = ['word_count', 'clean_word_count', 'char_length',
            'avg_word_len', 'unique_word_ratio',
            'bullish_keyword_count', 'bearish_keyword_count',
            'sentiment_keyword_ratio', 'exclamation_count', 'question_count']
# Only keep cols that actually exist
num_cols = [c for c in num_cols if c in df_balanced.columns]

X_num = df_balanced[num_cols].fillna(0).values
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

X_train_txt, X_test_txt, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, X_num_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=1,
                         sublinear_tf=True, strip_accents='unicode')
X_train_tfidf = tfidf.fit_transform(X_train_txt)
X_test_tfidf  = tfidf.transform(X_test_txt)

# Combine TF-IDF + numeric
from scipy.sparse import hstack, csr_matrix
X_train = hstack([X_train_tfidf, csr_matrix(X_train_num)])
X_test  = hstack([X_test_tfidf,  csr_matrix(X_test_num)])

# ── Train multiple models ──
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
    'Naive Bayes':         MultinomialNB(),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'Linear SVC':          LinearSVC(max_iter=2000, C=1.0, random_state=42, class_weight='balanced'),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_acc = 0
best_model = None
best_name = ''

for name, model in models.items():
    try:
        # NB needs non-negative input — use only TF-IDF
        if name == 'Naive Bayes':
            cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='accuracy')
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
        else:
            from sklearn.preprocessing import MaxAbsScaler
            X_tr_dense = X_train.toarray()
            X_te_dense = X_test.toarray()
            cv_scores = cross_val_score(model, X_tr_dense, y_train, cv=cv, scoring='accuracy')
            model.fit(X_tr_dense, y_train)
            y_pred = model.predict(X_te_dense)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=le_z.classes_, output_dict=True)

        results.append({
            'Model': name,
            'CV_Mean_Acc': round(cv_scores.mean(), 4),
            'CV_Std':      round(cv_scores.std(), 4),
            'Test_Acc':    round(acc, 4),
            'Precision_macro': round(report['macro avg']['precision'], 4),
            'Recall_macro':    round(report['macro avg']['recall'], 4),
            'F1_macro':        round(report['macro avg']['f1-score'], 4),
        })
        print(f"  {name:25s} | CV={cv_scores.mean():.3f}±{cv_scores.std():.3f} | Test={acc:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le_z.classes_, yticklabels=le_z.classes_, ax=ax)
        ax.set_title(f'Confusion Matrix — {name}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.tight_layout()
        safe_name = name.replace(' ', '_')
        plt.savefig(f'{OUT}/confusion_matrix_{safe_name}.png', dpi=120)
        plt.close()

        if acc > best_acc:
            best_acc   = acc
            best_model = model
            best_name  = name

    except Exception as e:
        print(f"  {name}: FAILED — {e}")

# ─────────────────────────────────────────────
# 7. ROC CURVE (Logistic Regression, OvR)
# ─────────────────────────────────────────────
print("\n[7] ROC curve...")
try:
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced')
    lr.fit(X_train.toarray(), y_train)
    y_prob = lr.predict_proba(X_test.toarray())

    fig, ax = plt.subplots(figsize=(7,5))
    for i, cls_name in enumerate(le_z.classes_):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
        auc_val = roc_auc_score((y_test == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, lw=2, label=f'{cls_name} (AUC={auc_val:.2f})')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (One-vs-Rest) — Logistic Regression')
    ax.legend(); plt.tight_layout()
    plt.savefig(f'{OUT}/roc_curve.png', dpi=120)
    plt.close()
    print("  Saved roc_curve.png")
except Exception as e:
    print(f"  ROC failed: {e}")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
print("\n[8] Feature importance...")
try:
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train.toarray(), y_train)

    tfidf_names = tfidf.get_feature_names_out().tolist()
    all_feat_names = tfidf_names + num_cols
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh([all_feat_names[i] for i in top_idx[::-1]],
            importances[top_idx[::-1]], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/feature_importance.png', dpi=120)
    plt.close()
    print("  Saved feature_importance.png")
except Exception as e:
    print(f"  Feature importance failed: {e}")

# ─────────────────────────────────────────────
# 9. SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("\n[9] Saving model artifacts...")
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUT}/model_results_summary.csv', index=False)
print("\n  Model Results:")
print(results_df.to_string(index=False))

with open(f'{OUT}/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open(f'{OUT}/label_encoder.pkl', 'wb') as f:
    pickle.dump(le_z, f)
if best_model:
    with open(f'{OUT}/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n  Best model: {best_name} (Test Acc = {best_acc:.4f})")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE. All outputs saved to:", OUT)
print("=" * 60)
