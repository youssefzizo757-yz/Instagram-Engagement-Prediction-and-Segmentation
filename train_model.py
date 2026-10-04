"""
Trains the Mixture-of-Experts (RF + HGBR) engagement predictor from the
original notebook and saves everything needed to run inference from raw
user-entered fields in the Streamlit app.

Run once locally:  python train_model.py
Produces: model_artifacts.joblib
"""
import os
import pandas as pd, numpy as np, holidays, warnings, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAMMY_CSV = os.path.join(BASE_DIR, "..", "data", "grammy_posts.csv")
ANALYTIC_CSV = os.path.join(BASE_DIR, "..", "data", "instagram_analytics.csv")
OUTPUT_ARTIFACT = os.path.join(BASE_DIR, "model_artifacts.joblib")

# ---------------------------------------------------------------- load data
df_grammy_raw = pd.read_csv(GRAMMY_CSV, sep=';')
df_analytic_raw = pd.read_csv(ANALYTIC_CSV)

df_grammy_raw['post_time'] = pd.to_datetime(df_grammy_raw['post_time'], format='%Y-%m-%d_%H-%M-%S_UTC')
df_grammy_raw['Engagement'] = df_grammy_raw['likes'] + df_grammy_raw['comments']

df_analytic_raw['post_datetime'] = pd.to_datetime(df_analytic_raw['post_datetime'], format='%Y-%m-%d %H:%M:%S')
df_analytic_raw['Engagement'] = (df_analytic_raw['likes'] + df_analytic_raw['comments']
                                  + df_analytic_raw['shares'] + df_analytic_raw['saves'])


def bucket(h):
    if h in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    elif h in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    else:
        return "normal"


# ---------------------------------------------------------- feature builders
def add_features_grammy(df):
    df = df.copy()
    df['post_date'] = df['post_time'].dt.date
    years = range(df['post_time'].dt.year.min(), df['post_time'].dt.year.max() + 1)
    ch = holidays.US(years=years)
    df['is_holiday'] = df['post_date'].isin(ch)
    df['hour'] = df['post_time'].dt.hour
    df['reach_time_bucket'] = df['hour'].apply(bucket)
    df = df.sort_values(['user', 'post_time'])
    df['user_post_count'] = df.groupby('user').cumcount()
    df['user_median_engagement'] = df.groupby('user')['Engagement'].transform(
        lambda x: x.expanding().median().shift(1)).fillna(0)
    df['hashtag_bucket'] = pd.cut(df['number_hashtags'], bins=[-1, 0, 5, 15, 30],
                                   labels=['none', 'low', 'medium', 'high'])
    df['caption_length_bucket'] = pd.cut(df['length_caption'], bins=[-1, 50, 150, 300, 10000],
                                          labels=['short', 'medium', 'long', 'very_long'])
    df['is_weekend'] = df['publication_weekday'].isin(['Saturday', 'Sunday']).astype(int)
    df.drop(columns=['post_time', 'user', 'post_date', 'hour', 'likes', 'comments',
                      'Unnamed: 0', 'awards', 'day_difference', 'IM_performance'], inplace=True)
    return df


def add_features_analytic(df):
    df = df.copy()
    df['post_date'] = df['post_datetime'].dt.date
    years = range(df['post_datetime'].dt.year.min(), df['post_datetime'].dt.year.max() + 1)
    ch = holidays.US(years=years)
    df['is_holiday'] = df['post_date'].isin(ch)
    df['hour'] = df['post_datetime'].dt.hour
    df['reach_time_bucket'] = df['hour'].apply(bucket)
    df = df.sort_values(['account_id', 'post_datetime'])
    df['user_post_count'] = df.groupby('account_id').cumcount()
    df['user_median_engagement'] = df.groupby('account_id')['Engagement'].transform(
        lambda x: x.expanding().median().shift(1)).fillna(0)
    df['hashtag_bucket'] = pd.cut(df['hashtags_count'], bins=[-1, 0, 5, 15, 30],
                                   labels=['none', 'low', 'medium', 'high'])
    df['caption_length_bucket'] = pd.cut(df['caption_length'], bins=[-1, 50, 150, 300, 10000],
                                          labels=['short', 'medium', 'long', 'very_long'])
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['post_images'] = (df['media_type'] == 'image').astype(np.int16)
    df['carousel'] = (df['media_type'] == 'carousel').astype(np.int16)
    df['video'] = (df['media_type'] == 'reel').astype(np.int16)
    df.drop(columns=['media_type'], inplace=True)
    df['publication_weekday'] = df['day_of_week']
    df['number_hashtags'] = df['hashtags_count']
    df['followers'] = df['follower_count']
    df['length_caption'] = df['caption_length']
    df.drop(columns=['day_of_week', 'hashtags_count', 'follower_count', 'caption_length'], inplace=True)
    df.drop(columns=['post_datetime', 'account_id', 'post_date', 'hour', 'likes', 'comments', 'shares',
                      'saves', 'post_id', 'account_type', 'content_category', 'traffic_source',
                      'has_call_to_action', 'post_hour', 'reach', 'impressions', 'engagement_rate',
                      'followers_gained', 'performance_bucket_label'], inplace=True, errors='ignore')
    return df


df_grammy_fe = add_features_grammy(df_grammy_raw)
df_analytic_fe = add_features_analytic(df_analytic_raw)

common = list(set(df_grammy_fe.columns) & set(df_analytic_fe.columns))
df = pd.concat([df_grammy_fe[common], df_analytic_fe[common]], ignore_index=True)
print("Combined dataset:", df.shape)
print("Feature columns:", common)

catcol = ['video', 'carousel', 'publication_weekday', 'caption_length_bucket',
          'hashtag_bucket', 'reach_time_bucket', 'is_weekend', 'is_holiday']
numcol_all = [c for c in df.columns if c not in catcol]  # includes Engagement

# ---------------------------------------------------------------- train/test split (for reported metrics)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

log_cols = [c for c in numcol_all if c != 'Engagement']


def preprocess(frame, cat_categories=None, fit_categories=False):
    frame = frame.copy()
    for col in log_cols:
        frame['log' + col] = np.log1p(frame[col])
        frame.drop(columns=col, inplace=True)
    if fit_categories:
        cat_categories = {c: sorted(frame[c].astype(str).unique().tolist()) for c in catcol}
    for c in catcol:
        frame[c] = pd.Categorical(frame[c].astype(str), categories=cat_categories[c])
    frame = pd.get_dummies(frame, columns=catcol, prefix=catcol, drop_first=True)
    cast_cols = [c for c in frame.columns if c != 'Engagement']
    frame[cast_cols] = frame[cast_cols].astype(float)
    return frame, cat_categories


train_processed, cat_categories = preprocess(train_df, fit_categories=True)
test_processed, _ = preprocess(test_df, cat_categories=cat_categories)
test_processed = test_processed.reindex(columns=train_processed.columns, fill_value=0.0)

X_train = train_processed.drop('Engagement', axis=1)
y_train_raw = train_processed['Engagement']
X_test = test_processed.drop('Engagement', axis=1)
y_test_raw = test_processed['Engagement']

feature_columns = X_train.columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------- KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
train_cluster = kmeans.fit_predict(X_train_scaled)
test_cluster = kmeans.predict(X_test_scaled)

# ------------------------------------------------------------ mixed experts
experts_mixed = {}
for c in range(3):
    mask = train_cluster == c
    Xc = X_train[mask]
    yc = y_train_raw[mask]
    if c == 0:
        m = RandomForestRegressor(n_estimators=300, max_depth=8 if mask.sum() > 1000 else 6,
                                   random_state=42, n_jobs=-1)
    else:
        m = HistGradientBoostingRegressor(max_depth=8 if mask.sum() > 1000 else 6,
                                           learning_rate=0.08, max_iter=300, random_state=42)
    m.fit(Xc, yc)
    experts_mixed[c] = m
    print(f"cluster {c} ({m.__class__.__name__}) n={mask.sum()} train R2 {r2_score(yc, m.predict(Xc)):.4f}")

pred_routed = np.zeros(len(X_test))
for c in range(3):
    mask = test_cluster == c
    if mask.sum() > 0:
        pred_routed[mask] = experts_mixed[c].predict(X_test[mask])

r2 = r2_score(y_test_raw, pred_routed)
rmse = np.sqrt(mean_squared_error(y_test_raw, pred_routed))
mae = mean_absolute_error(y_test_raw, pred_routed)
print(f"MoE-mixed held-out test: R2 {r2:.4f}  RMSE {rmse:.0f}  MAE {mae:.0f}")

# ------------------------------------------------------- refit on ALL data for deployment
full_processed, _ = preprocess(df, cat_categories=cat_categories)
X_full = full_processed.drop('Engagement', axis=1).reindex(columns=feature_columns, fill_value=0.0)
y_full = full_processed['Engagement']
X_full_scaled = scaler.fit_transform(X_full)

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
full_cluster = kmeans_final.fit_predict(X_full_scaled)

experts_final = {}
for c in range(3):
    mask = full_cluster == c
    Xc = X_full[mask]
    yc = y_full[mask]
    if c == 0:
        m = RandomForestRegressor(n_estimators=300, max_depth=8 if mask.sum() > 1000 else 6,
                                   random_state=42, n_jobs=-1)
    else:
        m = HistGradientBoostingRegressor(max_depth=8 if mask.sum() > 1000 else 6,
                                           learning_rate=0.08, max_iter=300, random_state=42)
    m.fit(Xc, yc)
    experts_final[c] = m

artifacts = {
    "scaler": scaler,
    "kmeans": kmeans_final,
    "experts": experts_final,
    "feature_columns": feature_columns,
    "log_cols": log_cols,
    "cat_categories": cat_categories,
    "catcol": catcol,
    "holdout_metrics": {"r2": r2, "rmse": rmse, "mae": mae},
}
joblib.dump(artifacts, OUTPUT_ARTIFACT)
print("Saved", OUTPUT_ARTIFACT)
