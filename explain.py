import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = 'SHAP_Analysis'
os.makedirs(output_dir, exist_ok=True)

TARGET = 'high_inefficiency'
LEAKAGE_FEATURES = [
    'renewable_penetration',
    'renewable_share',
    'total_renewable_generation',
]

data_dir = 'output'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
print(f"Original train features matrix: {train_df.shape}")
original_train = train_df.copy()

def compute_dynamic_thresholds(train):
    df = train.copy()
    df['hour_block'] = (df['hour'] // 4).astype(int)
    df['weather_regime'] = pd.cut(df['clouds_all'], bins=[-1, 25, 65, 101], labels=['clear', 'partial', 'overcast'])
    
    proxy = 'generation wind onshore'
    
    tbl = df.groupby(['season', 'hour_block', 'weather_regime'], observed=True).agg(
        n=(proxy, 'count'),
        proxy_mean=(proxy, 'mean'),
        proxy_std=(proxy, 'std'),
        ineff_rate=(TARGET, 'mean'),
        wind_mean=('wind_speed', 'mean'),
        temp_mean=('temp_celsius', 'mean'),
        solar_mean=('generation solar', 'mean'),
        fossil_mean=('generation fossil gas', 'mean'),
        load_mean=('total load actual', 'mean'),
    ).reset_index()
    
    tbl['proxy_std'] = tbl['proxy_std'].fillna(0)
    tbl['thr_high'] = tbl['proxy_mean'] + 1.5 * tbl['proxy_std']
    tbl['thr_low'] = (tbl['proxy_mean'] - 1.5 * tbl['proxy_std']).clip(lower=0)
    tbl['thr_range'] = tbl['thr_high'] - tbl['thr_low']
    
    return tbl

thr_table = compute_dynamic_thresholds(train_df)

def attach_threshold_features(df, tbl, proxy='generation wind onshore'):
    d = df.copy()
    d['hour_block'] = (d['hour'] // 4).astype(int)
    d['weather_regime'] = pd.cut(d['clouds_all'], bins=[-1, 25, 65, 101],labels=['clear', 'partial', 'overcast'])
    merge_cols = ['season', 'hour_block', 'weather_regime', 'proxy_mean', 'proxy_std', 'thr_high', 'thr_low', 'thr_range', 'ineff_rate']
    d = d.merge(tbl[merge_cols], on=['season', 'hour_block', 'weather_regime'], how='left')

    d['wind_above_high'] = (d[proxy] - d['thr_high']).clip(lower=0)
    d['wind_below_low'] = (d['thr_low'] - d[proxy]).clip(lower=0)
    d['wind_deviation'] = d[proxy] - d['proxy_mean']
    d['wind_zscore'] = d['wind_deviation'] / (d['proxy_std'] + 1e-6)
    d['exceeds_threshold'] = (d[proxy] > d['thr_high']).astype(int)
    d['below_threshold'] = (d[proxy] < d['thr_low']).astype(int)

    new_cols = ['proxy_mean', 'proxy_std', 'thr_high', 'thr_low', 'thr_range', 'ineff_rate',
                'wind_above_high', 'wind_below_low', 'wind_deviation',
                'wind_zscore', 'exceeds_threshold', 'below_threshold']
    d[new_cols] = d[new_cols].fillna(0)
    d.drop(columns=['hour_block', 'weather_regime'], inplace=True, errors='ignore')
    return d

train_df = attach_threshold_features(train_df, thr_table)

train_df = train_df.drop(columns=[f for f in LEAKAGE_FEATURES if f in train_df.columns])
if 'year' in train_df.columns:
    train_df = train_df.drop(columns=['year'])

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

print(f"Features for SHAP: {X_train.shape[1]} columns")

model = xgb.XGBClassifier()
model.load_model('output/model_output/xgb_model_fixed.json')
print("XGBoost Model loaded")

model_features = model.get_booster().feature_names

if len(model_features) == len(X_train.columns):
    X_train = X_train[model_features]
else:
    print(f"Count mismatch: Model: {len(model_features)}, Data: {X_train.shape[1]}")
    missing = set(model_features) - set(X_train.columns)
    extra = set(X_train.columns) - set(model_features)
    if missing:
        print(f"Missing features: {list(missing)[:5]}...")
    if extra:
        print(f"Extra features: {list(extra)[:5]}...")

print(f"Final feature matrix: {X_train.shape[0]} rows, {X_train.shape[1]} columns")

print(f"Computing SHAP values for {X_train.shape[0]} rows with {X_train.shape[1]} features.")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
print("SHAP values computed")

np.save(os.path.join(output_dir, 'ShapValues.npy'), shap_values)

plt.figure(figsize=(16, 12))
shap.summary_plot(shap_values, X_train, show=False,feature_names=X_train.columns.tolist(), max_display=15)
plt.title('SHAP Feature Importance Summary', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ShapSummary.png'), dpi=300, bbox_inches='tight')
plt.close()

feature_importance = pd.DataFrame({
    'feature': X_train.columns.tolist(),
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

print("\nTop 15 features by SHAP importance:")
for i, row in feature_importance.head(15).iterrows():
    print(f" {row['feature']:45s} {row['shap_importance']:.6f}")


top_n = 15
top_features = feature_importance.head(top_n)

plt.figure(figsize=(14, 10))
plt.barh(range(top_n), top_features['shap_importance'].values[::-1], color='steelblue')
plt.yticks(range(top_n), top_features['feature'].values[::-1])
plt.xlabel('SHAP feature importance score')
plt.title(f'Top {top_n} Features as per SHAP Values')
plt.tight_layout() 
plt.savefig(os.path.join(output_dir, 'TopFeatures.png'), dpi=300, bbox_inches='tight')
plt.close()


top_3_features = feature_importance.head(3)['feature'].tolist()

for feature in top_3_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X_train, feature_names=X_train.columns.tolist(), show=False)
    plt.title(f'SHAP Dependence Plot: {feature}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'SHAP_{feature}.png'), dpi=300, bbox_inches='tight')
    plt.close()


y_pred_full = model.predict(X_train)


if 'hour' in original_train.columns:
    df_hourly = original_train.copy()
    df_hourly['predicted'] = y_pred_full
    df_hourly['actual'] = y_train.values
    
    hourly_actual = df_hourly.groupby('hour')[TARGET].agg(['mean', 'std'])
    hourly_pred = df_hourly.groupby('hour')['predicted'].agg(['mean', 'std'])
    hourly_error = hourly_actual['mean'] - hourly_pred['mean']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].plot(hourly_actual.index, hourly_actual['mean'], marker='o', linewidth=2, markersize=8, label='Actual', color='darkblue')
    axes[0].plot(hourly_pred.index, hourly_pred['mean'], marker='s', linewidth=2, markersize=8, label='Predicted', color='darkred')
    axes[0].set_xlabel('Hour of Day', fontsize=12)
    axes[0].set_ylabel('Inefficiency Rate', fontsize=12)
    axes[0].set_title('Hourly Pattern: Actual vs Predicted', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24))
    
    colors = ['red' if x < 0 else 'green' for x in hourly_error]
    axes[1].bar(hourly_error.index, hourly_error, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Hour of Day', fontsize=12)
    axes[1].set_ylabel('Error (Actual - Predicted)', fontsize=12)
    axes[1].set_title('Prediction Error by Hour', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(0, 24))
    
    axes[2].plot(hourly_actual.index, hourly_actual['std'], marker='^', linewidth=2, markersize=8, label='Actual Std', color='purple')
    axes[2].plot(hourly_pred.index, hourly_pred['std'], marker='v', linewidth=2, markersize=8, label='Predicted Std', color='orange')
    axes[2].set_xlabel('Hour of Day', fontsize=12)
    axes[2].set_ylabel('Standard Deviation', fontsize=12)
    axes[2].set_title('Hourly Variability', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(range(0, 24))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'HourlyAnalysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
else:
    print("No hour colmn found")


if 'season' in original_train.columns:
    df_seasonal = original_train.copy()
    df_seasonal['predicted'] = y_pred_full
    
    season_names = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
    df_seasonal['season_name'] = df_seasonal['season'].map(season_names)
    
    seasonal_actual = df_seasonal.groupby('season_name')[TARGET].agg(['mean', 'std'])
    seasonal_pred = df_seasonal.groupby('season_name')['predicted'].agg(['mean', 'std'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    seasons_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    x_pos = np.arange(len(seasons_order))
    width = 0.35
    
    actual_means = [seasonal_actual.loc[s, 'mean'] if s in seasonal_actual.index else 0 for s in seasons_order]
    pred_means = [seasonal_pred.loc[s, 'mean'] if s in seasonal_pred.index else 0 for s in seasons_order]
    
    axes[0].bar(x_pos - width/2, actual_means, width, label='Actual', color='steelblue', alpha=0.8)
    axes[0].bar(x_pos + width/2, pred_means, width, label='Predicted', color='coral', alpha=0.8)
    axes[0].set_xlabel('Season', fontsize=12)
    axes[0].set_ylabel('Inefficiency Rate', fontsize=12)
    axes[0].set_title('Seasonal Pattern: Actual vs Predicted', fontsize=14)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(seasons_order)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if 'month' in df_seasonal.columns:
        monthly_actual = df_seasonal.groupby('month')[TARGET].mean()
        monthly_pred = df_seasonal.groupby('month')['predicted'].mean()
        
        axes[1].plot(monthly_actual.index, monthly_actual.values, marker='o', linewidth=2, markersize=8, label='Actual', color='darkgreen')
        axes[1].plot(monthly_pred.index, monthly_pred.values, marker='s', linewidth=2, markersize=8, label='Predicted', color='darkorange')
        axes[1].set_xlabel('Month', fontsize=12)
        axes[1].set_ylabel('Inefficiency Rate', fontsize=12)
        axes[1].set_title('Monthly Trend', fontsize=14)
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    seasonal_error = pd.Series(actual_means) - pd.Series(pred_means)
    error_colors = ['red' if x < 0 else 'green' for x in seasonal_error]
    axes[2].bar(seasons_order, seasonal_error, color=error_colors, alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[2].set_xlabel('Season', fontsize=12)
    axes[2].set_ylabel('Error (Actual - Predicted)', fontsize=12)
    axes[2].set_title('Prediction Error by Season', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SeasonalAnalysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
else:
    print("Season column not found")