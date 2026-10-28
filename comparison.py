import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

TARGET   = 'high_inefficiency'
SEQ_LEN  = 24
WITH_THR = 'WITH threshold'
NO_THR   = 'WITHOUT threshold'


LEAKAGE = ['renewable_penetration', 'renewable_share', 'total_renewable_generation']
SOURCES = ['generation wind onshore', 'generation wind offshore', 'generation solar']


THR_FEATS = [
    'renewable_proxy', 'proxy_mean', 'proxy_std',
    'thr_high', 'thr_low', 'thr_range', 'ineff_rate',
    'proxy_above_high', 'proxy_below_low',
    'proxy_deviation', 'proxy_zscore',
    'exceeds_threshold', 'below_threshold',
    'load_ratio', 'load_thr_high', 'load_thr_low',
    'load_above_high', 'load_below_low', 'load_zscore',
]

COLORS = dict(
    xgb='#2563EB', rf='#16A34A', lgbm='#D97706',
    lstm='#DC2626', gru='#7C3AED', baseline='#6B7280',
)


def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val   = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test  = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print(f'  train: {train.shape}   val: {val.shape}   test: {test.shape}')
    print(f'  positive rate (train): {train[TARGET].mean():.3f}')
    return train, val, test


def add_proxy(df):
    d = df.copy()
    cols = [c for c in SOURCES if c in d.columns]
    d['renewable_proxy'] = d[cols].clip(lower=0).sum(axis=1)
    return d


def build_threshold_table(train_df):
    df = add_proxy(train_df)
    df['hour_block']  = (df['hour'] // 4).astype(int)
    df['cloud_level'] = pd.cut(
        df['clouds_all'], bins=[-1, 25, 65, 101],
        labels=['clear', 'partial', 'overcast']
    )
    df['proxy_load_ratio'] = df['renewable_proxy'] / df['total load actual'].clip(lower=1)

    tbl = df.groupby(['season', 'hour_block', 'cloud_level'], observed=True).agg(
        proxy_mean      = ('renewable_proxy',  'mean'),
        proxy_std       = ('renewable_proxy',  'std'),
        ineff_rate      = (TARGET,             'mean'),
        load_ratio_mean = ('proxy_load_ratio', 'mean'),
        load_ratio_std  = ('proxy_load_ratio', 'std'),
    ).reset_index()

    tbl['proxy_std']      = tbl['proxy_std'].fillna(0)
    tbl['load_ratio_std'] = tbl['load_ratio_std'].fillna(0)

    tbl['thr_high'] = tbl['proxy_mean'] + 1.5 * tbl['proxy_std']
    tbl['thr_low']  = (tbl['proxy_mean'] - 1.5 * tbl['proxy_std']).clip(lower=0)
    tbl['thr_range'] = tbl['thr_high'] - tbl['thr_low']

    tbl['load_thr_high'] = tbl['load_ratio_mean'] + 1.5 * tbl['load_ratio_std']
    tbl['load_thr_low']  = (tbl['load_ratio_mean'] - 1.5 * tbl['load_ratio_std']).clip(lower=0)

    print(f'  threshold cells: {len(tbl)}   avg ineff rate: {tbl["ineff_rate"].mean():.3f}')
    return tbl


def attach_thresholds(df, tbl):
    d = add_proxy(df)
    d['hour_block']  = (df['hour'] // 4).astype(int)
    d['cloud_level'] = pd.cut(
        d['clouds_all'], bins=[-1, 25, 65, 101],
        labels=['clear', 'partial', 'overcast']
    )
    d['proxy_load_ratio'] = d['renewable_proxy'] / d['total load actual'].clip(lower=1)

    merge_cols = [
        'season', 'hour_block', 'cloud_level',
        'proxy_mean', 'proxy_std', 'thr_high', 'thr_low', 'thr_range', 'ineff_rate',
        'load_ratio_mean', 'load_ratio_std', 'load_thr_high', 'load_thr_low',
    ]
    d = d.merge(tbl[merge_cols], on=['season', 'hour_block', 'cloud_level'], how='left')

    d['proxy_above_high']  = (d['renewable_proxy'] - d['thr_high']).clip(lower=0)
    d['proxy_below_low']   = (d['thr_low'] - d['renewable_proxy']).clip(lower=0)
    d['proxy_deviation']   = d['renewable_proxy'] - d['proxy_mean']
    d['proxy_zscore']      = d['proxy_deviation'] / (d['proxy_std'] + 1e-6)
    d['exceeds_threshold'] = (d['renewable_proxy'] > d['thr_high']).astype(int)
    d['below_threshold']   = (d['renewable_proxy'] < d['thr_low']).astype(int)

    d['load_ratio']      = d['proxy_load_ratio']
    d['load_above_high'] = (d['proxy_load_ratio'] - d['load_thr_high']).clip(lower=0)
    d['load_below_low']  = (d['load_thr_low'] - d['proxy_load_ratio']).clip(lower=0)
    d['load_zscore']     = (
        (d['proxy_load_ratio'] - d['load_ratio_mean']) / (d['load_ratio_std'] + 1e-6)
    )

    thr_only = [f for f in THR_FEATS if f != 'renewable_proxy']
    d[thr_only] = d[thr_only].fillna(0)
    d.drop(columns=['hour_block', 'cloud_level', 'proxy_load_ratio',
                    'load_ratio_mean', 'load_ratio_std'], inplace=True, errors='ignore')
    return d


def build_feature_sets(train_df, val_df, test_df):
 
    base_drop = LEAKAGE + [TARGET, 'year']
    tbl = build_threshold_table(train_df)

    tr = attach_thresholds(train_df, tbl)
    va = attach_thresholds(val_df,   tbl)
    te = attach_thresholds(test_df,  tbl)

    def split_xy(df, extra_drop=None):
        drop = base_drop + (extra_drop or [])
        cols = [c for c in df.columns if c not in drop]
        return df[cols].values, df[TARGET].values, cols

    X_tr_w,  y_tr, feat_w  = split_xy(tr)
    X_va_w,  y_va, _       = split_xy(va)
    X_te_w,  y_te, _       = split_xy(te)

    X_tr_wo, _,    feat_wo = split_xy(tr, extra_drop=THR_FEATS)
    X_va_wo, _,    _       = split_xy(va, extra_drop=THR_FEATS)
    X_te_wo, _,    _       = split_xy(te, extra_drop=THR_FEATS)


    any_leak = False
    for f in THR_FEATS:
        if f in tr.columns:
            corr = abs(tr[f].corr(tr[TARGET]))
            if corr > 0.85:
                print(f'leakage warning: {f}  corr={corr:.3f}')
                any_leak = True
    if not any_leak:
        print('leakage check passed')

    return (
        (X_tr_w,  y_tr, X_va_w,  y_va, X_te_w,  y_te, feat_w),
        (X_tr_wo, y_tr, X_va_wo, y_va, X_te_wo, y_te, feat_wo),
        tbl,
    )


def best_threshold(y_true, probs):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 200):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def score(name, cond, y_true, y_pred, probs, thr, elapsed):
    m = dict(
        model     = name,
        condition = cond,
        threshold = round(float(thr), 4),
        train_sec = round(elapsed, 1),
        accuracy  = float(accuracy_score(y_true, y_pred)),
        f1        = float(f1_score(y_true, y_pred, zero_division=0)),
        precision = float(precision_score(y_true, y_pred, zero_division=0)),
        recall    = float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc   = float(roc_auc_score(y_true, probs)),
        avg_prec  = float(average_precision_score(y_true, probs)),
    )
    
    print(f'\n{name} | {cond}   thr={thr:.3f}   {elapsed:.1f}s')
    for k in ['f1', 'precision', 'recall', 'avg_prec', 'roc_auc', 'accuracy']:
        print(f'      {k:<12}: {m[k]:.4f}')
    return m


def run_persistence(y_te, cond):
    pred = np.concatenate([[0], np.array(y_te)[:-1]])
    return score('Persistence', cond, y_te, pred, pred.astype(float), 0.5, 0.0), pred.astype(float)


def run_rf(X_tr, y_tr, X_va, y_va, X_te, y_te, cond):
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        n_jobs=-1, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    thr   = best_threshold(y_va, clf.predict_proba(X_va)[:, 1])
    probs = clf.predict_proba(X_te)[:, 1]
    pred  = (probs >= thr).astype(int)
    return score('Random Forest', cond, y_te, pred, probs, thr, time.time() - t0), probs


def run_lgbm(X_tr, y_tr, X_va, y_va, X_te, y_te, cond):
    spw    = (y_tr == 0).sum() / (y_tr == 1).sum()
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    params = dict(
        objective='binary', metric=['binary_logloss', 'auc'],
        scale_pos_weight=spw, num_leaves=63, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1,
    )
    t0    = time.time()
    model = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dtrain, dval], valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=100)],
    )
    thr   = best_threshold(y_va, model.predict(X_va))
    probs = model.predict(X_te)
    pred  = (probs >= thr).astype(int)
    return score('LightGBM', cond, y_te, pred, probs, thr, time.time() - t0), probs


def run_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names, cond):
    spw    = (y_tr == 0).sum() / (y_tr == 1).sum()
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dval   = xgb.DMatrix(X_va, label=y_va, feature_names=feat_names)
    params = dict(
        objective='binary:logistic', eval_metric=['logloss', 'auc'],
        scale_pos_weight=spw, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method='hist', verbosity=0,
    )
    t0    = time.time()
    model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        callbacks=[xgb.callback.EarlyStopping(
            rounds=50, metric_name='auc', maximize=True, save_best=True)],
        verbose_eval=False,
    )
    print(f'  best iteration: {model.best_iteration}')
    thr   = best_threshold(y_va, model.predict(xgb.DMatrix(X_va, feature_names=feat_names)))
    probs = model.predict(xgb.DMatrix(X_te, feature_names=feat_names))
    pred  = (probs >= thr).astype(int)
    return score('XGBoost', cond, y_te, pred, probs, thr, time.time() - t0), probs


def make_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def run_rnn(arch, X_tr, y_tr, X_va, y_va, X_te, y_te, cond, out_dir):
    sc = StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(X_tr), sc.transform(X_va), sc.transform(X_te)

    Xtr, ytr = make_sequences(Xtr_s, y_tr)
    Xva, yva = make_sequences(Xva_s, y_va)
    Xte, yte = make_sequences(Xte_s, y_te)
    print(f'  sequences — train:{Xtr.shape}  val:{Xva.shape}  test:{Xte.shape}')

    Layer = LSTM if arch == 'LSTM' else GRU
    pos_w = (ytr == 0).sum() / (ytr == 1).sum()

    model = Sequential([
        Input(shape=(SEQ_LEN, Xtr.shape[2])),
        Layer(128, return_sequences=True),
        Dropout(0.3),
        Layer(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=Adam(1e-3),
        loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=['AUC'],
    )
    print(f'  params: {model.count_params():,}')

    t0 = time.time()
    history = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=50, batch_size=512,
        class_weight={0: 1.0, 1: float(pos_w)},
        callbacks=[
            EarlyStopping(monitor='val_AUC', patience=10, mode='max',
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_AUC', factor=0.5, patience=5,
                              mode='max', verbose=0),
        ],
        verbose=1,
    )
    elapsed = time.time() - t0

    thr   = best_threshold(yva, model.predict(Xva, verbose=0).ravel())
    probs = model.predict(Xte, verbose=0).ravel()
    pred  = (probs >= thr).astype(int)
    m     = score(arch, cond, yte, pred, probs, thr, elapsed)

    probs_full = np.concatenate([np.full(SEQ_LEN, 0.5), probs])
    return m, probs_full



def make_plots(results, y_te, out_dir):
    df   = pd.DataFrame(results)
    order = ['Persistence', 'Random Forest', 'LightGBM', 'XGBoost', 'LSTM', 'GRU']
    cmap  = {
        'Persistence': COLORS['baseline'], 'Random Forest': COLORS['rf'],
        'LightGBM': COLORS['lgbm'],        'XGBoost': COLORS['xgb'],
        'LSTM': COLORS['lstm'],            'GRU': COLORS['gru'],
    }
    models  = [m for m in order if m in df['model'].values]
    metrics = ['f1', 'precision', 'recall', 'avg_prec', 'accuracy', 'roc_auc']
    labels  = ['F1', 'Precision', 'Recall', 'Avg Precision', 'Accuracy', 'AUC-ROC']

   
    rows = []
    for mod in models:
        rw  = df[(df['model'] == mod) & (df['condition'] == WITH_THR)]
        rwo = df[(df['model'] == mod) & (df['condition'] == NO_THR)]
        if len(rw) and len(rwo):
            for met, lbl in zip(metrics, labels):
                rows.append(dict(model=mod, metric=lbl,delta=rw[met].values[0] - rwo[met].values[0]))
    if rows:
        pivot = (pd.DataFrame(rows)
                   .pivot(index='model', columns='metric', values='delta')
                   .reindex([m for m in models if m in pd.DataFrame(rows)['model'].values]))
        pivot = pivot[[lbl for lbl in labels if lbl in pivot.columns]]

        fig, ax = plt.subplots(figsize=(16, 6))
        pivot.plot(kind='bar', ax=ax, width=0.72, alpha=0.88)
        ax.axhline(0, color='black', lw=1.5, ls='--')
        ax.set_title(
            'Δ Performance from Dynamic Threshold Features (WITH − WITHOUT)',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('model'); ax.set_ylabel('Δ score')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
        ax.legend(title='metric', fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'threshold_delta.png'), dpi=150, bbox_inches='tight')
        plt.close()


def print_summary(results, out_dir):
    cols = ['model', 'condition', 'f1', 'precision', 'recall',
            'avg_prec', 'accuracy', 'roc_auc', 'threshold', 'train_sec']
    df   = pd.DataFrame(results)[cols]
    df['_ord'] = df['condition'].map({WITH_THR: 0, NO_THR: 1})
    df   = df.sort_values(['_ord', 'f1'], ascending=[True, False]).drop(columns='_ord')

    print('\nResults:')
    for cond in [WITH_THR, NO_THR]:
        print(f'\n  {cond}')
        sub = df[df['condition'] == cond].drop(columns='condition')
        print(sub.to_string(index=False))

    xw  = df[(df['model'] == 'XGBoost') & (df['condition'] == WITH_THR)]
    xwo = df[(df['model'] == 'XGBoost') & (df['condition'] == NO_THR)]
    if len(xw) and len(xwo):
        print('\n  XGBoost gains from dynamic threshold:')
        for m in ['f1', 'precision', 'recall', 'avg_prec', 'roc_auc']:
            print(f'    {m:<12}: {xw[m].values[0] - xwo[m].values[0]:+.4f}')

    return df



def main():
  
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    if not os.path.isdir(data_dir):
        sys.exit(f'Data dir not found: {data_dir}')

    out_dir = os.path.join(data_dir, 'ComparisonResults')
    os.makedirs(out_dir, exist_ok=True)

    train_df, val_df, test_df = load_data(data_dir)
    set_w, set_wo, _ = build_feature_sets(train_df, val_df, test_df)
    X_tr_w, y_tr, X_va_w, y_va, X_te_w, y_te, feat_w = set_w
    X_tr_wo, _, X_va_wo, _, X_te_wo, _, feat_wo = set_wo

    results = []
    probs = {}

    print('\nPersistence model')
    for cond in [WITH_THR, NO_THR]:
        m, p = run_persistence(y_te, cond)
        results.append(m)
        probs[('Persistence', cond)] = p

    print('\nRandom Forest')
    for cond, X_tr, X_va, X_te in [
        (WITH_THR, X_tr_w, X_va_w, X_te_w),
        (NO_THR, X_tr_wo, X_va_wo, X_te_wo),
    ]:
        m, p = run_rf(X_tr, y_tr, X_va, y_va, X_te, y_te, cond)
        results.append(m)
        probs[('Random Forest', cond)] = p

    print('\nLightGBM')
    for cond, X_tr, X_va, X_te in [
        (WITH_THR, X_tr_w, X_va_w, X_te_w),
        (NO_THR, X_tr_wo, X_va_wo, X_te_wo),
    ]:
        m, p = run_lgbm(X_tr, y_tr, X_va, y_va, X_te, y_te, cond)
        results.append(m)
        probs[('LightGBM', cond)] = p

    print('\nXGBoost')
    for cond, X_tr, X_va, X_te, fn in [
        (WITH_THR, X_tr_w, X_va_w, X_te_w, feat_w),
        (NO_THR, X_tr_wo, X_va_wo, X_te_wo, feat_wo),
    ]:
        m, p = run_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te, fn, cond)
        results.append(m)
        probs[('XGBoost', cond)] = p

    
    for arch in ['LSTM', 'GRU']:
        print(f'\n── {arch} ──')
        for cond, X_tr, X_va, X_te in [
            (WITH_THR, X_tr_w, X_va_w, X_te_w),
            (NO_THR, X_tr_wo, X_va_wo, X_te_wo),
            ]:
            m, p = run_rnn(arch, X_tr, y_tr, X_va, y_va, X_te, y_te, cond, out_dir)
            results.append(m)
            probs[(arch, cond)] = p
   
        
    for r in results:
        r['prob'] = probs.get((r['model'], r['condition']))

    make_plots(results, y_te, out_dir)
    print_summary(results, out_dir)


if __name__ == '__main__':
    main()