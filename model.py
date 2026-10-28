import argparse, os, sys, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay,
    average_precision_score,
)
import xgboost as xgb

warnings.filterwarnings('ignore')

COLORS = dict(primary='#2563EB', secondary='#DC2626', success='#16A34A',
              warning='#D97706', purple='#7C3AED', teal='#0D9488', gray='#6B7280')
SEASON_NAMES  = {0:'Winter', 1:'Spring', 2:'Summer', 3:'Autumn'}
SEASON_COLORS = {0:'#3B82F6', 1:'#22C55E', 2:'#EF4444', 3:'#F59E0B'}
TARGET = 'high_inefficiency'

LEAKAGE_FEATURES = [
    'renewable_penetration',
    'renewable_share',
    'total_renewable_generation',
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=None)
    return p.parse_args()

def load_data(data_dir):
    print('\nSTEP 1: LOADING PREPROCESSED DATA')
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val   = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test  = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print(f'  Train: {train.shape}  Val: {val.shape}  Test: {test.shape}')
    print(f'  Target dist (train): {train[TARGET].value_counts().to_dict()}')

    print('\n  Leakage check:')
    for f in LEAKAGE_FEATURES:
        if f in train.columns:
            corr = train[f].corr(train[TARGET])
            print(f'  REMOVED  {f:<45s} corr={corr:.4f}')
    return train, val, test

def compute_dynamic_thresholds(train, out_dir):
    print('\nSTEP 2: DYNAMIC THRESHOLD MODELING')

    df = train.copy()
    df['hour_block']     = (df['hour'] // 4).astype(int)
    df['weather_regime'] = pd.cut(df['clouds_all'], bins=[-1,25,65,101],
                                   labels=['clear','partial','overcast'])

    proxy = 'generation wind onshore'

    tbl = df.groupby(['season','hour_block','weather_regime'], observed=True).agg(
        n            = (proxy,                   'count'),
        proxy_mean   = (proxy,                   'mean'),
        proxy_std    = (proxy,                   'std'),
        ineff_rate   = (TARGET,                  'mean'),
        wind_mean    = ('wind_speed',             'mean'),
        temp_mean    = ('temp_celsius',           'mean'),
        solar_mean   = ('generation solar',       'mean'),
        fossil_mean  = ('generation fossil gas',  'mean'),
        load_mean    = ('total load actual',      'mean'),
    ).reset_index()

    tbl['proxy_std'] = tbl['proxy_std'].fillna(0)
    tbl['thr_high']  = tbl['proxy_mean'] + 1.5 * tbl['proxy_std']
    tbl['thr_low']   = (tbl['proxy_mean'] - 1.5 * tbl['proxy_std']).clip(lower=0)
    tbl['thr_range'] = tbl['thr_high'] - tbl['thr_low']

    print(f'  Threshold cells: {len(tbl)}')
    worst = tbl.loc[tbl['ineff_rate'].idxmax()]
    print(f'  Worst cell: Season={SEASON_NAMES[worst["season"]]}  '
          f'Hour-block={worst["hour_block"]}  Regime={worst["weather_regime"]}  '
          f'Ineff={worst["ineff_rate"]:.3f}')

    tbl.to_csv(os.path.join(out_dir, 'dynamic_thresholds_fixed.csv'), index=False)
    return tbl

def attach_threshold_features(df, tbl, proxy='generation wind onshore'):
    d = df.copy()
    d['hour_block']     = (d['hour'] // 4).astype(int)
    d['weather_regime'] = pd.cut(d['clouds_all'], bins=[-1,25,65,101],
                                  labels=['clear','partial','overcast'])
    merge_cols = ['season','hour_block','weather_regime',
                  'proxy_mean','proxy_std','thr_high','thr_low','thr_range','ineff_rate']
    d = d.merge(tbl[merge_cols],
                on=['season','hour_block','weather_regime'], how='left')

    d['wind_above_high']   = (d[proxy] - d['thr_high']).clip(lower=0)
    d['wind_below_low']    = (d['thr_low'] - d[proxy]).clip(lower=0)
    d['wind_deviation']    = d[proxy] - d['proxy_mean']
    d['wind_zscore']       = d['wind_deviation'] / (d['proxy_std'] + 1e-6)
    d['exceeds_threshold'] = (d[proxy] > d['thr_high']).astype(int)
    d['below_threshold']   = (d[proxy] < d['thr_low']).astype(int)

    new_cols = ['proxy_mean','proxy_std','thr_high','thr_low','thr_range','ineff_rate',
                'wind_above_high','wind_below_low','wind_deviation',
                'wind_zscore','exceeds_threshold','below_threshold']
    d[new_cols] = d[new_cols].fillna(0)
    d.drop(columns=['hour_block','weather_regime'], inplace=True, errors='ignore')
    return d

def get_features(df):
    drop = LEAKAGE_FEATURES + [TARGET, 'year']
    feats = [c for c in df.columns if c not in drop]
    return df[feats], df[TARGET]

def train_xgboost(X_tr, y_tr, X_va, y_va, out_dir):
    print('\nSTEP 3: XGBOOST TRAINING')

    spw = (y_tr==0).sum() / (y_tr==1).sum()
    print(f'  scale_pos_weight = {spw:.3f}')

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns))
    dval   = xgb.DMatrix(X_va, label=y_va, feature_names=list(X_va.columns))

    params = dict(
        objective='binary:logistic', eval_metric=['logloss','auc'],
        scale_pos_weight=spw, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method='hist', verbosity=0,
    )
    evals_result = {}
    cb = [xgb.callback.EarlyStopping(rounds=50, metric_name='auc',
                                      maximize=True, save_best=True)]
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dtrain,'train'),(dval,'val')],
                      callbacks=cb, evals_result=evals_result, verbose_eval=50)

    model.save_model(os.path.join(out_dir, 'xgb_model_fixed.json'))
    print(f'  Best iteration: {model.best_iteration}')
    print(f'  Best val AUC  : {model.best_score:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('XGBoost Training Curves', fontsize=13, fontweight='bold')
    for ax, metric in zip(axes, ['auc','logloss']):
        ax.plot(evals_result['train'][metric], label='Train',
                color=COLORS['primary'], lw=1.5)
        ax.plot(evals_result['val'][metric], label='Val',
                color=COLORS['secondary'], lw=1.5, ls='--')
        ax.axvline(model.best_iteration, color=COLORS['gray'], ls=':',
                   label=f'Best ({model.best_iteration})')
        ax.set_title(metric.upper())
        ax.set_xlabel('Round')
        ax.legend()
        ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'xgb_training_curves_fixed.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: xgb_training_curves_fixed.png')

    prob = model.predict(dval)
    pred = (prob >= 0.5).astype(int)
    m = _metrics(y_va, pred, prob, 'XGBoost (Val)')
    return model, m, prob

def find_optimal_threshold(y_true, y_prob, out_dir):
    print('\nSTEP 4: THRESHOLD OPTIMISATION')
    thresholds = np.linspace(0.05, 0.95, 200)
    f1s, precs, recs, accs = [], [], [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, pred, zero_division=0))
        precs.append(precision_score(y_true, pred, zero_division=0))
        recs.append(recall_score(y_true, pred, zero_division=0))
        accs.append(accuracy_score(y_true, pred))
    best_idx = int(np.argmax(f1s))
    opt_thr  = thresholds[best_idx]
    print(f'  Optimal threshold : {opt_thr:.4f}')
    print(f'  Best val F1       : {f1s[best_idx]:.4f}')
    print(f'  Precision  opt   : {precs[best_idx]:.4f}')
    print(f'  Recall     opt   : {recs[best_idx]:.4f}')
    return opt_thr

def _metrics(y_true, y_pred, y_prob, label):
    m = dict(label=label,
             accuracy =accuracy_score(y_true, y_pred),
             f1       =f1_score(y_true, y_pred, zero_division=0),
             precision=precision_score(y_true, y_pred, zero_division=0),
             recall   =recall_score(y_true, y_pred, zero_division=0),
             roc_auc  =roc_auc_score(y_true, y_prob),
             avg_prec =average_precision_score(y_true, y_prob))
    print(f'\n  [{label}]')
    for k,v in m.items():
        if k != 'label': print(f'    {k:12s}: {v:.4f}')
    return m

def evaluate_test(model, opt_thr, X_te, y_te, out_dir, feature_names):
    print('\nSTEP 5: FINAL TEST EVALUATION')

    dte = xgb.DMatrix(X_te, label=y_te, feature_names=feature_names)
    prob_xgb = model.predict(dte)
    pred_opt = (prob_xgb >= opt_thr).astype(int)
    pred_05  = (prob_xgb >= 0.50).astype(int)
    m_opt = _metrics(y_te, pred_opt, prob_xgb, f'XGBoost (Test, thr={opt_thr:.3f})')
    m_05  = _metrics(y_te, pred_05,  prob_xgb, 'XGBoost (Test, thr=0.50)')

    print('\n  Classification Report (XGBoost, optimal threshold):')
    print(classification_report(y_te, pred_opt,
                                 target_names=['Normal','High Ineff.'], digits=4))

    fig = plt.figure(figsize=(20, 20))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle('Model Evaluation Dashboard - Test Set',
                 fontsize=15, fontweight='bold')

    for col, (pred, lbl) in enumerate([
        (pred_opt, f'XGBoost (thr={opt_thr:.2f})'),
        (pred_05,  'XGBoost (thr=0.50)'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt='d',
                    cmap='Blues', ax=ax,
                    xticklabels=['Normal','High'], yticklabels=['Normal','High'])
        ax.set_title(f'Confusion Matrix\n{lbl}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    ax_roc = fig.add_subplot(gs[1, 0])
    RocCurveDisplay.from_predictions(y_te, prob_xgb, ax=ax_roc,
                                      name=f'XGBoost opt (AUC={m_opt["roc_auc"]:.3f})',
                                      color=COLORS['primary'], lw=1.8)
    ax_roc.plot([0,1],[0,1],'k--',lw=1)
    ax_roc.set_title('ROC Curve', fontsize=11, fontweight='bold')
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=.3)

    ax_pr = fig.add_subplot(gs[1, 1])
    PrecisionRecallDisplay.from_predictions(y_te, prob_xgb, ax=ax_pr,
                                             name=f'XGBoost (AP={m_opt["avg_prec"]:.3f})',
                                             color=COLORS['primary'], lw=1.8)
    ax_pr.set_title('Precision-Recall Curve', fontsize=11, fontweight='bold')
    ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=.3)

    ax_bar = fig.add_subplot(gs[1, 2])
    mk = ['accuracy','f1','precision','recall','roc_auc']
    ml = ['Accuracy','F1','Precision','Recall','AUC']
    x  = np.arange(len(mk)); w = 0.35
    ax_bar.bar(x-w/2, [m_opt[k] for k in mk], w, label='XGBoost opt', color=COLORS['primary'])
    ax_bar.bar(x+w/2, [m_05[k]  for k in mk], w, label='XGBoost 0.5', color=COLORS['teal'])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ml, fontsize=9)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel('Score')
    ax_bar.set_title('Metric Comparison', fontsize=11, fontweight='bold')
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis='y', alpha=.3)
    for c in ax_bar.containers: ax_bar.bar_label(c, fmt='%.2f', fontsize=7, padding=2)

    ax_dist = fig.add_subplot(gs[2, 0])
    ax_dist.hist(prob_xgb[y_te==0], bins=60, alpha=0.6,
                 color=COLORS['primary'], label='Normal', density=True)
    ax_dist.hist(prob_xgb[y_te==1], bins=60, alpha=0.6,
                 color=COLORS['secondary'], label='High Ineff.', density=True)
    ax_dist.axvline(opt_thr, color='black', ls='--', lw=1.5,
                    label=f'Opt thr={opt_thr:.3f}')
    ax_dist.set_title('Predicted Probability Distributions', fontsize=11, fontweight='bold')
    ax_dist.set_xlabel('P(High Inefficiency)')
    ax_dist.set_ylabel('Density')
    ax_dist.legend(fontsize=9)
    ax_dist.grid(alpha=.3)

    ax_cal = fig.add_subplot(gs[2, 1])
    frac_pos, mean_pred = calibration_curve(y_te, prob_xgb, n_bins=15)
    ax_cal.plot(mean_pred, frac_pos, 's-', color=COLORS['primary'], lw=2, label='XGBoost')
    ax_cal.plot([0,1],[0,1], 'k--', lw=1, label='Perfect')
    ax_cal.set_title('Calibration Curve', fontsize=11, fontweight='bold')
    ax_cal.set_xlabel('Mean Predicted Prob.')
    ax_cal.set_ylabel('Fraction Positive')
    ax_cal.legend(fontsize=9)
    ax_cal.grid(alpha=.3)

    ax_tbl = fig.add_subplot(gs[2, 2])
    ax_tbl.axis('off')
    rows = [['Metric','XGBoost\n(opt)','XGBoost\n(0.5)']]
    for k, lbl in zip(mk, ml):
        rows.append([lbl, f'{m_opt[k]:.4f}', f'{m_05[k]:.4f}'])
    rows.append(['Avg Prec', f'{m_opt["avg_prec"]:.4f}', f'{m_05["avg_prec"]:.4f}'])
    t = ax_tbl.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.35, 0.32, 0.32])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    ax_tbl.set_title('Metrics Summary', fontsize=11, fontweight='bold')

    plt.savefig(os.path.join(out_dir, 'evaluation_dashboard_fixed.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: evaluation_dashboard_fixed.png')
    return m_opt, m_05


def main():
    args = parse_args()
    if args.data_dir:
        data_dir = args.data_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir   = os.path.join(script_dir, 'output')
    if not os.path.isdir(data_dir):
        sys.exit(f'ERROR: data dir not found: {data_dir}\nRun Preproc.py first.')

    out_dir = os.path.join(data_dir, 'model_output')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')

    train_df, val_df, test_df = load_data(data_dir)

    thr_table = compute_dynamic_thresholds(train_df, out_dir)

    print('\n  Attaching threshold features...')
    train_df = attach_threshold_features(train_df, thr_table)
    val_df   = attach_threshold_features(val_df,   thr_table)
    test_df  = attach_threshold_features(test_df,  thr_table)
    print(f'  Feature count after attachment: {len(train_df.columns)}')

    X_tr, y_tr = get_features(train_df)
    X_va, y_va = get_features(val_df)
    X_te, y_te = get_features(test_df)
    feature_names = list(X_tr.columns)
    print(f'  Final feature count (no leakage): {len(feature_names)}')

    xgb_model, m_xgb_val, prob_xgb_va = train_xgboost(X_tr, y_tr, X_va, y_va, out_dir)

    opt_thr = find_optimal_threshold(y_va, prob_xgb_va, out_dir)
    joblib.dump({'optimal_threshold': opt_thr},
                os.path.join(out_dir, 'threshold.pkl'))

    m_opt, m_05 = evaluate_test(xgb_model, opt_thr, X_te, y_te, out_dir, feature_names)

    

    all_m = [m_xgb_val, m_opt, m_05]
    pd.DataFrame([{k:v for k,v in m.items()} for m in all_m]).to_csv(
        os.path.join(out_dir, 'model_metrics_fixed.csv'), index=False)

    print('\nPIPELINE COMPLETE')
    print(f'\n  XGBoost Test AUC  : {m_opt["roc_auc"]:.4f}')
    print(f'  XGBoost Test F1   : {m_opt["f1"]:.4f}')
    print(f'  XGBoost Precision : {m_opt["precision"]:.4f}')
    print(f'  XGBoost Recall    : {m_opt["recall"]:.4f}')
    print(f'  Optimal Threshold : {opt_thr:.4f}')
    print(f'\n  All outputs -> {out_dir}')

if __name__ == '__main__':
    main()
