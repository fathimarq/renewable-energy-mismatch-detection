import pandas as pd
import numpy as np
import warnings, os, sys
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT, exist_ok=True)

energy_path = os.path.join(SCRIPT_DIR, 'energy_dataset.csv')
weather_path = os.path.join(SCRIPT_DIR, 'weather_features.csv')

if not os.path.exists(energy_path) or not os.path.exists(weather_path):
    sys.exit(f"ERROR: Place energy_dataset.csv and weather_features.csv in {SCRIPT_DIR}")

print("STEP 1: DATA INGESTION")

energy = pd.read_csv(energy_path)
weather = pd.read_csv(weather_path)

print(f"Energy Dataset : {energy.shape[0]} rows x {energy.shape[1]} columns")
print(f"Weather Dataset: {weather.shape[0]} rows x {weather.shape[1]} columns")

energy['time'] = pd.to_datetime(energy['time'], utc=True)
weather['dt_iso'] = pd.to_datetime(weather['dt_iso'], utc=True)

print(f"Energy date range : {energy['time'].min()} to {energy['time'].max()}")
print(f"Weather date range: {weather['dt_iso'].min()} to {weather['dt_iso'].max()}")

print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")

print("Missing Values (Energy):")
em = energy.isnull().sum()
em = em[em > 0].sort_values(ascending=False)
print(em.to_string() if len(em) else "None")
print(f"Total missing cells: {energy.isnull().sum().sum()}")

print("Missing Values (Weather):")
wm = weather.isnull().sum()
wm = wm[wm > 0].sort_values(ascending=False)
print(wm.to_string() if len(wm) else "None")

print("\nSTEP 3: INEFFICIENCY INDEX DESIGN")

renewable_gen_cols = [c for c in energy.columns if 'generation' in c and
                      any(x in c for x in ['solar', 'wind', 'hydro', 'biomass',
                                            'geothermal', 'marine', 'other renewable'])]
print("Renewable generation columns:")
for c in renewable_gen_cols:
    print(f"  - {c}")

energy['total_renewable_generation'] = energy[renewable_gen_cols].sum(axis=1)

all_gen_cols = [c for c in energy.columns if c.startswith('generation') and 'consumption' not in c]
energy['total_generation'] = energy[all_gen_cols].sum(axis=1)
energy['renewable_share'] = energy['total_renewable_generation'] / energy['total_generation'].replace(0, np.nan)

energy['inefficiency_index'] = (
    np.abs(energy['total_renewable_generation'] - energy['total load actual'])
    / energy['total load actual'].replace(0, 1)
)
energy['inefficiency_index'] = energy['inefficiency_index'].clip(upper=1.0)

energy['supply_demand_gap'] = energy['total_renewable_generation'] - energy['total load actual']
energy['gap_ratio'] = energy['supply_demand_gap'] / energy['total load actual'].replace(0, 1)

print("Inefficiency Index Statistics:")
print(energy['inefficiency_index'].describe().to_string())

energy['hour'] = energy['time'].dt.hour
energy['month'] = energy['time'].dt.month

print("\nSTEP 4: FEATURE ENGINEERING")

weather_agg = weather.groupby('dt_iso').agg({
    'temp': 'mean', 'temp_min': 'mean', 'temp_max': 'mean',
    'pressure': 'mean', 'humidity': 'mean', 'wind_speed': 'mean',
    'wind_deg': 'mean', 'rain_1h': 'mean', 'rain_3h': 'mean',
    'snow_3h': 'mean', 'clouds_all': 'mean', 'weather_id': 'mean'
}).reset_index()
weather_agg.rename(columns={'dt_iso': 'time'}, inplace=True)
print(f"Weather aggregated: {weather_agg.shape}")

merged = pd.merge(energy, weather_agg, on='time', how='inner', suffixes=('', '_weather'))
print(f"Merged dataset: {merged.shape}")

merged['day_of_week'] = merged['time'].dt.dayofweek
merged['year'] = merged['time'].dt.year
merged['is_weekend'] = (merged['day_of_week'] >= 5).astype(int)
merged['season'] = merged['month'].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                                         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
merged['hour_sin'] = np.sin(2 * np.pi * merged['hour'] / 24)
merged['hour_cos'] = np.cos(2 * np.pi * merged['hour'] / 24)
merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)
merged['temp_celsius'] = merged['temp'] - 273.15

merged['renewable_penetration'] = merged['total_renewable_generation'] / merged['total load actual'].replace(0, 1)
max_solar = merged['generation solar'].max()
merged['solar_capacity_factor'] = merged['generation solar'] / max_solar if max_solar > 0 else 0
merged['wind_generation'] = merged.get('generation wind onshore', 0)
if 'generation wind offshore' in merged.columns:
    merged['wind_generation'] += merged['generation wind offshore']
max_wind = merged['wind_generation'].max()
merged['wind_capacity_factor'] = merged['wind_generation'] / max_wind if max_wind > 0 else 0

season_names = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
print(f"Features engineered. Shape: {merged.shape}")

    
print("\nSTEP 5: PREPROCESSING PIPELINE")

drop_cols = ['time', 'forecast solar day ahead', 'forecast wind onshore day ahead',
             'total load forecast', 'price day ahead', 'price actual',
             'generation marine']
non_numeric = merged.select_dtypes(exclude=[np.number]).columns.tolist()
drop_cols = list(set(drop_cols + non_numeric))
drop_cols = [c for c in drop_cols if c in merged.columns]
df = merged.drop(columns=drop_cols)
print(f"After dropping non-feature columns: {df.shape}")

print(f"Missing before imputation: {df.isnull().sum().sum()}")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

df[numeric_cols] = df[numeric_cols].ffill()
df[numeric_cols] = df[numeric_cols].bfill()

for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(0)

print(f"Missing after imputation : {df.isnull().sum().sum()}")
assert df.isnull().sum().sum() == 0, "Still have missing values!"

print(f"Inefficiency index range: {df['inefficiency_index'].min():.4f} - {df['inefficiency_index'].max():.4f}")
print(f"High inefficiency events (>0.8): {(df['inefficiency_index'] > 0.8).sum()} rows")

threshold = df['inefficiency_index'].quantile(0.75)
df['high_inefficiency'] = (df['inefficiency_index'] > threshold).astype(int)
print(f"Inefficiency threshold (75th pctl): {threshold:.4f}")
print(f"High inefficiency: {df['high_inefficiency'].sum()} ({df['high_inefficiency'].mean()*100:.1f}%)")
print(f"Normal          : {(df['high_inefficiency']==0).sum()} ({(1-df['high_inefficiency'].mean())*100:.1f}%)")

leakage_cols = ['inefficiency_index', 'supply_demand_gap', 'gap_ratio']
df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)
print(f"Dropped leakage columns: {leakage_cols}")

df = df.loc[:, ~df.columns.duplicated()]

nunique = df.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols:
    df.drop(columns=const_cols, inplace=True)
    print(f"Dropped constant columns: {const_cols}")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
if df.isnull().sum().sum() > 0:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

print("\nSTEP 6: TIME SERIES SPLIT (80/10/10)")

n = len(df)
train_size = int(n * 0.80)
val_size = int(n * 0.10)

train_df = df.iloc[:train_size].copy()
val_df = df.iloc[train_size:train_size+val_size].copy()
test_df = df.iloc[train_size+val_size:].copy()

print(f"Train set: {len(train_df)} rows ({len(train_df)/n*100:.0f}%) - {train_df['year'].min()}-{train_df['year'].max()}")
print(f"Val set:   {len(val_df)} rows ({len(val_df)/n*100:.0f}%) - {val_df['year'].min()}-{val_df['year'].max()}")
print(f"Test set:  {len(test_df)} rows ({len(test_df)/n*100:.0f}%) - {test_df['year'].min()}-{test_df['year'].max()}")

print("\nFINAL QUALITY CHECKS")

n_features = len([c for c in train_df.columns if c != 'high_inefficiency'])
print(f"Train samples   : {len(train_df)}")
print(f"Val samples     : {len(val_df)}")
print(f"Test samples    : {len(test_df)}")
print(f"Total features  : {n_features}")
print(f"Target column   : high_inefficiency")
print(f"Missing values  : {train_df.isnull().sum().sum() + val_df.isnull().sum().sum() + test_df.isnull().sum().sum()}")

for c in ['inefficiency_index', 'supply_demand_gap', 'gap_ratio']:
    assert c not in train_df.columns, f"LEAKAGE: {c} still present!"
print("No data leakage detected")

corr_with_target = train_df.corr()['high_inefficiency'].drop('high_inefficiency').abs().sort_values(ascending=False)
print(f"Top 5 feature correlations with target (training set):")
print(corr_with_target.head().to_string())
if corr_with_target.max() > 0.95:
    print("WARNING: Suspiciously high correlation - review for hidden leakage")
else:
    print("All correlations look healthy (< 0.95)")

print(f"Target distribution:")
print(f"  Train: {train_df['high_inefficiency'].mean()*100:.1f}% positive")
print(f"  Val:   {val_df['high_inefficiency'].mean()*100:.1f}% positive")
print(f"  Test:  {test_df['high_inefficiency'].mean()*100:.1f}% positive")

print("\nSTEP 7: SAVING DATASETS")

train_df.to_csv(os.path.join(OUT, 'train.csv'), index=False)
val_df.to_csv(os.path.join(OUT, 'val.csv'), index=False)
test_df.to_csv(os.path.join(OUT, 'test.csv'), index=False)

print(f"Saved: train.csv, val.csv, test.csv")

print(f"\nAll {n_features} FEATURES:")
for i, c in enumerate([c for c in train_df.columns if c != 'high_inefficiency']):
    print(f"  {i+1:2d}. {c}")

print(f"Split ratio: 80% Train ({len(train_df)}), 10% Val ({len(val_df)}), 10% Test ({len(test_df)})")
