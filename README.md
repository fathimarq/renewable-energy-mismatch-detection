# Renewable Energy Inefficiency Prediction (Spain)

Predicts hours of **high energy-system inefficiency** in Spain from hourly generation, load and weather data. Uses XGBoost with context-aware dynamic thresholds (season × time of day × weather), compares it against other models, explains it with SHAP, and ships a Streamlit dashboard.

## Data

Hourly Spanish energy and weather data (`energy_dataset.csv`, `weather_features.csv`), likely from the Kaggle dataset [*Energy consumption, generation, prices and weather*](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather).

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

## Usage

Run in this order from the project root:

```bash
python Preproc.py       # preprocessing + train/val/test split
python model.py         # trains XGBoost
python comparison.py    # model comparison with/without threshold features
python explain.py       # SHAP analysis
streamlit run dashboard.py
```

## Results

XGBoost on the chronological test set (threshold tuned on validation):

| Accuracy | F1 | Precision | Recall | ROC-AUC |
|---------:|---:|----------:|-------:|--------:|
| 0.964 | 0.931 | 0.960 | 0.905 | 0.996 |

Plots and metrics are in `output/` and `SHAP_Analysis/`.
