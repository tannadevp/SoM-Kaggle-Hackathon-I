# Urban Micro-Mobility Demand Forecasting

Regression solution for the **SoM '26 Kaggle Hackathon-I**: predicting hourly two-wheeler rental demand
(`Rented_Bike_Count`) in a major metropolitan city from one year of weather and calendar data.

## Problem

Two-wheeler ride-hailing platforms need to reposition fleets *before* demand
spikes or drops, driven heavily by weather and time-of-day effects. Given hourly weather
readings, season, holiday flag, and functioning-day flag, predict `Rented_Bike_Count` for every hour in
the test set.

- **Task type:** Time-series regression
- **Metric:** RMSE
- **Target transform:** `Rented_Bike_Count` is provided **pre-`log1p`-transformed** in `train.csv`. Models
  are trained and submit directly on this log scale. RMSE on the log scale is mathematically equivalent
  to RMSLE on raw counts. `numpy.expm1()` is only needed to inspect real counts during EDA.
- **Data span:** December 2017 – November 2018.

See [`data/README.md`](data/README.md) for the full column schema.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md              # schema + instructions to fetch train/test.csv from Kaggle
├── notebooks/
│   ├── 01_baseline_linear_regression.ipynb
│   ├── 02_ridge_feature_engineering.ipynb
│   └── 03_ridge_polynomial_features.ipynb
└── submissions/
    ├── submission1.csv
    ├── submission2.csv        # produced by 01_baseline_linear_regression.ipynb
    ├── submission3.csv
    ├── submission4.csv        # produced by 02_ridge_feature_engineering.ipynb
    ├── submission5.csv        
    └── submission6.csv        # produced by 03_ridge_polynomial_features.ipynb (see Known Issues)
```

## Approach

| # | Notebook | Model | Key idea | Local val. RMSE |
|---|---|---|---|---|
| 1 | `01_baseline_linear_regression.ipynb` | `LinearRegression` | Numeric columns only, categorical columns dropped, random 80/20 split | **0.9585** |
| 2 | `02_ridge_feature_engineering.ipynb` | `Ridge(alpha=10)` | Full feature engineering pass (below) + `StandardScaler`, chronological 80/20 split | **0.7880** |
| 3 | `03_ridge_polynomial_features.ipynb` | `Ridge(alpha=10)` | Same as #2 + squared terms for `Temperature`, `Humidity`, `Wind speed`, `Rainfall`, `Snowfall` | **0.7880** |


### Feature engineering (notebooks 02 & 03)

Starting from the baseline's 9 raw numeric columns, the following were engineered:

- **Calendar:** `Month`, `DayOfWeek`, `IsWeekend` parsed from `Date`
- **Cyclical encodings:** `hour_sin`/`hour_cos`, `month_sin`/`month_cos` (so hour 23 and hour 0 are
  treated as adjacent, not maximally distant)
- **Rush-hour flags:** `is_morning_rush` (7–9), `is_evening_rush` (17–19), `is_rush_hour`
- **Weather flags & interactions:** `rain_flag`, `snow_flag`, `bad_weather`, `rush_x_bad_weather`,
  `rush_x_temp`, `temp_comfortable` (10–25 °C)
- **Categoricals → binary flags:** `is_holiday`, `functioning_day_flag`, one-hot season flags
  (`is_summer`, `is_winter`, `is_spring`, `is_autumn`)
- **Post-processing:** predictions for rows where `Functioning Day == No` are hard-clamped to `0`, since
  the system genuinely cannot rent bikes on those hours (confirmed in EDA — 100% of non-functioning-day
  training rows have zero demand)

### Top features by |coefficient| (Ridge, notebook 02)

`functioning_day_flag` > `hour_sin` > `bad_weather` > `snow_flag` > `Humidity(%)` > `Dew point
temperature(°C)` > `is_morning_rush` > `Temperature(°C)`. This confirms the domain hypothesis that
operational status, time-of-day cyclicality, and adverse weather dominate demand.

## Known issues / lessons learned

- **`submission5.csv` is broken.** The final cell of `03_ridge_polynomial_features.ipynb` calls
  `model.predict(X_test)` on the **raw, unscaled** test features using the `Ridge` object fit earlier on
  **scaled** training data (`X_train_sc`), instead of the correctly-scaled `ridge_final` / `X_test_sc`
  pair computed two cells earlier. This produces wildly out-of-range predictions (roughly −7400 to −190
  on the log scale). It's kept in the repo as-is for transparency and as a reminder to always predict
  with the same preprocessing pipeline used at fit time — ideally by wrapping scaler + model in a single
  `sklearn.pipeline.Pipeline` so this class of bug becomes structurally impossible.
- **Polynomial terms added zero value on top of Ridge** (#2 vs #3 tie at 0.7880 RMSE). Squared terms
  alone don't capture the actual nonlinearity in this data (e.g. the rush-hour × weather interactions
  already engineered manually).
- **Train/test have disjoint seasons** — train covers Winter/Spring/Summer, test is entirely Autumn. Any
  model relying too heavily on season one-hot flags risks not generalizing; cyclical month/hour encodings
  are more robust here.

## Setup & running

```bash
git clone <this-repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt

# then, per data/README.md:
# download train.csv, test.csv, sample_submission.csv from the Kaggle Data tab into data/

jupyter notebook notebooks/01_baseline_linear_regression.ipynb
```

Each notebook is self-contained end-to-end: load data → EDA → preprocess → train → validate → generate
`submission.csv` in Kaggle's required `Id,Rented_Bike_Count` format.

## License

Code in this repository is provided under the MIT License. The competition dataset itself is licensed
CC BY-NC-SA 4.0 by the host and is **not redistributed** here — see [`data/README.md`](data/README.md).
