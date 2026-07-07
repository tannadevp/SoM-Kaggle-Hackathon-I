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


## Repo structure

```
.
├── README.md
├── requirements.txt
├── notebook/
│   └── submission.ipynb
```

## Approach

1. **Load & inspect :** `train.csv` / `test.csv` loaded; a quick check confirms `Seasons` is disjoint
   between splits — train covers Winter/Spring/Summer, test is entirely Autumn.
2. **Correlation scan :** Pearson correlation of each raw numeric column against the (log1p) target to
   sanity-check signal direction before modeling — `Temperature`, `Dew point temperature`, and `Hour`
   come out strongest.
3. **EDA :** Mean demand by hour, demand on functioning vs. non-functioning days (confirms
   non-functioning days have exactly zero demand — a hard rule, not a soft trend), and a boxplot of
   demand distribution per hour.
4. **Feature engineering :** (`preprocess_data`), applied identically to train and test:
   - **Calendar:** `Month`, `DayOfWeek`, `IsWeekend` parsed from `Date`
   - **Cyclical encodings:** `hour_sin`/`hour_cos`, `month_sin`/`month_cos`
   - **Rush-hour flags:** `is_morning_rush`, `is_evening_rush`, `is_rush_hour`
   - **Weather flags & interactions:** `rain_flag`, `snow_flag`, `bad_weather`, `rush_x_bad_weather`,
     `rush_x_temp`, `temp_comfortable`
   - **Categorical → binary:** `functioning_day_flag`, `is_holiday`, one-hot season flags
   - **Polynomial terms (new vs. the prior notebook):** `Temp_sq`, `Humidity_sq`, `Wind_sq`, `Rain_sq`,
     `Snow_sq` — manually squared versions of five raw columns
   - Raw `Id`, `Date`, `Holiday`, `Functioning Day`, `Hour`, `Month`, `DayOfWeek`, `Seasons` dropped
     once their engineered replacements exist
5. **Chronological 80/20 split :** (`X.iloc[:split]` / `X.iloc[split:]`) rather than a random split —
   appropriate for hourly time-series data, since a random split would let the model "see the future"
   via adjacent hours leaking into training.
6. **Scaling :** `StandardScaler` fit on the training fold only, then applied to validation and test —
   necessary because Ridge's penalty is scale-sensitive (an unscaled feature with large magnitude is
   penalized less per unit of predictive power than a small-magnitude one).
7. **Model :** `Ridge(alpha=10)` fit on scaled training data. Local validation RMSE: **0.7880**.
8. **Interpretation :** Coefficient magnitudes ranked to identify influential features
   (`functioning_day_flag`, `hour_sin`, `bad_weather` top the list).
9. **Refit on full data** with a fresh scaler (`scaler_final`) and generate test predictions
   (`ridge_final`) — correct practice in isolation.
10. **Post-processing :** Predictions for `Functioning Day == No` rows are hard-clamped to `0`.
11. **Export** 

## Concepts used

| Concept | Why it's here |
|---|---|
| **Ridge regression (L2 regularization)** | Shrinks coefficients toward zero to reduce variance/overfitting, especially useful with many correlated engineered features |
| **Feature standardization** | Puts all features on comparable scale so the L2 penalty treats them fairly |
| **Cyclical (sin/cos) encoding** | Encodes periodic variables (hour, month) so e.g. hour 23 and hour 0 are numerically adjacent instead of maximally far apart |
| **Interaction features** | Manually crafted terms (`rush_x_temp`, `rush_x_bad_weather`) let a linear model approximate conditional effects it can't otherwise express |
| **Polynomial (quadratic) features** | Squared terms let a linear model fit curvature (e.g. demand peaking at moderate temperature, not increasing linearly forever) |
| **Chronological train/validation split** | Avoids the optimistic bias of random splits on temporally ordered data |
| **Domain-rule post-processing** | Encodes a known deterministic fact (system closed ⇒ demand is 0) directly, rather than hoping the model learns it |

### Top features by |coefficient| 

`functioning_day_flag` > `hour_sin` > `bad_weather` > `snow_flag` > `Humidity(%)` > `Dew point
temperature(°C)` > `is_morning_rush` > `Temperature(°C)`. This confirms the domain hypothesis that
operational status, time-of-day cyclicality, and adverse weather dominate demand.


## License

Code in this repository is provided under the MIT License. The competition dataset itself is licensed
CC BY-NC-SA 4.0 by the host and is **not redistributed** here.
