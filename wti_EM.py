"""
Python Translation of EViews Forecasting Script for WTI Returns and Volatility

This script replicates the steps taken in the EViews script:
1. Data loading and visualization
2. Stationarity testing (ADF)
3. Autocorrelation analysis (ACF/PACF, Ljung-Box)
4. Mean-model estimation:
   - ARMA with exogenous variables
   - Threshold Autoregressive (TAR) *
   - Self-Exciting TAR (SETAR) *
   - Markov Switching (MS)
5. Forecast evaluation (5‐ and 22‐day horizons) using univariate loss functions
6. Volatility modeling:
   - GARCH(1,1), EGARCH, GJR, FIGARCH
7. Volatility forecast evaluation
8. Comparison vs. benchmark (Simple Exponential Smoothing)

*Note: Python does not include a built‐in TAR/SETAR estimator in core libraries. Below we provide a template for TAR/SETAR that can be implemented with specialized packages or custom code (e.g., by splitting the sample by threshold and fitting separate regressions). For full functionality, you may explore packages like `pystatsmodels-regime` or implement a custom grid search for threshold values.

Assumptions:
- Data file is a CSV named “wti_data.csv” with a Date column and columns: WTIRET, BRENTRET, GASOLINERET, PROPANERET.
- Date format is YYYY-MM-DD. We set Date as index.
- Required packages: pandas, numpy, matplotlib, statsmodels, arch, sklearn.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf, q_stat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan, normal_ad
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel

# 1. LOAD DATA
# ---------------------------------------------------------------------------------
# Read CSV into DataFrame, parse dates, set Date as index.
data = pd.read_csv("wti_data.csv", parse_dates=["Date"])
data.set_index("Date", inplace=True)

# Ensure the required columns exist
required_cols = ["WTIRET", "BRENTRET", "GASOLINERET", "PROPANERET"]
if not all(col in data.columns for col in required_cols):
    raise ValueError(f"Dataframe must contain columns: {required_cols}")

# 2. VISUAL INSPECTION OF THE SERIES
# ---------------------------------------------------------------------------------
# The EViews workflow began by plotting each series to examine mean reversion and volatility clustering (e.g. during 2008).
# Plot WTIRET, BRENTRET, GASOLINERET, PROPANERET
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

series_names = ["WTIRET", "BRENTRET", "GASOLINERET", "PROPANERET"]
for ax, name in zip(axes, series_names):
    data[name].plot(ax=ax, title=f"{name} Time Series")
    ax.axhline(0, color='black', linewidth=0.5, linestyle="--")

plt.tight_layout()
plt.show()

# 3. STATIONARITY TESTING (ADF)
# ---------------------------------------------------------------------------------
# As in the EViews code we verify stationarity using the Augmented Dickey-Fuller test (H0: unit root).
def adf_test(series, signif=0.05, name="Series"):
    """
    Perform Augmented Dickey-Fuller test and print results.
    H0: Unit root (non-stationary)
    Ha: Stationary
    """
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"ADF Test on {name}:")
    print(f"  Test Statistic        = {result[0]:.4f}")
    print(f"  p-value               = {result[1]:.4f}")
    for key, val in result[4].items():
        print(f"     Critical Value({key}) = {val:.4f}")
    if result[1] < signif:
        print(f"  => Reject H0: {name} is stationary.")
    else:
        print(f"  => Fail to reject H0: {name} is non-stationary.")
    print("-" * 60)

adf_test(data["WTIRET"], name="WTI Returns")
adf_test(data["BRENTRET"], name="Brent Returns")
adf_test(data["GASOLINERET"], name="Gasoline Returns")
adf_test(data["PROPANERET"], name="Propane Returns")

# 4. AUTOCORRELATION ANALYSIS FOR WTIRET
# ---------------------------------------------------------------------------------
# Plot ACF and PACF for WTIRET
# EViews found small but significant autocorrelations suggesting an ARMA structure. Here we inspect ACF/PACF and run the Ljung-Box test.
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(data["WTIRET"].dropna(), lags=36, ax=ax[0], title="ACF of WTIRET")
plot_pacf(data["WTIRET"].dropna(), lags=36, ax=ax[1], title="PACF of WTIRET")
plt.tight_layout()
plt.show()

# Ljung-Box Q-test for autocorrelation up to lag 36
lb_test = acorr_ljungbox(data["WTIRET"].dropna(), lags=[36], return_df=True)
print("Ljung-Box test (lag=36) for WTIRET residuals:")
print(lb_test)
print("-" * 60)

# 5. MEAN MODEL ESTIMATION
# ---------------------------------------------------------------------------------

# 5.1 ARMA(2,1) WITH EXOGENOUS VARIABLES (BRENTRET, GASOLINERET, PROPANERET)
#----------------------------------------------------------------------------------
# We choose ARMA(2,1) based on Schwarz information criterion (similar to EViews autoarma with max_ar=2, max_ma=2)
# Fit SARIMAX (ARMA is ARIMA with order=(p,0,q)) with exog.
endog = data["WTIRET"].dropna()
exog = data[["BRENTRET", "GASOLINERET", "PROPANERET"]].loc[endog.index]

# Fit ARIMA(2,0,1) with exogenous regressors
arma_order = (2, 0, 1)
# EViews selected an ARMA(2,1) when searching over AR and MA lags with SIC. We fit the same specification here.
model_arma21 = ARIMA(endog, order=arma_order, exog=exog).fit()
print(model_arma21.summary())

# Diagnose residuals
resid_arma21 = model_arma21.resid.dropna()

# 5.1.1 Ljung-Box on residuals
# The following residual diagnostics mirror those performed in EViews: autocorrelation (Ljung-Box), ARCH effects and normality.
lb_resid = acorr_ljungbox(resid_arma21, lags=[10, 20, 36], return_df=True)
print("Ljung-Box test (lags=10,20,36) on ARMA(2,1) residuals:")
print(lb_resid)

# 5.1.2 ARCH test for heteroskedasticity
arch_test = het_arch(resid_arma21)
print("\nARCH test on ARMA(2,1) residuals:")
print(f"  LM statistic = {arch_test[0]:.4f}, p-value = {arch_test[1]:.4f}")

# 5.1.3 Jarque-Bera test for normality
jb_stat, jb_pvalue, _, _ = jarque_bera(resid_arma21)
print("\nJarque-Bera test on ARMA(2,1) residuals:")
print(f"  JB statistic = {jb_stat:.4f}, p-value = {jb_pvalue:.4f}")
print("-" * 60)

# 5.2 THRESHOLD AUTOREGRESSIVE (TAR) – TEMPLATE
# In EViews we estimated TAR models to allow regime changes driven by an exogenous variable. Python lacks a built-in routine, so we outline the steps conceptually.
# ---------------------------------------------------------------------------------
# Note: Implementing a full TAR requires specialized routines. Below is a conceptual template:
#
# from some_tar_package import TAR  # hypothetical package
#
# # We want to use BRENTRET as threshold variable.
# # Fit TAR with up to, say, 5 regimes (as EViews identified).
# tar_model = TAR(endog, exog=exog, thresh=exog["BRENTRET"], k_regimes=5)
# tar_res = tar_model.fit()
# print(tar_res.summary())
#
# The interpretation of each regime’s coefficients follows the EViews output logic.

print("TAR modeling requires specialized routines or custom implementation. Placeholder provided.")
print("-" * 60)

# 5.3 SETAR – TEMPLATE
# SETAR in EViews used lagged WTIRET as the threshold variable. Here we also provide a high-level template.
# ---------------------------------------------------------------------------------
# Note: SETAR uses lagged WTIRET as threshold. Below is a conceptual template:
#
# from some_tar_package import SETAR  # hypothetical package
#
# setar_model = SETAR(endog, exog=exog, thresh= endog.shift(1), lags=[1,12], k_regimes=4)
# setar_res = setar_model.fit()
# print(setar_res.summary())
#
# Interpretation follows regime‐specific coefficients as in EViews.

print("SETAR modeling requires specialized routines or custom implementation. Placeholder provided.")
print("-" * 60)

# 5.4 MARKOV SWITCHING (MS)
# ---------------------------------------------------------------------------------
# Following the EViews analysis we estimate a two-regime Markov Switching model to capture stochastic regime changes.
# Fit a two-regime Markov Switching regression with exogenous variables, allowing for switching in variance.
# We let “k_regimes=2” and switch='variance' to capture volatility regimes.
# NOTE: statsmodels’ MarkovRegression only allows switching in mean. To allow for switching variance, we can use
# MarkovRegression with 'switching_variance=True' (statsmodels >= 0.12).

ms_model = MarkovRegression(
    endog,
    k_regimes=2,
    exog=exog,
    switching_variance=True,
    switching_trend=False,
    trend="c"
).fit(em_iter=50, search_reps=20)
print(ms_model.summary())

# The transition probabilities and smoothed state probabilities are available:
print("\nTransition Matrix:")
print(ms_model.transition_matrices)

# Following the EViews script, we reserve the last 22 observations for testing and generate 5- and 22-day ahead forecasts.
# 6. FORECASTING MEAN MODELS (5- and 22-DAY HORIZONS)
# ---------------------------------------------------------------------------------
# Reserve last 22 observations for out-of-sample evaluation
horizons = [5, 22]
train_end = -np.max(horizons)
train_data = endog.iloc[:train_end]
train_exog = exog.iloc[:train_end]
test_data = endog.iloc[train_end:]
test_exog = exog.iloc[train_end:]

# 6.1 Benchmark: Simple Exponential Smoothing on WTIRET
# SES serves as the benchmark model used in EViews to check if more complex models add value.
ses_model = ExponentialSmoothing(
    train_data,
    trend=None,
    seasonal=None,
    initialization_method="estimated"
).fit()

# Generate forecasts
ses_forecasts = pd.DataFrame(index=test_data.index, columns=[f"H{h}" for h in horizons])
for h in horizons:
    # Rolling forecast: predict h steps ahead for each point in test_data
    ses_forecasts[f"H{h}"] = ses_model.forecast(steps=len(test_data))

# 6.2 ARMA(2,1) Forecasts
# Forecasts are aligned to the correct horizon following the same approach used in the EViews code.
# We need dynamic forecasting with exogenous variables. We'll use the fitted model_arma21.
arma_forecasts = pd.DataFrame(index=test_data.index, columns=[f"H{h}" for h in horizons])
for h in horizons:
    # Statsmodels ARIMA.forecast can forecast multi-step with exog forward values
    arma_forecasts[f"H{h}"] = model_arma21.get_forecast(
        steps=len(test_data),
        exog=test_exog
    ).predicted_mean.shift(h - 1)  # Align to horizon

# 6.3 TAR, SETAR Forecasts – Placeholders
# In the EViews workflow we compared TAR and SETAR forecasts with ARMA. Here we create placeholder DataFrames for those models.
tar_forecasts = pd.DataFrame(index=test_data.index, columns=[f"H{h}" for h in horizons])
setar_forecasts = pd.DataFrame(index=test_data.index, columns=[f"H{h}" for h in horizons])
print("TAR/SETAR forecasts require custom code. Placeholders created.")

# 6.4 Markov Switching Forecasts
# The statsmodels MarkovRegression object does not directly provide multi-step forecasts.
# One approach: simulate future states or use the last smoothed state to generate mean forecasts.
# Placeholder MS forecasts follow the same spirit as the EViews code but require simulation logic.
# Here we provide a placeholder for MS forecasts:

ms_forecasts = pd.DataFrame(index=test_data.index, columns=[f"H{h}" for h in horizons])
print("Markov-Switching forecasting requires custom logic (e.g., Monte Carlo sim, filtering). Placeholder created.")

# 7. FORECAST EVALUATION FOR MEAN MODELS
# ---------------------------------------------------------------------------------
# Metrics mirror the univariate loss functions used in the EViews project.
def compute_loss_metrics(actual, forecast):
    """
    Compute a dictionary of loss metrics between actual and forecast series:
    - MSE, MAE, RMSE, MAPE, SMAPE, Theil's U
    """
    mask = ~np.isnan(forecast)
    y_true = actual[mask]
    y_pred = forecast[mask]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)).replace(0, np.nan)) * 100
    # Theil's U: sqrt( sum((f_t - y_t)^2) / sum(y_t^2) )
    u = np.sqrt(np.sum((y_pred - y_true) ** 2) / np.sum(y_true ** 2))

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "SMAPE(%)": smape,
        "Theil_U": u
    }

# Example: Evaluate ARMA(2,1) vs SES at 5-day horizon
h = 5
actual_5 = test_data.shift(- (h - 1))  # Align actual to 5-day ahead
mse_arma5 = compute_loss_metrics(actual_5, arma_forecasts[f"H{h}"])
mse_ses5 = compute_loss_metrics(actual_5, ses_forecasts[f"H{h}"])

print(f"Mean Model Loss (5-day horizon):\n ARMA21: {mse_arma5}\n SES: {mse_ses5}")

# Diebold-Mariano test for paired forecast errors (5-day horizon)
# Same hypothesis tests as in the EViews analysis are implemented via the Diebold-Mariano statistic.
def diebold_mariano(e1, e2, h=1, alternative="two-sided"):
    """
    Diebold-Mariano test statistic for two series of forecast errors.
    e1, e2: arrays of forecast errors (actual - forecast)
    h: forecast horizon
    """
    d = e1**2 - e2**2  # loss differential using squared errors (MSE losses)
    d = d.dropna()
    T = len(d)
    # Newey-West estimator for variance of d with lag h-1
    from statsmodels.stats.stattools import acovf
    gamma = acovf(d, fft=False, missing='drop')  # autocovariances
    # Compute Newey-West variance: var = (gamma[0] + 2*sum_{i=1 to h-1} gamma[i]) / T
    var_dm = (gamma[0] + 2 * np.sum(gamma[1:h])) / T
    dm_stat = np.mean(d) / np.sqrt(var_dm)
    # Two-sided p-value
    from scipy.stats import t
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=T - 1))
    return dm_stat, p_value

# Compute forecast errors
e_arma5 = actual_5 - arma_forecasts[f"H{h}"]
e_ses5 = actual_5 - ses_forecasts[f"H{h}"]
dm_stat5, dm_pval5 = diebold_mariano(e_arma5, e_ses5, h=h)
print(f"Diebold-Mariano test (5-day): DM stat={dm_stat5:.4f}, p-value={dm_pval5:.4f}")

# Repeat similarly for 22-day horizon and other model pairs (omitted for brevity)

# 8. VOLATILITY MODELING
# After modelling the mean, the EViews script estimated various GARCH-type models to capture volatility clustering and leverage effects.
# ---------------------------------------------------------------------------------
# We estimate volatility models on WTIRET residuals or directly on WTIRET (since mean ~ 0).
# Common practice: Fit on returns series, with mean model=constant or exog if desired.
vol_data = data["WTIRET"].dropna()

# 8.1 GARCH(1,1)
garch11 = arch_model(vol_data, x=None, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
res_garch11 = garch11.fit(update_freq=10)
print(res_garch11.summary())
# EGARCH explicitly models leverage effects similar to the EViews approach.

# 8.2 EGARCH
egarch = arch_model(vol_data, x=None, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t")
res_egarch = egarch.fit(update_freq=10)
print(res_egarch.summary())
# GJR is another asymmetric specification used in the EViews comparison.

# 8.3 GJR-GARCH (Threshold GARCH)
gjr = arch_model(vol_data, x=None, mean="Constant", vol="GARCH", p=1, o=1, q=1, power=2.0, dist="t")
res_gjr = gjr.fit(update_freq=10)
print(res_gjr.summary())

# FIGARCH captures long memory effects as analysed in the EViews file.
# 8.4 FIGARCH
# The arch package supports FIGARCH. We set p=1, q=1, and fractional differencing parameter.
figarch = arch_model(vol_data, x=None, mean="Constant", vol="FIGARCH", p=1, q=1, dist="t")
res_figarch = figarch.fit(update_freq=10)
print(res_figarch.summary())

# Note: GARCH-MIDAS omitted due to lack of low-frequency data.

# 9. VOLATILITY FORECASTS (5- and 22-DAY AHEAD)
# ---------------------------------------------------------------------------------
# Use the “forecast” method to generate h-step ahead forecasts of conditional variance.
# arch_model.forecast returns a dict-like with horizon h; align index to test periods manually.

def get_vol_forecasts(res_model, start, horizon):
    """
    Generate volatility forecasts from a fitted arch model.
    - res_model: fitted arch model result
    - start: index label where forecasts begin
    - horizon: integer, max horizon
    """
    # Dynamic forecasts from "start" onward
    # We'll use res_model.forecast() specifying horizon
    fore = res_model.forecast(start=start, horizon=horizon, method="simulation")
    # The forecasted variances are in fore.variance: a DataFrame indexed by date, columns=1..horizon
    return fore.variance

# Choose start date for forecast: the first date of test_data
start_date = test_data.index[0]
h = 22  # max horizon
vol_forecasts = {
    "GARCH": get_vol_forecasts(res_garch11, start_date, h),
    "EGARCH": get_vol_forecasts(res_egarch, start_date, h),
    "GJR": get_vol_forecasts(res_gjr, start_date, h),
    "FIGARCH": get_vol_forecasts(res_figarch, start_date, h)
}

# Align 5- and 22-day ahead variance forecasts
vol_fc_5 = {name: vol_forecasts[name].loc[:, 5].reindex(test_data.index) for name in vol_forecasts}
vol_fc_22 = {name: vol_forecasts[name].loc[:, 22].reindex(test_data.index) for name in vol_forecasts}

# 10. VOLATILITY FORECAST EVALUATION
# As in the EViews script, we evaluate volatility forecasts using MSE, MAE and RMSE at 5- and 22-day horizons.
# ---------------------------------------------------------------------------------
# Define loss metrics for volatility (compare squared returns vs. forecasted variance)
actual_var = vol_data.iloc[train_end + (h - 1):] ** 2  # squared returns as proxy for realized variance

def eval_vol_loss(actual_var, fc_var_dict, horizon):
    """
    Compare volatility forecasts (forecasted variance) with actual realized variance (squared returns)
    using MSE, MAE, etc.
    Returns a DataFrame summarizing loss metrics for each model.
    """
    metrics = {}
    for name, fc in fc_var_dict.items():
        # Align actual_var to forecast index
        aligned_actual = actual_var.shift(-(horizon - 1)).reindex(fc.index)
        # Compute MSE, MAE between forecasted var (fc) and actual_var (aligned_actual)
        mse = mean_squared_error(aligned_actual.dropna(), fc.dropna())
        mae = mean_absolute_error(aligned_actual.dropna(), fc.dropna())
        rmse = np.sqrt(mse)
        metrics[name] = {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse
        }
    return pd.DataFrame(metrics).T

vol_loss_5 = eval_vol_loss(vol_data.iloc[:], vol_fc_5, horizon=5)
vol_loss_22 = eval_vol_loss(vol_data.iloc[:], vol_fc_22, horizon=22)

print("\nVolatility Forecast Loss (5-day horizon):")
print(vol_loss_5)

print("\nVolatility Forecast Loss (22-day horizon):")
print(vol_loss_22)

# Diebold-Mariano tests between GARCH variants (example: EGARCH vs GARCH for 5-day horizon)
e_garch5 = actual_var.shift(-(5 - 1)) - vol_fc_5["EGARCH"]
e_garch11_5 = actual_var.shift(-(5 - 1)) - vol_fc_5["GARCH"]
# Perform Diebold-Mariano tests between volatility models as done in the EViews exercise.
dm_stat_vol5, dm_pval_vol5 = diebold_mariano(e_garch5, e_garch11_5, h=5)
print(f"\nDiebold-Mariano (Volatility, 5-day) EGARCH vs GARCH: DM stat={dm_stat_vol5:.4f}, p-value={dm_pval_vol5:.4f}")

# 11. SUMMARY AND MODEL SELECTION
# ---------------------------------------------------------------------------------
# Based on mean and volatility evaluation, compare across:
# These comments mirror the concluding discussion in the EViews file where models were ranked by forecast accuracy.
# - Mean models: ARMA(2,1), TAR, SETAR, MS, SES benchmark
# - Volatility models: GARCH, EGARCH, GJR, FIGARCH

# In practice:
# - Compare loss metrics at 5- and 22-day horizons
# - Check Diebold-Mariano p-values for significant differences
# - Choose best-performing models (e.g., MS for mean, GJR for volatility) or combinations

# END OF SCRIPT
