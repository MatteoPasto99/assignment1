import pandas as pd
from numpy.linalg import solve
import numpy as np

# Load the dataset

df = pd.read_csv("current.csv")
df

# Clean the DataFrame by removing the row with transformation codes (first row)
df_cleaned = df.drop(index=0)
df_cleaned
df_cleaned.reset_index(
    drop=True, inplace=True
)  # we eliminated the first row and restored the previous indexation of the remaining rows.
# drop and inplace are 'functional' to set the index
# drop=False	Mantiene il vecchio indice come colonna
# drop=True	Rimuove completamente il vecchio indice
# inplace=False	Ritorna una nuova copia del DataFrame
# inplace=True	Modifica il DataFrame originale
df_cleaned["sasdate"] = pd.to_datetime(df_cleaned["sasdate"], format="%m/%d/%Y")
df_cleaned
# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes

# df.iloc[0, 1:]
# iloc[0, 1:] utilizza index location (iloc) per selezionare dati dal DataFrame.
# 0 → Seleziona la prima riga (0 è l'indice della prima riga).
# 1: → Seleziona tutte le colonne a partire dalla seconda colonna (evita la prima).
# .to_frame() Converte la Serie Pandas ottenuta dalla selezione precedente in un DataFrame.
# .reset_index() Resetta l'indice della Serie trasformata in DataFrame, riportando l'indice originale come colonna separata.

transformation_codes.columns = [
    "Series",
    "Transformation_Code",
]  # we have renamed the cols
transformation_codes

# The transformation codes map variables to the transformations we must
# #apply to each variable to render them (approximately) stationary.
# The data frame transformation_codes has the variable’s name (Series)
# and its transformation (Transformation_Code). There are six possible
# transformations ( denotes the variable to which the transformation is
# to be applied):
## transformation_codes contains the transformation codes
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$


## if condizione1:
#   # Se condizione1 è vera, esegui questo blocco di codice
# elif condizione2:
#   # Se condizione1 è falsa ma condizione2 è vera, esegui questo blocco
# else:
#   # Se nessuna delle condizioni precedenti è vera, esegui questo blocco
# elif permette di aggiungere più condizioni senza nidificare più if, rendendo il codice più leggibile.


# # Function to apply transformations based on the transformation code
def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")


# Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(
        df_cleaned[series_name].astype(float), float(code)
    )

# Scorre ogni coppia (series_name, code) nella variabile transformation_codes.
# Converte la colonna df_cleaned[series_name] in float (per evitare errori con stringhe o interi).
# Applica la funzione apply_transformation alla colonna con il codice di trasformazione corrispondente.

# Sovrascrive df_cleaned[series_name] con i nuovi valori trasformati.
df_cleaned = df_cleaned[
    2:
]  # Since some transformations induce missing values, we drop the first two observations of the dataset
df_cleaned.reset_index(
    drop=True, inplace=True
)  # We reset the index so that the first observation of the dataset has index 0

df_cleaned.head()

############################################################################################################
## Plot transformed series
############################################################################################################
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

series_to_plot = ["INDPRO", "CPIAUCSL", "TB3MS"]
series_names = [
    "Industrial Production",
    "Inflation (CPI)",
    "3-month Treasury Bill rate",
]  # We consider three series (INDPRO, CPIAUCSL, TB3MS) and assign them human-readable names (“Industrial Production”, “Inflation (CPI)”, “3-month Treasury Bill rate.”).

# Create a figure and a grid of subplots
fig, axs = plt.subplots(
    len(series_to_plot), 1, figsize=(8, 15)
)  # We create a figure with three (len(series_to_plot)) subplots arranged vertically. The figure size is 8x15 inches.

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if (
        series_name in df_cleaned.columns
    ):  # We check if the series exists in each series df_cleaned DataFrame columns.
        dates = pd.to_datetime(
            df_cleaned["sasdate"], format="%m/%d/%Y"
        )  # We convert the sasdate column to datetime format (not necessary, since sasdate was converter earlier)
        ax.plot(
            dates, df_cleaned[series_name], label=plot_title
        )  # We plot each series against the sasdate on the corresponding subplot, labeling the plot with its human-readable name.
        ax.xaxis.set_major_locator(
            mdates.YearLocator(base=5)
        )  # We format the x-axis to display ticks and label the x-axis with dates taken every five years.
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(
            plot_title
        )  # Each subplot is titled with the name of the economic indicator.
        ax.set_xlabel(
            "Year"
        )  #  We label the x-axis “Year,” and the y-axis “Transformed Value,” to indicate that the data was transformed before plotting.
        ax.set_ylabel("Transformed Value")
        ax.legend(
            loc="upper left"
        )  # A legend is added to the upper left of each subplot for clarity.
        plt.setp(
            ax.xaxis.get_majorticklabels(), rotation=45, ha="right"
        )  # We rotate the x-axis labels by 45 degrees to prevent overlap and improve legibility.
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout()  # plt.tight_layout() automatically adjusts subplot parameters to give specified padding and avoid overlap.
plt.show()  # plt.show() displays the figure with its subplots.


############################################################################################################
## Create y and X for estimation of parameters
############################################################################################################
## Forecasting in Time Series
## Forecasting in time series analysis involves using historical data to predict future values. The objective is to model the conditional expectation of a time series based on past observations.

### Direct Forecasts
## Direct forecasting involves modeling the target variable directly at the desired forecast horizon. Unlike iterative approaches, which forecast one step ahead and then use those forecasts as inputs for subsequent steps,
## direct forecasting directly models the relationship between past observations and future value.

### ARX Models
## Autoregressive Moving with predictors (ARX) models are a class of univariate time series models that extend ARMA models by incorporating exogenous (independent) variables. These models are formulated as follows:
## Y_(t+h) = alpha + Phi_0 * Y_(t) + Phi_1 * Y_(t-1) + ... + Phi_(p) * Y_(t-p) +
## Theta_(0,1) * X_(t,1) + Theta_(1,1) * X_(t-1,1) + ... + Theta_(p,1) * X_(t-p,1) + ... +
## Theta_(0,k) * X_(t,k) + ... + Theta_(p,k) * X_(t-p, k) + u_(t+h) =
## = alpha + sum_{i=0}^{p} [ phi_(i) * Y_(t-i) ] + sum_{j=1}^{k} sum_{s=0}^{p} [theta_(s,j) * X_(t-s,j)+ epsilon_(t+h)
##
## - Y_(t+h): The target variable at time (t+h).
## - X_(t,j): Predictors (variable j=(1,...,k) at time (t)).
## - p number of lags of the target and the predictors. NB: Theoretically, the number of lags for the target variables and the predictors could be different. Here, we consider the simpler case in which both are equal.
## - phi_(i), i=(0,...,p), and theta_(j,s), j=(1,...,k), s=(1,...,r): Parameters of the model.
## - epsilon_(t+h): error term.
##
## For instance, to predict Industrial Prediction using as predictor inflation and the 3-month t-bill, the target variable is INDPRO, and the predictors are CPIAUSL and TB3MS.
## Notice that the target and the predictors are the transformed variables. Thus, if we use INDPRO as the target, we are predicting the log-difference of industrial production,
## which is a good approximation for its month-to-month percentage change.
## By convention, the data ranges from t=(1,...,T), where T is the last period, we have data (for the df_cleaned dataset, T corresponds to January 2024).
##
### Forecasting with ARX
## Suppose that we know the parameters of the model for the moment. To obtain a forecast for Y_(T+h), the h-step ahead forecast, we calculate
## hat{Y}_(T+h) = alpha + phi_0 * Y_T + phi_1 * Y_(T-1) + ... + phi_p * Y_(T-p) +
## theta_(0,1) * X_(T,1) + theta_(1,1) * X_(T-1,1) + ... + theta_(p,1) * X_(T-p,1) +
## ... + theta_(0,k) * X_(T,k) + ... + theta_(p,k) * X_(T-p,k) =
## alpha + sum_{i=0}^p [ phi_(i) * Y_(T-i) + sum_{j=1}^k sum_{s=0}^p [ theta_(s,j) * X_(T-s,j) ]
##
## While this is conceptually easy, implementing the steps needed to calculate the forecast is insidious, and care must be taken to ensure we are calculating the correct forecast.
## To start, it is convenient to rewrite the model in @eq-model as a linear model
## The following equation is in matricial form:   (vector)Y = (metrix)X ** (vector)beta + (vector)u
##
## The size of beta is (1 + (1+p) + (1+k)) x 1, it contains alpha, Phi_(p) (p = 0,...,p), Theta(p,k) (k = 1,...,k)
## The size of y is (T - p - h) x 1
## The size of X is (T - p - h) x (1 + (1+k) + (1+p))
##
## The matrix X  can be obtained in the following way:

Yraw = df_cleaned["INDPRO"]
Xraw = df_cleaned[["CPIAUCSL", "TB3MS"]]

## Number of lags and leads
num_lags = 4  ## this is p, the model will use the last 4 months of INDPRO, CPIAUCSL and TB3MS as predictors
num_leads = 1  ## this is h, we are predicting one step ahead (next months INDPRO)

X = (
    pd.DataFrame()
)  # Creates an empty DataFrame X where we will store lagged values of INDPRO, CPIAUCSL, and TB3MS.

## Add the lagged values of Y (INDPRO)
col = "INDPRO"
for lag in range(0, num_lags + 1):
    # Shift each column in the DataFrame and name it with a lag suffix
    X[f"{col}_lag{lag}"] = Yraw.shift(
        lag
    )  # This creates lagged columns for INDPRO (Y), meaning that each row will contain values of INDPRO from previous months.
    # The .shift(lag) function moves the values down by lag steps, so the value at time t appears in row t+lag.
## Add the lagged values of X (CPIAUCSL and TB3MS)
for col in Xraw.columns:
    for lag in range(0, num_lags + 1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f"{col}_lag{lag}"] = Xraw[col].shift(
            lag
        )  # Similar to INDPRO, this loop shifts each predictor variable (CPIAUCSL, TB3MS) by lag periods.
## Add a column on ones (for the intercept)
X.insert(
    0, "Ones", np.ones(len(X))
)  # This is needed because in regression models, the intercept (alpha) is modeled as a separate coefficient.

## X is now a DataFrame with lagged values of Y and X
X.head()  # Note that the first 4 rows of X have missing values.

## The vector y  can be similarly created as:
## Y is now the leaded target variable
y = Yraw.shift(
    -num_leads
)  # The .shift(-1) moves values up by num_leads (predicting the next value).
y  # This creates y such that for each row at time t, we have the INDPRO value at time t+1.

## The variable y has missing values in the last h positions (it is not possible to lead the target beyond T).
## Notice also that we must keep the last row of X for constructing the forecast.


############################################################################################################
## Estimation and forecast
############################################################################################################
## Now we create two numpy arrays with the missing values stripped:
## Save last row of X (converted to numpy), This row corresponds to the most recent available data, which we will use for forecasting.
X_T = X.iloc[-1:].values

## Subset getting only rows of X and y from p+1 to h-1 and convert to numpy array
y = y.iloc[
    num_lags:-num_leads
].values  # y.iloc[num_lags:-num_leads]: Removes the first num_lags rows (which have NaN due to shifting) and the last num_leads rows (NaN due to shifting forward).
X = X.iloc[
    num_lags:-num_leads
].values  # X.iloc[num_lags:-num_leads]: Removes the first num_lags rows to align X with y.

X_T

## Now, we have to estimate the parameters and obtain the forecast.
##
## ESTIMATION
## The parameters of the model can be estimated by OLS (the OLS estimates the coefficient of the linear projection of Y_(t+h) on its lags and the lags of X_t).
## The OLS Estimator of Beta is: beta_hat = (X'X)^(-1) X'Y
## While this is the formula used to describe the OLS estimator, from a computational point of view is much better to define the estimator as the solution
## of the set of linear equations: (X'X) beta = X'Y
## The function solve can be used to solve this linear system of equation.
## Import the solve function from numpy.linalg
from numpy.linalg import solve

# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(
    X.T @ X, X.T @ y
)  # solve(A, B) efficiently computes 𝐴^(-1)𝐵 without explicitly computing 𝐴^(-1), which improves numerical stability.
beta_ols
## Produce the One step ahead forecast
## % change month-to-month of INDPRO
forecast = X_T @ beta_ols * 100  # The @ operator performs matrix multiplication.
forecast  # * 100 converts the forecast into percentage change month-to-month

## The variable forecast contains now the one-step ahead (h=1 forecast) of INDPRO.
## Since INDPRO has been transformed in logarithmic differences, we are forecasting the percentage change (and multiplying by 100 gives the forecast in percentage points).
## To obtain the h-step ahead forecast, we must repeat all the above steps using a different h.

############################################################################################################
## Forecasting Exercise - Real-Time Valuation
#############################################################################################################
## How good is the forecast that the model is producing? One thing we could do to assess the forecast’s quality is to wait for the new data on industrial production and see how big the forecasting error is.
## However, this evaluation would not be appropriate because we need to evaluate the forecast as if it were repeatedly used to forecast future values of the target variables.
## To properly assess the model and its ability to forecast INDPRO, we must keep producing forecasts and calculating the errors as new data arrive.
## This procedure would take time as we must wait for many months to have a series of errors that is large enough.
## A different approach is to do what is called a Real-time evaluation.
## A Real-time evaluation procedure consists of putting ourselves in the shoes of a forecaster who has been using the forecasting model for a long time.
##
## In practice, that is what are the steps to follow to do a Real-time evaluation of the model:
## 0. Set T such that the last observation of df coincides with December 1999;
## 1. Estimate the model using the data up to T
## 2. Produce hat_Y_(T+1),..., hat_Y_(T+H)
## 3. Since we have the actual data for January, February, …, we can calculate the forecasting errors of our model
##    hat_e_(T+h) = hat_Y_(T+h) - Y_(T+h) , h=(1,...,H)
## 4. Set T = T+1 and do all the steps above
## The process results are a series of forecasting errors we can evaluate using several metrics. The most commonly used is the MSFE, which is defined as:
## MSFE_h = (1/J) * sum_{j=1}^J hat_e^2_(T+j+h)
## where J is the number of errors we collected through our real-time evaluation.
## This assignment asks you to perform a real-time evaluation assessment of our simple forecasting model and calculate the MSFE for steps h = 1,4,8
## As a bonus, we can evaluate different models and see how they perform differently. For instance, you might consider different numbers of lags and/or different variables in the model.


## Hint
## A sensible way to structure the code for real-time evaluation is to use several functions. For instance, you can define a function that calculates the forecast given the DataFrame.
## def calculate_forescast(df, target_var, predictors, num_lags, horizons)
##    - df: Pandas DataFrame containing the dataset.
##    - target_var: Name of the target variable (e.g., 'INDPRO').
##    - predictors: List of predictor variables (e.g., ['CPIAUCSL', 'TB3MS']).
##    - num_lags: Number of lags to include in the model.
##    - horizons: List of forecast horizons (e.g., [1, 4, 8]).
##    - end_date: Extract the dataset up to a given date ('12/1/1999')
##
## This function calculate_forecast is structured to:
##    - Extract the dataset up to a given date (end_date) to simulate real-time forecasting.
##    - Generate lagged features for both the target variable (INDPRO) and predictors (CPIAUCSL, TB3MS).
##    - Train an OLS model using historical data up to end_date.
##    - Generate forecasts for multiple steps ahead (H = [1,4,8]).
##    - Calculate forecast errors by comparing the forecasts to the actual observed values.
def calculate_forecast(
    df_cleaned,
    p=4,
    H=[1, 4, 8],
    end_date="12/1/1999",
    target="INDPRO",
    xvars=["CPIAUCSL", "TB3MS"],
):

    ## Subset df_cleaned to use only data up to end_date
    rt_df = df_cleaned[
        df_cleaned["sasdate"] <= pd.Timestamp(end_date)
    ]  # This filters df_cleaned to include only data up to end_date.
    # It simulates a real-time scenario where we don’t use future data for training.
    ## Get the actual values of target (INDPRO) at different steps ahead, for comparison
    Y_actual = (
        []
    )  # This extracts the actual future values of INDPRO for each forecast horizon H = [1,4,8].
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(
            df_cleaned[df_cleaned["sasdate"] == os][target] * 100
        )  # The values are multiplied by 100 to express them as percentage changes.
        ## Now Y contains the true values at T+H (multiplying * 100)               # These actual values will later be compared with our model’s forecasts.

    ## Create the Lagged Variables for X and Y
    Yraw = rt_df[target]
    Xraw = rt_df[
        xvars
    ]  # Extracts the target variable (INDPRO) and the predictor variables (CPIAUCSL, TB3MS) from rt_df.

    X = pd.DataFrame()  # Initializes an empty DataFrame X to store lagged features.

    ## Add the lagged target values of Y
    for lag in range(
        0, p
    ):  # Creates p lagged versions of INDPRO to serve as autoregressive predictors.

        # Shift each column in the DataFrame and name it with a lag suffix
        X[f"{target}_lag{lag}"] = Yraw.shift(lag)
    ## Add Lagged Predictors
    for col in Xraw.columns:  # Creates p lagged versions of each predictor.
        for lag in range(0, p):
            X[f"{col}_lag{lag}"] = Xraw[col].shift(lag)

    ## Add a column on ones (for the intercept)
    X.insert(0, "Ones", np.ones(len(X)))

    ## Save last row of X (converted to numpy)
    X_T = X.iloc[
        -1:
    ].values  # Extracts the last available row of X to use as input for future forecasts.

    ## Compute the forescast for different horizons
    ## While the X will be the same, Y needs to be leaded differently
    Yhat = []  # Loops over forecast horizons (H = [1, 4, 8]).
    for h in H:
        y_h = Yraw.shift(
            -h
        )  # Yraw.shift(-h): Moves the target variable h steps ahead, aligning it for training.

        ## Subset getting only rows of X and y from p+1 to h-1
        y = y_h.iloc[
            p:-h
        ].values  # Removes missing values to ensure proper alignment between X and y.
        X_ = X.iloc[
            p:-h
        ].values  # Ensures X_ and y contain only valid historical data for training.

        # Solving for the OLS estimator beta: (X'X)^{-1} X'Y. Estimate the model with OLS
        beta_ols = solve(X_.T @ X_, X_.T @ y)

        ## Produce the One step ahead forecast
        ## % change month-to-month INDPRO
        Yhat.append(
            X_T @ beta_ols * 100
        )  # Forecast for each h. Uses the latest available data (X_T) to make a forecast.

    ## Now calculate the forecasting error and return the errors for h=(1,4,8)
    return np.array(Y_actual) - np.array(Yhat)


## With this function, you can calculate real-time errors by looping over the end_date to ensure you end the loop at the right time.

t0 = pd.Timestamp(
    "12/1/1999"
)  # Defines the initial time point for evaluation (t0 = December 1999). This is where the rolling evaluation begins.
e = []  # e: Stores forecast errors for each time step.
T = []  # T: Stores the corresponding timestamps.

# Rolling Forecast Loop (Real-Time Evaluation)
for j in range(
    0, 10
):  # Runs the evaluation for 10 months, rolling forward one step at a time.

    # Advance the time window
    t0 = t0 + pd.DateOffset(months=1)  # Moves t0 one month ahead at each iteration.
    # Print the Current Evaluation Date
    print(f"Using data up to {t0}")  # Displays the current cutoff date for forecasting
    # Generate Forecast Errors
    ehat = calculate_forecast(
        df_cleaned, p=4, H=[1, 4, 8], end_date=t0
    )  # Uses only data up to t0. Estimates an ARX model. Produces forecast errors for horizons [1, 4, 8] months.
    # Store the Forecast Errors
    e.append(
        ehat.flatten()
    )  # Flattens ehat (which is an array) into a single row and appends it to e.
    T.append(t0)  # Saves t0 in T, tracking the date for each error.


## Convert Error List (e) into a Pandas DataFrame
## Create a pandas DataFrame from the list
edf = pd.DataFrame(e)  # Each row represents errors for a different month.
# Each column corresponds to a different forecast horizon (h = 1, 4, 8).
## Calculate the RMSFE, that is, the square root of the MSFE
np.sqrt(
    edf.apply(np.square).mean()
)  # Squares all errors in edf → gives MSFE (Mean Squared Forecast Error).
# Takes the mean across rows → computes average MSFE for each horizon.
# Takes the square root → converts MSFE to RMSFE.

## You may change the function calculate_forecast to output also the actual data end the forecast, so you can, for instance, construct a plot.
