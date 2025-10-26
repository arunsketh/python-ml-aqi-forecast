"""
ML_Assesement

This script builds a machine learning pipeline to forecast the 7-day
Air Quality Index (AQI) for multiple cities.

Methodology:
1.  Data Loading & Cleaning: Load daily AQI data and geographical data.
    Impute missing AQI values using a city-specific mean.
2.  Feature Engineering: Create lag features (e.g., AQI 1 day ago) and
    rolling 7-day averages to capture recent trends. Create 7 target
    columns (target_day_1...target_day_7) for the 7-day forecast.
3.  Model Training: Train multiple regression models
    (RandomForest, Polynomial Regression, ANN) using MultiOutputRegressor
    to predict all 7 targets simultaneously.
4.  Model Evaluation: Compare models using RMSE, MAE, and R2 metrics on an 80-20 train-test split.
5.  Prediction & Visualization: Use the best-performing model to
    generate a 7-day forecast for all cities and visualize the
    results in a heatmap.
"""

# ===================================================================
# === 1. Import Libraries
# ===================================================================

# Core libraries
import os  # For interacting with the operating system (e.g., file paths)
import warnings  # To control warning messages
from datetime import datetime, timedelta  # For manipulating date and time objects

# Data handling
import joblib  # For saving and loading trained scikit-learn models
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis (DataFrames)

# Modeling
from sklearn.multioutput import MultiOutputRegressor  # Wrapper for models that don't natively support multi-target prediction
from sklearn.linear_model import LinearRegression  # Baseline linear regression model
from sklearn.ensemble import RandomForestRegressor  # Ensemble model (decision trees)
from sklearn.preprocessing import PolynomialFeatures  # For creating polynomial features
from sklearn.pipeline import Pipeline  # For chaining multiple processing steps (e.g., features + model)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error  # Evaluation metrics
from sklearn.model_selection import ParameterGrid, train_test_split  # For hyperparameter tuning and splitting data

# Deep Learning
import tensorflow as tf  # Main library for deep learning
from tensorflow import keras  # High-level API for building and training models in TensorFlow

# Visualization
import matplotlib.pyplot as plt  # For creating static plots
import seaborn as sns  # For creating statistical visualizations (like heatmaps)
import altair as alt  # For creating interactive visualizations


# ===================================================================
# === 2. Function Definitions
# ===================================================================


def load_geog_data(path: str) -> pd.DataFrame | None:
    """
    Load and clean city geographical dataset.

    Steps:
    1) Select City, lat, lng, population
    2) Clean commas from population strings and coerce to numeric
    3) Drop rows with missing values

    Args:
        path (str): The file path to the geographical data CSV.

    Returns:
        pd.DataFrame | None: A cleaned DataFrame with city geo-data, or None if loading fails.
    """
    try:
        # Check if the file exists at the given path
        if not os.path.exists(path):
            print(f"❌ Error: File not found at path: {path}")
            print("📂 Please check your Google Drive path and file name.")
            return None

        # Specify columns to load for efficiency
        geog_cols = ['City', 'lat', 'lng', 'population']
        # Read the CSV, only loading specified columns
        df_geog = pd.read_csv(path, usecols=geog_cols)

        # Clean 'population' column: remove commas
        df_geog['population'] = df_geog['population'].replace(',', '', regex=True)
        # ...and convert to numeric, coercing errors (non-numeric values) to NaN
        df_geog['population'] = pd.to_numeric(df_geog['population'], errors='coerce')

        # Drop rows with any missing values (lat, lng, or population)
        df_geog = df_geog.dropna()

        print(f"🌍 Geographical data loaded and cleaned. {len(df_geog)} cities processed.")
        return df_geog

    except FileNotFoundError:
        # Handle file not found error
        print(f"❌ Error: Geographical data file not found at {path}")
        return None
    except Exception as e:
        # Handle other potential errors during loading
        print(f"❌ An error occurred while loading geographical data: {e}")
        return None


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers time-series features and future targets from the AQI data.

    Features created:
    - Day_of_year: Captures seasonality.
    - Lag features (1, 7, 14 days): AQI from previous days.
    - Rolling mean features (7, 14 days): Average AQI over past windows.

    Targets created:
    - target_day_1 ... target_day_7: AQI for the next 7 days.

    Args:
        df (pd.DataFrame): The input DataFrame with 'City', 'Date', and 'AQI' columns.

    Returns:
        pd.DataFrame: A new DataFrame with engineered features and target columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_feat = df.copy()

    # 1. Time feature: captures seasonality
    df_feat['Day_of_year'] = df_feat['Date'].dt.dayofyear

    # Group by city to calculate temporal features independently for each city
    # This prevents data from one city leaking into the features of another
    grouped = df_feat.groupby('City')['AQI']

    # 2. Lag features: AQI from previous days
    df_feat['AQI_lag_1'] = grouped.shift(1)  # AQI from 1 day ago
    df_feat['AQI_lag_7'] = grouped.shift(7)  # AQI from 7 days ago
    df_feat['AQI_lag_14'] = grouped.shift(14) # AQI from 14 days ago

    # 3. Rolling mean features: average AQI over past windows
    # .shift(1) is used to prevent data leakage (ensures mean is calculated
    # using data *before* the current day)
    df_feat['AQI_roll_mean_7'] = grouped.shift(1).rolling(window=7, min_periods=1).mean()
    df_feat['AQI_roll_mean_14'] = grouped.shift(1).rolling(window=14, min_periods=1).mean()

    # 4. Target features: AQI for the *next* 7 days
    # This shifts future data backwards to align with the current day's features
    # For a given row (day D), 'target_day_1' will be the AQI from day D+1
    for i in range(1, 8):
        df_feat[f'target_day_{i}'] = grouped.shift(-i)

    return df_feat


def get_aqi_bucket(aqi: float) -> str:
    """
    Converts a numerical AQI value to its qualitative category.

    Args:
        aqi (float): The numerical Air Quality Index value.

    Returns:
        str: The corresponding qualitative category (e.g., "Good", "Moderate").
    """
    # Standard AQI category definitions
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def predict_next_7_days_aqi_geo(city_name: str, model_list, df_geog: pd.DataFrame, full_data: pd.DataFrame):
    """
    Generates and PRINTS a 7-day forecast for a specific city.
    This function performs autoregressive forecasting.

    Args:
        city_name (str): The name of the city to forecast.
        model_list (list): The list of 7 trained model estimators (one for each day).
        df_geog (pd.DataFrame): The cleaned dataframe with city geo-data.
        full_data (pd.DataFrame): The full, imputed AQI dataframe.

    Returns:
        None: This function prints the forecast to the console but does not return a value.
    """
    print(f"\n--- 🏙️ Generating 7-Day Forecast for: {city_name} ---")

    # 1. Get static geographical features for the city
    try:
        city_geo_data = df_geog.loc[df_geog['City'] == city_name].iloc[0]
        lat = city_geo_data['lat']
        lng = city_geo_data['lng']
        population = city_geo_data['population']
    except IndexError:
        # Handle case where city is not in the geo-data
        print(f"❌ Error: Geographical data not found for city: {city_name}")
        return

    # 2. Get historical AQI data for the city
    city_data = full_data[full_data['City'] == city_name].sort_values(by='Date')
    if city_data.empty:
        # Handle case where city has no AQI data
        print(f"❌ Error: No historical AQI data found for city: {city_name}")
        return

    # Get the last known day of data
    last_known_day = city_data.iloc[-1]
    last_known_date = last_known_day['Date']

    # 3. Initialize prediction loop
    # Get the last 14 days of AQI to build features for the first prediction
    recent_aqi = list(city_data['AQI'].iloc[-14:])
    predicted_aqi_7_days = []

    # Loop to predict 7 days, one by one (autoregressive forecasting)
    for i in range(1, 8):
        # 4. Create features for the future day
        future_date = last_known_date + timedelta(days=i)
        day_of_year = future_date.dayofyear

        # Build lag and roll features from the 'recent_aqi' list
        # This list includes both historical data and previous predictions
        aqi_lag_1 = recent_aqi[-1]
        aqi_lag_7 = recent_aqi[-7] if len(recent_aqi) >= 7 else np.nan
        aqi_lag_14 = recent_aqi[-14] if len(recent_aqi) >= 14 else np.nan
        roll_mean_7 = pd.Series(recent_aqi[-7:]).mean()
        roll_mean_14 = pd.Series(recent_aqi).mean()

        # Define the feature vector in the correct order
        features_list = [
            day_of_year, lat, lng, population,
            aqi_lag_1, aqi_lag_7, aqi_lag_14,
            roll_mean_7, roll_mean_14
        ]
        # Create a single-row DataFrame with the features
        features_df = pd.DataFrame([features_list], columns=features)
        # Fill any NaNs (e.g., at the start of prediction) with the last known AQI
        features_df = features_df.fillna(aqi_lag_1)

        # 5. Predict using the correct model
        # Select the specific model trained for this forecast day (from MultiOutputRegressor)
        # model_list[0] predicts Day 1, model_list[1] predicts Day 2, etc.
        model_for_day = model_list[i - 1]
        predicted_aqi = model_for_day.predict(features_df)[0]

        # Store the prediction
        predicted_aqi_7_days.append(predicted_aqi)

        # 6. Store prediction and update history (Autoregressive step)
        # Add the new prediction to the history to be used for the *next* day's features
        recent_aqi.append(predicted_aqi)
        if len(recent_aqi) > 14:
            # Keep the history window at 14 days
            recent_aqi.pop(0)

    # 7. Report the results
    print(f"📈 Prediction based on historical data up to: {last_known_date.date()}")
    print("-------------------------------------------")
    print("🗓️  7-DAY AQI FORECAST:")
    for i in range(7):
        day_num = i + 1
        future_date_str = (last_known_date + timedelta(days=day_num)).strftime('%Y-%m-%d')
        pred = predicted_aqi_7_days[i]
        # Get the qualitative category
        bucket = get_aqi_bucket(pred)
        print(f"  📊 {future_date_str} (Day +{day_num}): {pred:6.2f} ({bucket})")
    print("-------------------------------------------\n")


def get_forecast_for_city(city_name, model_list, df_geog, full_data, features):
    """
    Generates a 7-day AQI forecast and returns it as a list.

    This is a non-printing version of 'predict_next_7_days_aqi_geo'
    designed for data collection.

    Args:
        city_name (str): The name of the city to forecast.
        model_list (list): The list of 7 trained model estimators.
        df_geog (pd.DataFrame): The cleaned dataframe with city geo-data.
        full_data (pd.DataFrame): The full, imputed AQI dataframe.
        features (list): The list of feature names used during training.

    Returns:
        list: A list of 7 predicted AQI values, or a list of 7
              np.nan values if data is insufficient.
    """
    # This function is nearly identical to the previous one,
    # but it RETURNS a list instead of printing.

    # --- 1. Get Static Geographical Features ---
    try:
        city_geo_data = df_geog.loc[df_geog['City'] == city_name].iloc[0]
        lat = city_geo_data['lat']
        lng = city_geo_data['lng']
        population = city_geo_data['population']
    except IndexError:
        # City not found in geo-data
        return [np.nan] * 7

    # --- 2. Get Last Known AQI Data (Dynamic Features) ---
    city_data = full_data[full_data['City'] == city_name].sort_values(by='Date')
    if city_data.empty or len(city_data) < 14:
        # Not enough historical data to build initial features
        return [np.nan] * 7

    last_known_day = city_data.iloc[-1]
    last_known_date = last_known_day['Date']

    # --- 3. Initialize Features from Last Known Day ---
    recent_aqi = list(city_data['AQI'].iloc[-14:])
    predicted_aqi_7_days = []

    # --- 4. Autoregressive Prediction Loop ---
    for i in range(1, 8):
        # --- A. Create Features for the Day ---
        future_date = last_known_date + timedelta(days=i)
        day_of_year = future_date.dayofyear

        aqi_lag_1 = recent_aqi[-1]
        aqi_lag_7 = recent_aqi[-7] if len(recent_aqi) >= 7 else np.nan
        aqi_lag_14 = recent_aqi[-14] if len(recent_aqi) >= 14 else np.nan

        roll_mean_7 = pd.Series(recent_aqi[-7:]).mean()
        roll_mean_14 = pd.Series(recent_aqi).mean()

        features_list = [day_of_year, lat, lng, population,
                         aqi_lag_1, aqi_lag_7, aqi_lag_14,
                         roll_mean_7, roll_mean_14]

        features_df = pd.DataFrame([features_list], columns=features)
        features_df = features_df.fillna(aqi_lag_1)

        # --- B. Predict using the correct model for that day ---
        model_for_day = model_list[i - 1]
        predicted_aqi = model_for_day.predict(features_df)[0]

        # --- C. Store Prediction & Update Features ---
        predicted_aqi_7_days.append(predicted_aqi)
        recent_aqi.append(predicted_aqi)
        if len(recent_aqi) > 14:
            recent_aqi.pop(0)

    # --- 5. Return Data ---
    return predicted_aqi_7_days


def evaluate_model(model, X, y, model_name, targets):
    """
    Calculates and prints regression metrics for a given model.

    Args:
        model (sklearn.base.BaseEstimator): The trained model to evaluate.
        X (pd.DataFrame): The feature matrix (X_test).
        y (pd.DataFrame): The target matrix (y_test).
        model_name (str): The name of the model for printing.
        targets (list): The list of target column names.

    Returns:
        dict | None: A dictionary of metrics (RMSE, MAE, MSE, R2), or None if evaluation fails.
    """
    print(f"\n🔍 Evaluating {model_name}...")
    try:
        # Get predictions
        y_pred = model.predict(X)
        print("✅ Predictions made.")

        # Dictionaries to store metrics for each target day
        rmse_scores = {}
        mae_scores = {}
        mse_scores = {}
        r2_scores = {}

        # Loop through each of the 7 target days
        for i, target in enumerate(targets):
            actual = y[target]  # True values for this day
            predicted = y_pred[:, i] # Predicted values for this day

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actual, predicted)) # Root Mean Squared Error
            mae = mean_absolute_error(actual, predicted)      # Mean Absolute Error
            mse = mean_squared_error(actual, predicted)         # Mean Squared Error
            r2 = r2_score(actual, predicted)                    # R-squared (Coefficient of Determination)

            # Store scores
            rmse_scores[target] = rmse
            mae_scores[target] = mae
            mse_scores[target] = mse
            r2_scores[target] = r2

        print(f"📊 Evaluation complete for {model_name}.")
        # Return metrics as a dictionary
        return {
            "RMSE": rmse_scores,
            "MAE": mae_scores,
            "MSE": mse_scores,
            "R2": r2_scores
        }

    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        return None


def evaluate_metrics(y_true, y_pred):
    """
    Calculates average RMSE, MAE, MSE, and R2 across all outputs.

    Args:
        y_true (pd.DataFrame): The true target values.
        y_pred (np.array): The predicted target values.

    Returns:
        tuple: A tuple containing the average RMSE, MAE, MSE, and R2 scores.
    """
    # 'uniform_average' calculates the metric for each target and then averages them
    rmse = root_mean_squared_error(y_true, y_pred, multioutput='uniform_average')
    mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    return rmse, mae, mse, r2

def build_ann_model(input_shape, output_shape, n_layers, units, dropout):
    """
    Builds and compiles a Keras ANN model based on hyperparameters.

    Args:
        input_shape (tuple): The shape of the input features (e.g., (num_features,)).
        output_shape (int): The number of output neurons (e.g., 7 for 7 days).
        n_layers (int): The number of hidden layers (1 or 2).
        units (int): The number of neurons in the first hidden layer.
        dropout (float): The dropout rate for regularization.

    Returns:
        keras.Sequential: A compiled Keras ANN model.
    """
    model = keras.Sequential()
    # Define the input layer
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    # Add the first hidden layer
    model.add(keras.layers.Dense(units, activation='relu'))
    model.add(keras.layers.Dropout(dropout)) # Dropout for regularization

    # Add a second hidden layer if n_layers == 2
    if n_layers == 2:
        model.add(keras.layers.Dense(int(units / 2), activation='relu'))

    # Add the output layer
    # 'linear' activation is used for regression (predicting continuous values)
    model.add(keras.layers.Dense(output_shape, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam',      # Adam is a popular and effective optimizer
                  loss='mse',          # Mean Squared Error is the loss function for regression
                  metrics=['mae'])     # Track Mean Absolute Error during training
    return model


def print_results(results_df, model_name, sort_by='RMSE', ascending=True):
    """
    Sorts, prints, and shows the best result from an evaluation DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame containing hyperparameters and metrics.
        model_name (str): The name of the model for printing.
        sort_by (str): The metric column to sort by (e.g., 'RMSE').
        ascending (bool): Sort order (True for ascending, e.g., for RMSE).

    Returns:
        None: This function prints the results to the console.
    """
    if results_df.empty:
        print(f"\n⚠️ No results found for {model_name} evaluation.")
        return

    # Sort by the specified metric
    results_df = results_df.sort_values(by=sort_by, ascending=ascending)

    print(f"\n🏆 {model_name} Evaluation Summary (sorted by best {sort_by}):")
    # .to_markdown() provides a nicely formatted table in the console
    print(results_df.to_markdown(index=False))

    # Print the best parameters
    best_params = results_df.iloc[0]
    print(f"\n--- 🥇 Best {model_name} Configuration Found (on test data) ---")
    for key, val in best_params.items():
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")


def get_forecast_for_city_ann(city_name, ann_model_single, df_geog, full_data, features):
    """
    Adapted forecast function for a single Keras ANN model.
    Assumes the ANN model predicts all 7 days simultaneously.
    This is NOT autoregressive; it predicts all 7 days based on the last known day.

    Args:
        city_name (str): The name of the city to forecast.
        ann_model_single (keras.Sequential): The single trained Keras ANN model.
        df_geog (pd.DataFrame): The cleaned dataframe with city geo-data.
        full_data (pd.DataFrame): The full, imputed AQI dataframe.
        features (list): The list of feature names used during training.

    Returns:
        list: A list of 7 predicted AQI values, or 7 np.nan values if data is insufficient.
    """
    # --- 1. Get Static Geographical Features ---
    try:
        city_geo_data = df_geog.loc[df_geog['City'] == city_name].iloc[0]
        lat = city_geo_data['lat']
        lng = city_geo_data['lng']
        population = city_geo_data['population']
    except IndexError:
        # City not found in geo-data
        return [np.nan] * 7

    # --- 2. Get Last Known AQI Data ---
    city_data = full_data[full_data['City'] == city_name].sort_values(by='Date')
    if city_data.empty or len(city_data) < 14:
        # Not enough historical data to build features
        return [np.nan] * 7

    last_known_day_data = city_data.iloc[-1]
    last_known_date = last_known_day_data['Date']

    # --- 3. Build Features for the LAST DAY ---
    # This is different: it's not autoregressive.
    # It predicts all 7 days based on the last known state.
    day_of_year = last_known_date.dayofyear
    recent_aqi = list(city_data['AQI'].iloc[-14:]) # Get last 14 days of AQI
    aqi_lag_1 = recent_aqi[-1]
    aqi_lag_7 = recent_aqi[-7] if len(recent_aqi) >= 7 else np.nan
    aqi_lag_14 = recent_aqi[-14] if len(recent_aqi) >= 14 else np.nan
    roll_mean_7 = pd.Series(recent_aqi[-7:]).mean()
    roll_mean_14 = pd.Series(recent_aqi).mean()

    features_list = [day_of_year, lat, lng, population,
                     aqi_lag_1, aqi_lag_7, aqi_lag_14,
                     roll_mean_7, roll_mean_14]

    features_df = pd.DataFrame([features_list], columns=features)
    features_df = features_df.fillna(aqi_lag_1) # Fill NaNs with last known value

    # --- 4. Predict all 7 days with the single ANN model ---
    try:
        # .predict() returns a 2D array, e.g., [[...]], so we take the first element [0]
        predicted_aqi_7_days = ann_model_single.predict(features_df, verbose=0)[0].tolist()
    except Exception as e:
        print(f"❌ Error during ANN prediction for {city_name}: {e}")
        return [np.nan] * 7

    # --- 5. Return Data ---
    return predicted_aqi_7_days


# ===================================================================
# === 3. Main Script Logic
# ===================================================================

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

print("📚 All required libraries and modules have been imported.")

# ===================================================================
# === Configuration and Geographical Data Ingestion
# ===================================================================

# Define base directory for data files (adjust if needed)
base_path = "C:/Users/asankar/Downloads/"

# Path to the main AQI data
aqi_data_path = os.path.join(base_path, "city_day_imputed_aqi.csv")
# Path to the supplementary geographical data
geog_data_path = os.path.join(base_path, "cities_geog.csv")

# Default model save path (will be overridden by specific model names)
model_save_path = "lgbm_multi_geo_model.joblib"

# Execute the function to load geographical data
df_geog_clean = load_geog_data(geog_data_path)
# If loading was successful, print a sample
if df_geog_clean is not None:
    print("📊 Geographical data sample:")
    display(df_geog_clean.head()) # 'display' is used in notebooks for rich output

# ===================================================================
# === Primary Data Ingestion and Merging
# ===================================================================

try:
    # Load the main time-series AQI data
    df_aqi = pd.read_csv(aqi_data_path)
    print(f"📄 Loaded {aqi_data_path} with {len(df_aqi)} records.")

    # Merge AQI data with geographical data on 'City' (left join)
    # This adds 'lat', 'lng', and 'population' to the AQI data
    df_merged = pd.merge(df_aqi, df_geog_clean, on='City', how='left')

    # Check for AQI records that didn't have a matching city in the geo-data
    missing_geo_count = df_merged['lat'].isnull().sum()
    if missing_geo_count > 0:
        # If there are mismatches, print a warning...
        print(f"⚠️ Warning: {missing_geo_count} AQI records had no matching geo-data. These records will be dropped.")
        # ...and drop those records (since geo features are needed for the model)
        df_merged = df_merged.dropna(subset=['lat', 'lng', 'population'])

    print("✅ Successfully merged AQI and geographical data.")
    print("📊 Merged data sample:")
    display(df_merged.head())

except FileNotFoundError:
    # Handle missing AQI file
    print(f"❌ Error: Primary AQI data file not found at {aqi_data_path}")
except Exception as e:
    # Handle other merge/load errors
    print(f"❌ An error occurred during data loading and merging: {e}")

# ===================================================================
# === Data Preprocessing and Imputation
# ===================================================================

# Parse the 'Date' column, 'mixed' format handles potential inconsistencies
df_merged['Date'] = pd.to_datetime(df_merged['Date'], format='mixed')

# Sort by City and Date, which is crucial for creating time-series features
df_merged = df_merged.sort_values(by=['City', 'Date'])

# Impute missing 'AQI' values using the mean AQI for *that specific city*
print("🔧 Imputing missing 'AQI' values...")
# .transform() applies the function (fillna with mean) back to the original DataFrame shape
df_merged['AQI'] = df_merged.groupby('City')['AQI'].transform(lambda x: x.fillna(x.mean()))

# Drop rows if AQI is still NaN (e.g., a city with no AQI data at all)
df_merged = df_merged.dropna(subset=['AQI'])

# Create a final, clean copy for feature engineering
df_imputed = df_merged.copy()
print(f"✅ Data imputation complete. Dataset now contains {len(df_imputed)} records.")

# ===================================================================
# === Feature Engineering
# ===================================================================

print("⚙️ Starting feature engineering process...")
# Apply the feature engineering function defined earlier
df_model_data = create_features(df_imputed)

# Drop rows with NaN values. These are created by lag/shift operations
# (e.g., the first 14 days of a city, which have no lag_14)
# and by the target creation (e.g., the last 7 days of a city, which have no target_day_7)
df_model_data = df_model_data.dropna()
print(f"✅ Feature engineering complete. Final model-ready dataset shape: {df_model_data.shape}")

# ===================================================================
# === Define Features (X) and Targets (y)
# ===================================================================

# Dynamically find all created lag columns
lag_cols = [col for col in df_model_data.columns if 'AQI_lag' in col]
# Dynamically find all created rolling mean columns
roll_cols = [col for col in df_model_data.columns if 'AQI_roll' in col]

# Define the final list of features (X) for the model
# These are the inputs to the model
features = ['Day_of_year', 'lat', 'lng', 'population'] + lag_cols + roll_cols

# Define the list of target variables (y) - the next 7 days
# These are the outputs the model will learn to predict
targets = [f'target_day_{i}' for i in range(1, 8)]

# Create the feature matrix
X = df_model_data[features]
# Create the target matrix
y = df_model_data[targets]

# NEW: Apply 80-20 train-test split
# Split the data into training (80%) and testing (20%) sets
# 'random_state=42' ensures the split is reproducible
# 'shuffle=True' randomizes the data before splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"📏 Train set: {X_train.shape}, Test set: {X_test.shape}")

print(f"📚 Training dataset prepared with {len(X_train)} samples.")
print(f"📋 Number of features: {len(features)}")
print(f"🎯 Number of targets: {len(targets)}")

# ===================================================================
# === Model Training and Saving (Original Linear Regression)
# ===================================================================
# This serves as the baseline model for comparison.

print("🏃 Commencing model training (Original Linear Regression)...")

# Base estimator
base_model = LinearRegression()

# Wrap the base model in MultiOutputRegressor. This will train one
# separate LinearRegression model for each of the 7 target days.
model = MultiOutputRegressor(base_model)

# Train all 7 models on training split
model.fit(X_train, y_train)
print("✅ Model training has completed.")

# Save the trained model (all 7 estimators) to disk
model_save_path = "linear_regression_multi_geo_model.joblib"
joblib.dump(model, model_save_path)
print(f"💾 Trained model has been saved to: {model_save_path}")

print(f"📦 Model type persisted: {type(model)}")
print(f"🔢 Number of estimators in model: {len(model.estimators_)}")

# ===================================================================
# === Model Evaluation (Original Model) - Now on Test Set
# ===================================================================

print("--- 📉 Evaluating Model Performance (Original Linear Regression on Test Data) ---")

# Get predictions on the test data (data the model has not seen)
y_test_pred = model.predict(X_test)
print("✅ Predictions made on test data.")

# Initialize dictionaries to store metrics for each target day
rmse_scores = {}
mae_scores = {}
mse_scores = {}
r2_scores = {}

# Loop through each target day (day 1 to day 7)
for i, target in enumerate(targets):
    # Get the true values for this day
    actual = y_test[target]
    # Get the predicted values for this day (column 'i' from the prediction matrix)
    predicted = y_test_pred[:, i]

    # Calculate standard regression metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    # Store scores
    rmse_scores[target] = rmse
    mae_scores[target] = mae
    mse_scores[target] = mse
    r2_scores[target] = r2

    # Print scores - Updated to indicate Test Data
    print(f"\nMetrics for {target} (Test Data):")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  MSE:  {mse:.2f}")
    print(f"  R2:   {r2:.4f}")

# Store the baseline model's metrics in a dedicated dictionary for later
linear_regression_metrics = {
    "RMSE": rmse_scores,
    "MAE": mae_scores,
    "MSE": mse_scores,
    "R2": r2_scores
}
print("\n📌 Baseline Linear Regression metrics captured for comparison.")

print("\n✅ --- Evaluation Complete ---")

# ===================================================================
# === Prediction Utilities
# ===================================================================

# Note: All prediction functions are defined at the top of the script.
print("🛠️ Forecast generation function 'get_forecast_for_city' is defined.")

# ===================================================================
# === Generate and Visualize All-City Forecasts (Original Model)
# ===================================================================

print("--- 🚀 Starting All-City Forecast Generation (Original Model) ---")

try:
    # --- 1. Load Required Assets ---

    # Load the saved Linear Regression model's estimators
    model_list = joblib.load(model_save_path).estimators_
    print(f"📤 Loaded {len(model_list)} models from {model_save_path}")

    # Check if necessary DataFrames and lists exist from previous steps
    if 'df_geog_clean' not in globals() or 'df_imputed' not in globals():
        print("❌ Error: 'df_geog_clean' or 'df_imputed' not found.")
        print("🔄 Please re-run Cell 2 and Cell 4 before this cell.")
        raise NameError("Missing required dataframes")

    # Check if 'features' list exists from Cell 6
    if 'features' not in globals():
        print("❌ Error: 'features' list not found.")
        print("🔄 Please re-run Cell 6 before this cell.")
        raise NameError("Missing features list")

    # --- 2. Loop Through All Cities and Collect Data ---
    # Get a list of all unique cities
    all_cities = np.sort(df_geog_clean['City'].unique())
    all_forecasts = {}  # Dictionary to store forecasts

    print(f"⏳ Generating forecasts for {len(all_cities)} cities. This may take a moment...")

    for city in all_cities:
        # Use the utility function to get the 7-day forecast
        forecast = get_forecast_for_city(
            city,
            model_list,
            df_geog_clean,
            df_imputed,
            features
        )
        all_forecasts[city] = forecast

    print("✅ ...All forecasts generated successfully.")

    # --- 3. Create the Forecast "Matrix" (DataFrame) ---
    # Convert the dictionary to a DataFrame (index=city, columns=days)
    df_forecast = pd.DataFrame.from_dict(all_forecasts, orient='index')
    # Name the columns
    df_forecast.columns = [f'Day +{i}' for i in range(1, 8)]

    # Drop cities that failed prediction (returned NaNs)
    df_forecast = df_forecast.dropna()

    # Save the forecast data to a CSV
    csv_filename = "7_day_forecast_matrix.csv"
    df_forecast.to_csv(csv_filename)
    print(f"💾 Forecast matrix saved to '{csv_filename}'")

    # --- 4. Create the Color-Coded Heatmap ---

    # Set theme for seaborn
    sns.set_theme(style="white")

    # Dynamically set figure height based on the number of cities
    n_cities = len(df_forecast)
    fig_height = max(10, n_cities * 0.5)

    plt.figure(figsize=(14, fig_height))

    # Create the heatmap
    heatmap = sns.heatmap(
        df_forecast,
        annot=True,  # Show the AQI values in each cell
        fmt=".0f",  # Format values as integers
        linewidths=.5,  # Add thin lines between cells
        cmap="YlOrRd",  # Yellow-Orange-Red colormap (good for AQI)
        cbar_kws={'label': 'Predicted AQI'}  # Label for the color bar
    )

    plt.title('7-Day AQI Forecast Matrix for All Cities', fontsize=18, pad=20)
    plt.xlabel('Forecast Day', fontsize=12)
    plt.ylabel('City', fontsize=12)
    plt.yticks(rotation=0)  # Keep city names horizontal

    # Save the plot as a PNG
    plot_filename = "aqi_forecast_heatmap.png"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
    print(f"🖼️ Heatmap saved to '{plot_filename}'")

    # Show the plot
    plt.show()

    print("\n--- 🎉 Process Complete ---")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("🔄 Please ensure all previous cells have been run successfully.")

# ===================================================================
# === Worst-Case Finder (Original Model)
# ===================================================================

print("\n--- 🔥 Identifying Top 5 Worst-Case Forecasts ---")
try:
    # Check if forecast data exists
    if 'df_forecast' not in globals():
        print("❌ Error: 'df_forecast' DataFrame not found. Run the all-city forecast cell first.")
        raise NameError("'df_forecast' not found")

    # Convert the wide forecast matrix (cities x days) to a long format
    # (city, day, value)
    df_long = df_forecast.stack().reset_index()
    df_long.columns = ['City', 'Forecast_Day', 'Predicted_AQI']

    # Sort by predicted AQI to find the worst 5
    df_top5 = df_long.sort_values(by='Predicted_AQI', ascending=False).head(5)

    # Check if utility function exists
    if 'get_aqi_bucket' not in globals():
        raise NameError("'get_aqi_bucket' not found")

    print("☣️ The 5 highest predicted pollution events (worst cases) in the 7-day forecast are:")
    print("-----------------------------------------------------------------------------")
    # Loop through the top 5 and print details
    for row in df_top5.itertuples():
        aqi_bucket = get_aqi_bucket(row.Predicted_AQI)
        print(f"    City: {row.City}")
        print(f"    Day: {row.Forecast_Day}")
        print(f"    Forecast: {row.Predicted_AQI:.0f} AQI ({aqi_bucket}) ")
        print("")
    print("-----------------------------------------------------------------------------")

except NameError as e:
    print(f"❌ A required object was not found: {e}")
except Exception as e:
    print(f"❌ An error occurred while identifying worst cases: {e}")

# ===================================================================
# === Past vs. Future Heatmap (Original Model)
# ===================================================================

print("\n--- 📅 Generating 30-Day Past vs. 7-Day Future Heatmap ---")
try:
    # Check for required data
    if 'df_imputed' not in globals() or 'df_forecast' not in globals():
        print("❌ Error: 'df_imputed' or 'df_forecast' missing. Re-run preprocessing and all-city forecast.")
        raise NameError("Missing required DataFrames")

    # Get list of all cities (needed if df_geog_clean is not available, though it should be)
    if 'df_geog_clean' not in globals():
        all_cities = np.sort(df_geog_clean['City'].unique())

    print("🔍 Extracting last 30 days of actual data for all cities...")
    # 1. Extract last 30 days of *actual* data for each city
    past_data = {}
    for city in all_cities:
        city_aqi_history = df_imputed[df_imputed['City'] == city]['AQI'].values
        if len(city_aqi_history) >= 30:
            past_data[city] = list(city_aqi_history[-30:])
        else:
            # Pad with NaNs if history is shorter than 30 days
            pad_width = 30 - len(city_aqi_history)
            past_data[city] = [np.nan] * pad_width + list(city_aqi_history)

    df_past = pd.DataFrame.from_dict(past_data, orient='index')
    # Name columns for past days
    df_past.columns = [f'Day -{i}' for i in range(30, 0, -1)]

    # 2. Combine past data with future (predicted) data
    df_combined = pd.concat([df_past, df_forecast], axis=1).dropna()
    print(f"✅ Combined matrix created with shape: {df_combined.shape}")

    # 3. Plot the combined heatmap
    sns.set_theme(style="white")
    n_cities = len(df_combined)
    fig_height = max(10, n_cities * 0.5)
    plt.figure(figsize=(22, fig_height))

    sns.heatmap(
        df_combined,
        annot=True,
        fmt=".0f",
        linewidths=.5,
        cmap="YlOrRd",
        cbar_kws={'label': 'AQI Value (Actual & Predicted)'},
        annot_kws={"size": 8} # Font size for annotations
    )

    # 4. Add a visual divider line
    # Draw a line at column 30 to separate past from future
    plt.axvline(x=30, color='blue', linestyle='--', linewidth=2)
    # Add labels for each section
    plt.text(15, -0.5, 'PAST 30 DAYS (Actual)', ha='center', va='center',
             fontsize=14, fontweight='bold', color='black')
    plt.text(33.5, -0.5, 'FUTURE 7 DAYS (Predicted)', ha='center', va='center',
             fontsize=14, fontweight='bold', color='blue')

    plt.title('30-Day Past (Actual) vs. 7-Day Future (Predicted) AQI Matrix', fontsize=20, pad=40)
    plt.xlabel('Day Relative to Present', fontsize=12)
    plt.ylabel('City', fontsize=12)
    plt.yticks(rotation=0)

    # Adjust x-axis ticks for better readability (show every 2nd tick label)
    ticks = plt.xticks()
    plt.xticks(ticks[0][::2], df_combined.columns[::2], rotation=45, ha='right')

    plot_filename = "aqi_past_future_heatmap.png"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=200) # Save with high resolution
    print(f"🖼️ Combined Past/Future heatmap saved to '{plot_filename}'")

    # Show the plot
    plt.show()

    print("\n--- 🎉 Combined Matrix Process Complete ---")

except NameError as e:
    print(f"❌ A required object was not found: {e}")
except Exception as e:
    print(f"❌ \nAn error occurred during plotting: {e}")

# ===================================================================
# === Train and Evaluate New Models (Initial) - Now on Test Set
# ===================================================================
# This cell trains the new models (RandomForest, Polynomial Regression, ANN)
# for the first time. Later cells will perform hyperparameter tuning.

print("--- 🚀 Commencing training for RandomForest, Polynomial Regression, and ANN models ---")

# Dictionary to store all evaluation results for comparison
evaluation_results = {}

# Add the baseline metrics (from the LinearRegression model) to the results dictionary
if 'linear_regression_metrics' in globals():
    evaluation_results["LinearRegression"] = linear_regression_metrics
    print("📌 Added baseline Linear Regression metrics to comparison.")
else:
    print("⚠️ Warning: Baseline Linear Regression metrics not found. Run previous cells.")


# --- 1. Train RandomForest Model ---
# Using RandomForest as a more robust model than simple Linear Regression
print("🌳 Training RandomForest Regressor...")
# Using n_estimators=3, which is very low, just for an initial quick test.
# n_jobs=-1 uses all available CPU cores.
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=3, random_state=15, n_jobs=-1))
rf_model.fit(X_train, y_train)  # Train on train split
# Save the RF model
joblib.dump(rf_model, 'random_forest_model.joblib')
print("✅ RandomForest Regressor training complete and model saved to 'random_forest_model.joblib'.")

# Evaluate the RF model on test split
rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest (Test)", targets)
if rf_metrics:
    # Store its metrics
    evaluation_results["RandomForest"] = rf_metrics


# --- 2. Train Polynomial Regression Model ---
print("\n📈 Training Polynomial Regression Model...")
# Create a pipeline to combine polynomial feature creation and linear regression
# This automates the process:
# 1. 'poly_features': Transforms X into X_poly (with degree 2)
# 2. 'linear_regression': Fits a LinearRegression model on X_poly
poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_reg_model = MultiOutputRegressor(Pipeline([
    ('poly_features', poly_features),
    ('linear_regression', LinearRegression())
]))

# Train on a subset of the training data for speed.
# PolynomialFeatures(degree=2) creates many features, which is memory-intensive.
subset_size = 5000
X_train_subset = X_train.sample(n=min(subset_size, len(X_train)), random_state=42)
y_train_subset = y_train.loc[X_train_subset.index]

poly_reg_model.fit(X_train_subset, y_train_subset)  # Train on subset of train split
# Save the PolyReg model
joblib.dump(poly_reg_model, 'polynomial_regression_model.joblib')
print("✅ Polynomial Regression Model training complete and model saved to 'polynomial_regression_model.joblib'.")

# Evaluate the PolyReg model on full test set
poly_metrics = evaluate_model(poly_reg_model, X_test, y_test, "Polynomial Regression (Test)", targets)
if poly_metrics:
    evaluation_results["Polynomial Regression"] = poly_metrics


# --- 3. Train ANN Model ---
print("\n🧠 Training ANN Model...")

# Define ANN input shape based on number of features
input_shape = (X_train.shape[1],)
# Define ANN output shape (7 targets)
output_shape = y_train.shape[1]

# Check if ANN model already exists and train if not or retrain
try:
    # Try to load an existing ANN model
    ann_model = tf.keras.models.load_model('ann_model.h5', compile=False)
    print("🔄 Existing ANN model found. Compiling and continuing training.")
    # Re-compile the loaded model
    ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

except (FileNotFoundError, ValueError, tf.errors.NotFoundError) as e:
    print(f"💡 No existing ANN model found or error loading ({e}). Building a new one.")
    # Build a new sequential model
    ann_model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),  # Dropout for regularization
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(output_shape, activation='linear')  # 'linear' activation for regression
    ])

    # Compile the model with Adam optimizer and Mean Squared Error loss
    ann_model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])

    print("✅ New ANN model built and compiled.")

# Train the ANN model on train split
print("🏃 Fitting ANN model...")
history = ann_model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        verbose=0)  # Suppress training log output

print("✅ ANN model training complete.")

# Save the trained ANN model
try:
    ann_model.save('ann_model.h5')
    print("💾 ANN model saved to 'ann_model.h5'.")
except Exception as e:
    print(f"❌ Error saving ANN model: {e}")

# Evaluate the ANN model on test split
ann_metrics = evaluate_model(ann_model, X_test, y_test, "ANN (Test)", targets)
if ann_metrics:
    evaluation_results["ANN"] = ann_metrics

# --- Print Comparison of Evaluation Metrics ---
print("\n--- 📊 Model Comparison (Evaluation Metrics on Test Data) ---")

if not evaluation_results:
    print("⚠️ No models were successfully evaluated.")
else:
    # Create a structured output for comparison
    # Loop through each metric type
    for metric_name in ['RMSE', 'MAE', 'R2','MSE']:
        print(f"\n{metric_name} Scores:")
        # Determine the maximum length of model names for alignment
        max_name_len = max(len(name) for name in evaluation_results.keys())
        # Print a formatted header
        header = f"{'Model':<{max_name_len + 5}}" + "".join([f"{f'Day +{i + 1}':>10}" for i in range(len(targets))])
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        # Print scores for each model
        for model_name in sorted(evaluation_results.keys()):
            metrics = evaluation_results[model_name]
            if metric_name in metrics:
                scores = metrics[metric_name]
                score_line = f"{model_name:<{max_name_len + 5}}"
                for target in targets:
                    score_value = scores.get(target, np.nan)
                    # Format R2 to 4 decimal places, others to 2
                    if metric_name == 'R2':
                        score_line += f"{score_value:>10.4f}" if not np.isnan(score_value) else f"{'N/A':>10}"
                    else:
                        score_line += f"{score_value:>10.2f}" if not np.isnan(score_value) else f"{'N/A':>10}"
                print(score_line)
        print("-" * len(header))

print("\n--- 🎉 Model training and initial evaluation complete for all three models ---")

# ===================================================================
# === Hyperparameter Tuning: RandomForest - Now Evaluate on Test
# ===================================================================
# This cell tests different values for 'n_estimators' in the
# RandomForestRegressor and plots the impact on performance.

# Check if X_train and y_train exist
try:
    X_train.shape
    y_train.shape
    print(f"✅ Using existing X_train with shape {X_train.shape} and y_train with shape {y_train.shape}")
except NameError:
    print("❌ X_train or y_train not found")

# 1. Initialize lists to store metrics
# Define the range of n_estimators to test
estimators_range = range(5, 101, 5)  # Test 5, 10, 15, ..., 100
n_estimators_list = []
rmse_list = []
r2_list = []

print("\n🔄 Starting evaluation loop for RandomForestRegressor...")

# 2. Create the for loop to iterate through different n_estimators values
for n in estimators_range:
    print(f"--- 🏃 Training with n_estimators={n} ---")

    # Define the model with the current 'n'
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=n, random_state=15, n_jobs=-1)
    )

    # Train the model on train split
    model.fit(X_train, y_train)

    # Evaluate the model on test split
    y_pred_test = model.predict(X_test)

    # Calculate average metrics across all outputs
    rmse = root_mean_squared_error(y_test, y_pred_test, multioutput='uniform_average')
    r2 = r2_score(y_test, y_pred_test, multioutput='uniform_average')

    # Store results
    n_estimators_list.append(n)
    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f"📊 Metrics (Test): RMSE = {rmse:.4f}, R2 = {r2:.4f}")

print("--- ✅ Evaluation loop complete. ---")

# 3. Create a DataFrame with the results
metrics_df = pd.DataFrame({
    'n_estimators': n_estimators_list,
    'RMSE': rmse_list,
    'R2_Score': r2_list
})

print("\n📈 Evaluation Metrics vs. n_estimators (on test data):")
print(metrics_df.to_markdown(index=False))

# 4. Create line charts using Altair
# Melt the DataFrame for easier plotting with Altair
# Converts from wide (n_estimators, RMSE, R2_Score) to long (n_estimators, Metric, Value)
metrics_df_melted = metrics_df.melt(
    'n_estimators',
    var_name='Metric',
    value_name='Value'
)

# Create a base chart
base = alt.Chart(metrics_df_melted).mark_line(point=True).encode(
    x=alt.X('n_estimators', title='Number of Estimators', axis=alt.Axis(tickMinStep=1)),
    y=alt.Y('Value', title='Metric Value'),
    color='Metric', # Different color line for each metric
    tooltip=['n_estimators', 'Metric', alt.Tooltip('Value', format='.4f')]
).properties(
    title='RandomForest Performance vs. Number of Estimators (on Test Data)'
).interactive() # Allow zooming and panning

# Facet the chart by 'Metric' to create two separate plots (one for RMSE, one for R2)
# This gives each metric its own independent Y-axis scale
chart = base.facet(
    row=alt.Row('Metric', title='Evaluation Metric', header=alt.Header(titleOrient="top", labelOrient="top")),
    resolve=alt.Resolve(scale={'y': 'independent'})  # Use independent Y-scales
)

# Save the chart as a JSON file
chart_filename = 'rf_n_estimators_vs_metrics_chart.json'
chart.save(chart_filename)
# chart.show() # This line is commented out to avoid dependency/display errors in some environments
print(f"🖼️ Chart saved to '{chart_filename}'")

# ===================================================================
# === Hyperparameter Tuning: Polynomial Regression - Now Evaluate on Test
# ===================================================================
# This cell tests different values for 'degree' in PolynomialFeatures
# and plots the impact on performance.

print("\n--- 📈 Training Polynomial Regression Model with varying degree ---")

# Define the range of polynomial degrees to test
degree_range = range(2, 5)  # Test degrees 2, 3, and 4

# Lists to store *average* metrics across target days for plotting
poly_avg_rmse_scores = []
poly_avg_mae_scores = []
poly_avg_mse_scores = []
poly_avg_r2_scores = []
degrees_list = []

# --- Check if X_train and y_train exist ---
try:
    X_train.shape
    y_train.shape
    print(f"✅ Using existing X_train with shape {X_train.shape} and y_train with shape {y_train.shape}")
except NameError:
    print("❌ X_train or y_train not found. Please load data.")

# Subset the training data for faster training during hyperparameter tuning
subset_size = 5000
X_train_subset = X_train.sample(n=min(subset_size, len(X_train)), random_state=42)
y_train_subset = y_train.loc[X_train_subset.index]
print(f"🔬 Using a subset of {len(X_train_subset)} samples for tuning.")

print("\n🔄 Starting evaluation loop for Polynomial Regression degree...")

# Loop through different polynomial degrees
for degree in degree_range:
    print(f"\n🏃 Training Polynomial Regression with degree = {degree}...")

    try:
        # Create the pipeline
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        poly_reg_model = MultiOutputRegressor(Pipeline([
            ('poly_features', poly_features),
            ('linear_regression', LinearRegression())
        ]))

        # Train on the subset of train split
        poly_reg_model.fit(X_train_subset, y_train_subset)
        print(f"✅ Polynomial Regression Model training complete for degree = {degree}.")

        # Evaluate on the test split
        poly_metrics = evaluate_model(poly_reg_model, X_test, y_test, f"Polynomial Regression (degree={degree}, Test)", targets)

        if poly_metrics:
            # Calculate average metrics across all target days
            avg_rmse = np.mean(list(poly_metrics["RMSE"].values()))
            avg_mae = np.mean(list(poly_metrics["MAE"].values()))
            avg_mse = np.mean(list(poly_metrics["MSE"].values()))
            avg_r2 = np.mean(list(poly_metrics["R2"].values()))

            # Store the average metrics
            poly_avg_rmse_scores.append(avg_rmse)
            poly_avg_mae_scores.append(avg_mae)
            poly_avg_mse_scores.append(avg_mse)
            poly_avg_r2_scores.append(avg_r2)
            degrees_list.append(degree)
        else:
            # Append NaN if evaluation failed
            poly_avg_rmse_scores.append(np.nan)
            poly_avg_mae_scores.append(np.nan)
            poly_avg_mse_scores.append(np.nan)
            poly_avg_r2_scores.append(np.nan)
            degrees_list.append(degree)

    except Exception as e:
        # Handle errors (e.g., memory errors for high degrees)
        print(f"❌ An error occurred during training or evaluation for degree {degree}: {e}. Skipping.")
        poly_avg_rmse_scores.append(np.nan)
        poly_avg_mae_scores.append(np.nan)
        poly_avg_mse_scores.append(np.nan)
        poly_avg_r2_scores.append(np.nan)
        degrees_list.append(degree)

print("\n--- ✅ Evaluation loop for Polynomial Regression degree complete. ---")

# Create a DataFrame with the results for plotting
poly_metrics_df = pd.DataFrame({
    'Degree': degrees_list,
    'Average RMSE': poly_avg_rmse_scores,
    'Average MAE': poly_avg_mae_scores,
    'Average R2': poly_avg_r2_scores
})

print("\n📊 Polynomial Regression Average Evaluation Metrics vs. Degree (on test data):")
print(poly_metrics_df.to_markdown(index=False))

# --- Plotting Degree vs Average Evaluation Metrics ---
print("\n--- 🖼️ Plotting Degree vs Average Evaluation Metrics ---")

# Check if there are valid results to plot
if not poly_metrics_df.empty and poly_metrics_df[['Average RMSE', 'Average MAE', 'Average R2']].notna().any().any():
    # Create a plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot RMSE and MAE on the primary y-axis (ax1)
    color = 'tab:blue'
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('Average Error (RMSE, MAE)', fontsize=12, color=color)
    ax1.plot(poly_metrics_df['Degree'], poly_metrics_df['Average RMSE'], marker='o', linestyle='-', label='Average RMSE', color='blue')
    ax1.plot(poly_metrics_df['Degree'], poly_metrics_df['Average MAE'], marker='s', linestyle='-', label='Average MAE', color='cyan')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a secondary y-axis (ax2) for R2
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Average R2 Score', fontsize=12, color=color)
    ax2.plot(poly_metrics_df['Degree'], poly_metrics_df['Average R2'], marker='x', linestyle='--', color=color, label='Average R2')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.title('Polynomial Regression Performance vs. Degree (on Test Data)', fontsize=16)
    plt.xticks(degree_range)  # Set x-ticks to the exact degrees
    ax1.grid(True)
    fig.tight_layout()  # Adjust plot to prevent overlap
    plt.show()
else:
    print("⚠️ Plotting skipped as no valid metrics were collected.")

print("\n--- 🎉 Polynomial Regression training and evaluation complete ---")

# --- Retrain Final Model ---
# Based on the evaluation, degree 2 is typically the best (avoids overfitting)
chosen_degree = 2

print(f"\n🚀 Retraining and saving final Polynomial Regression model with degree = {chosen_degree}...")
final_poly_features = PolynomialFeatures(degree=chosen_degree, include_bias=False)
final_poly_reg_model = MultiOutputRegressor(Pipeline([
    ('poly_features', final_poly_features),
    ('linear_regression', LinearRegression())
]))

# Train the final model on the full train split (not the subset)
final_poly_reg_model.fit(X_train, y_train)
joblib.dump(final_poly_reg_model, 'polynomial_regression_model.joblib')
print("💾 Final Polynomial Regression Model saved to 'polynomial_regression_model.joblib'.")

# Evaluate and store metrics for the final chosen model on test
print("🔍 Evaluating final model on test data...")
final_poly_metrics = evaluate_model(final_poly_reg_model, X_test, y_test, "Polynomial Regression (Test)", targets)
if final_poly_metrics:
    # Overwrite the initial 'Polynomial Regression' entry with the tuned one
    evaluation_results["Polynomial Regression"] = final_poly_metrics
    print("✅ Final model evaluation metrics stored.")

# ===================================================================
# === Hyperparameter Tuning: ANN - Now Evaluate on Test
# ===================================================================
# This cell performs a grid search for ANN hyperparameters.

# --- Setup: Define X_train and y_train ---
try:
    X_train.shape
    y_train.shape
    print(f"✅ Using existing X_train with shape {X_train.shape} and y_train with shape {y_train.shape}")
except NameError:
    print("⚠️ Warning: X_train or y_train not found. Using small placeholder data.")
    X_train = np.random.rand(100, 20)
    y_train = np.random.rand(100, 7)

# Get input and output shapes (needed for ANN)
input_shape = (X_train.shape[1],)
output_shape = y_train.shape[1]

print("🧠 Part 2: Starting ANN Hyperparameter Search...")
print("=" * 50)

# 2.1. Define the hyperparameter grid as a dictionary
ann_param_grid = {
   # Example of a larger grid (commented out for speed)
   # 'n_layers': [1, 2, 3,4,5,],
   # 'epochs': [50, 100, 150,200,250,300],
   # 'units': [128],
   # 'dropout': [0.1],
   # 'batch_size': [50]
    
    # A specific grid for the final run
    'n_layers': [4],
    'epochs': [150],
    'units': [128],
    'dropout': [0.1],
    'batch_size': [50]
    
}

# Create the grid and initialize results list
grid = ParameterGrid(ann_param_grid)
ann_results_list = []
total_combinations = len(grid)

print(f"🧪 Testing {total_combinations} total ANN combinations.")

# 2.2. Single loop using ParameterGrid
for i, params in enumerate(grid):
    # Extract params
    n_layers = params['n_layers']
    epochs = params['epochs']
    units = params['units']
    dropout = params['dropout']
    batch_size = params['batch_size']

    current_config = (
        f"Layers: {n_layers}, Epochs: {epochs}, Units: {units}, "
        f"Dropout: {dropout}, Batch Size: {batch_size}"
    )
    print(f"\n--- 🔧 Testing ANN Config {i + 1}/{total_combinations}: {current_config} ---")

    # Build a new model *for each combination*
    ann_model = build_ann_model(input_shape, output_shape, n_layers, units, dropout)

    # Train the model on train split
    ann_model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=0) # verbose=0 suppresses training logs

    print("✅ Training complete. Evaluating...")

    # Evaluate on test split
    y_pred_test_ann = ann_model.predict(X_test, verbose=0)

    # Get average metrics
    rmse_ann, mae_ann, mse_ann, r2_ann = evaluate_metrics(y_test, y_pred_test_ann)
    print(f"📊 Metrics (Test): RMSE = {rmse_ann:.4f}, MAE = {mae_ann:.4f}, MSE = {mse_ann:.4f}, R2 = {r2_ann:.4f}")

    # Store results
    result_row = {**params, 'RMSE': rmse_ann, 'MAE': mae_ann, 'MSE': mse_ann, 'R2_Score': r2_ann}
    ann_results_list.append(result_row)

# --- ANN Reporting ---
ann_results_df = pd.DataFrame(ann_results_list)
# Print the sorted results table using the utility function
print_results(ann_results_df, "ANN", sort_by='RMSE', ascending=True)

print("\n" + "=" * 50)
print("🎉 Script Finished.")
print("=" * 50)

# ===================================================================
# === Final Model Training (Post-Tuning) - Now on Test
# ===================================================================
# This cell re-trains all three models using the best hyperparameters
# found in the tuning steps above.

print("--- 🚀 Commencing training for RandomForest, Polynomial Regression, and ANN models ---")

# Re-initialize dictionary to store final model metrics
evaluation_results = {}

# Add the baseline metrics (from the LinearRegression model) to the results dictionary
if 'linear_regression_metrics' in globals():
    evaluation_results["LinearRegression"] = linear_regression_metrics
    print("📌 Added baseline Linear Regression metrics to comparison.")
else:
    print("⚠️ Warning: Baseline Linear Regression metrics not found. Run previous cells.")


# --- 1. Train RandomForest Model (Tuned) ---
print("🌳 Training RandomForest Regressor...")
# Using n_estimators=50 (based on tuning, this is a good trade-off)
rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=15, n_jobs=-1))
rf_model.fit(X_train, y_train)  # Train on train split
joblib.dump(rf_model, 'random_forest_model.joblib')
print("✅ RandomForest Regressor training complete and model saved to 'random_forest_model.joblib'.")

# Evaluate RandomForest on test
rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest (Test)", targets)
if rf_metrics:
    evaluation_results["RandomForest"] = rf_metrics


# --- 2. Train Polynomial Regression Model (Tuned) ---
# Note: The tuning cell already trained and saved the final degree=2 model.
# This part is slightly redundant but ensures it's trained if the
# tuning cell was modified or skipped.
print("\n📈 Training Polynomial Regression Model...")
# Using degree=2 (based on tuning)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_reg_model = MultiOutputRegressor(Pipeline([
    ('poly_features', poly_features),
    ('linear_regression', LinearRegression())
]))

# Train on full train split (updated from subset for final)
poly_reg_model.fit(X_train, y_train)
joblib.dump(poly_reg_model, 'polynomial_regression_model.joblib')
print("✅ Polynomial Regression Model training complete and model saved to 'polynomial_regression_model.joblib'.")

# Evaluate Polynomial Regression on test
poly_metrics = evaluate_model(poly_reg_model, X_test, y_test, "Polynomial Regression (Test)", targets)
if poly_metrics:
    evaluation_results["Polynomial Regression"] = poly_metrics


# --- 3. Train ANN Model (Tuned) ---
print("\n🧠 Training ANN Model...")

# (Data check logic)
try:
    X_train.shape
    y_train.shape
    print(f"✅ Using existing X_train with shape {X_train.shape} and y_train with shape {y_train.shape}")
except NameError:
    print("⚠️ Warning: X_train or y_train not found. Using small placeholder data.")
    X_train = np.random.rand(100, 20)
    y_train = np.random.rand(100, 7)

# Get input and output shapes
input_shape = (X_train.shape[1],)
output_shape = y_train.shape[1]

print("🏃 Part 2: Starting ANN Final Training...")
print("=" * 50)

# Final ANN parameters (based on tuning results)
ann_param_grid = {
    'n_layers': [4],
    'epochs': [150],
    'units': [128],
    'dropout': [0.1],
    'batch_size': [50]
}

# Create the grid (will only have one item)
grid = ParameterGrid(ann_param_grid)
ann_results_list = []
total_combinations = len(grid)

print(f"🧪 Training {total_combinations} final ANN combination.")

# This loop will only run once
for i, params in enumerate(grid):
    # Extract params
    n_layers = params['n_layers']
    epochs = params['epochs']
    units = params['units']
    dropout = params['dropout']
    batch_size = params['batch_size']

    current_config = (
        f"Layers: {n_layers}, Epochs: {epochs}, Units: {units}, "
        f"Dropout: {dropout}, Batch Size: {batch_size}"
    )
    print(f"\n--- 🔧 Training ANN Config {i + 1}/{total_combinations}: {current_config} ---")

    # Build the final ANN model
    ann_model = build_ann_model(input_shape, output_shape, n_layers, units, dropout)

    # Train the final ANN model on train split
    ann_model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=0)

    print("✅ Training complete. Evaluating...")

    # Evaluate on test split
    y_pred_test_ann = ann_model.predict(X_test, verbose=0)

    # Get average metrics
    rmse_ann, mae_ann, mse_ann, r2_ann = evaluate_metrics(y_test, y_pred_test_ann)
    print(f"📊 Metrics (Test): RMSE = {rmse_ann:.4f}, MAE = {mae_ann:.4f}, MSE = {mse_ann:.4f}, R2 = {r2_ann:.4f}")

    # Store results
    result_row = {**params, 'RMSE': rmse_ann, 'MAE': mae_ann, 'MSE': mse_ann, 'R2_Score': r2_ann}
    ann_results_list.append(result_row)

# --- ANN Reporting ---
ann_results_df = pd.DataFrame(ann_results_list)
print_results(ann_results_df, "ANN", sort_by='RMSE', ascending=True)

print("\n" + "=" * 50)
print("🎉 Script Finished.")
print("=" * 50)

# Save the final tuned ANN model
try:
    ann_model.save('ann_model.h5')
    print("💾 Final tuned ANN model saved to 'ann_model.h5'.")
except Exception as e:
    print(f"❌ Error saving final ANN model: {e}")

# Evaluate the final ANN on test
ann_metrics = evaluate_model(ann_model, X_test, y_test, "ANN (Test)", targets)
if ann_metrics:
    evaluation_results["ANN"] = ann_metrics


# --- Print Comparison of Evaluation Metrics ---
print("\n--- 🏆 Model Comparison (Evaluation Metrics on Test Data) ---")

if not evaluation_results:
    print("⚠️ No models were successfully evaluated.")
else:
    # Create a structured output for comparison
    # Loop through each metric type
    for metric_name in ['RMSE', 'MAE', 'MSE', 'R2']:
        print(f"\n{metric_name} Scores:")
        # Get max name length for nice formatting
        max_name_len = max(len(name) for name in evaluation_results.keys())
        # Print header row
        header = f"{'Model':<{max_name_len + 5}}" + "".join([f"{f'Day +{i + 1}':>10}" for i in range(len(targets))])
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        # Print scores for each model
        for model_name in sorted(evaluation_results.keys()):
            metrics = evaluation_results[model_name]
            if metric_name in metrics:
                scores = metrics[metric_name]
                score_line = f"{model_name:<{max_name_len + 5}}"
                for target in targets:
                    score_value = scores.get(target, np.nan)
                    # Special formatting for R2
                    if metric_name == 'R2':
                        score_line += f"{score_value:>10.4f}" if not np.isnan(score_value) else f"{'N/A':>10}"
                    # Standard formatting for RMSE, MAE, and MSE
                    else:
                        score_line += f"{score_value:>10.2f}" if not np.isnan(score_value) else f"{'N/A':>10}"
                print(score_line)
        print("-" * len(header))

print("\n--- 🎉 Model training and initial evaluation complete for all three models ---")





# %%
# ===================================================================

# === Final Model Selection and Visualization - Uses Test Metrics
# ===================================================================
# This cell summarizes the performance of all trained models,
# selects the best one, and generates the final forecast visualizations.

print("--- 🏆 Summarizing Evaluation Metrics and Selecting Best Model ---")

# Assuming evaluation_results dictionary is available from previous cells
if 'evaluation_results' not in globals() or not evaluation_results:
    print("❌ Error: Evaluation results not found. Please run the model training cells.")
else:
    # Create a DataFrame to easily compare average metrics
    model_comparison_df = pd.DataFrame({
        'Model': evaluation_results.keys(),
        'Average RMSE': [np.mean(list(metrics['RMSE'].values())) for metrics in evaluation_results.values()],
        'Average MAE': [np.mean(list(metrics['MAE'].values())) for metrics in evaluation_results.values()],
        'Average MSE': [np.mean(list(metrics['MSE'].values())) for metrics in evaluation_results.values()],
        'Average R2': [np.mean(list(metrics['R2'].values())) for metrics in evaluation_results.values()]
    })

    # Sort by Average RMSE (lower is better) to find the best model
    model_comparison_df = model_comparison_df.sort_values(by='Average RMSE')

    print("\n--- 📊 Model Comparison (Average Metrics on Test Data) ---")
    
    # Reorder columns and use print() for standardized console output
    print(model_comparison_df[['Model', 'Average RMSE', 'Average MAE', 'Average MSE', 'Average R2']].to_markdown(index=False))
# %%
    print("\n--- 📋 Detailed Evaluation Metrics per Target Day ---")

    all_detailed_metrics = []
    
    # Loop through all models and all metric types
    for model_name, metrics in evaluation_results.items():
        for metric_type in ['RMSE', 'MAE', 'MSE', 'R2']: # Added 'MSE' here
            if metric_type in metrics:
                row = {'Model': model_name, 'Metric': metric_type}
                # Add each target day's score to the row
                for i, target in enumerate(targets):
                    row[target] = metrics[metric_type].get(target, np.nan)
                all_detailed_metrics.append(row)
    
    df_detailed_metrics = pd.DataFrame(all_detailed_metrics)
    
    # Reorder columns for readability
    ordered_columns = ['Model', 'Metric'] + targets
    df_detailed_metrics = df_detailed_metrics[ordered_columns]
    
    # Format the numeric columns for better display
    # This formats R2 to 4 decimal places, and others to 2
    for col in targets:
        # Define a function to apply to each row
        def format_value(row):
            value = row[col]
            metric_type = row['Metric']
            
            # Handle non-numeric or NaN values
            if not (isinstance(value, (int, float)) and not np.isnan(value)):
                return "N/A"
            
            if metric_type == 'R2':
                return f"{value:.4f}"  # 4 decimals for R2
            else: # RMSE, MAE, MSE
                return f"{value:.2f}"  # 2 decimals for errors

        # Apply the formatting function to the column
        df_detailed_metrics[col] = df_detailed_metrics.apply(format_value, axis=1)
    
    # This section now works for all 4 metrics
    
    # Get a list of all metrics we need to display
    metric_types_to_display = df_detailed_metrics['Metric'].unique()
    
    # Define a separator line
    separator = "-" * 100 
    
    # Loop and print a separate table for each metric
    for metric in metric_types_to_display:
        print(f"\n{metric} Scores:")
        print(separator)
        
        # Filter the DataFrame for the current metric
        df_metric = df_detailed_metrics[df_detailed_metrics['Metric'] == metric]
        
        # Drop the 'Metric' column since it's in the title
        df_metric_display = df_metric.drop(columns=['Metric'])
        
        # Print the DataFrame as a string for aligned output
        print(df_metric_display.to_string(index=False))
        
        print(separator)

# --- End of modified section ---

    # Determine the best model based on Average RMSE
    best_model_name = model_comparison_df.iloc[0]['Model']
    print(f"\n🥇 Based on Average RMSE on the test data, the best model is: {best_model_name}")

    # --- Load the Best Model ---
    print(f"\n📤 Loading the best model: {best_model_name}")

    best_model = None
    model_list = []
    if best_model_name == 'RandomForest':
        best_model_path = 'random_forest_model.joblib'
        best_model = joblib.load(best_model_path)
        model_list = best_model.estimators_  # Get the list of 7 estimators
    elif best_model_name == 'Polynomial Regression':
        best_model_path = 'polynomial_regression_model.joblib'
        best_model = joblib.load(best_model_path)
        model_list = best_model.estimators_
    elif best_model_name == 'ANN':
        best_model_path = 'ann_model.h5'
        try:
            # Keras models need to be re-built and then load weights
            if 'build_ann_model' in globals():
                input_shape = (X_train.shape[1],)
                output_shape = y_train.shape[1]
                # Use the best parameters found during tuning
                best_ann_params = {
                    'n_layers': 4, 
                    'units': 128, 
                    'dropout': 0.1
                }
                best_model = build_ann_model(input_shape, output_shape, **best_ann_params)
                best_model.load_weights(best_model_path)
                # ANN is a single model, wrap it in a list for consistency in later steps
                model_list = [best_model]
            else:
                print("❌ Error: build_ann_model function not found. Cannot load ANN model.")
                best_model = None
                model_list = []
        except Exception as e:
            print(f"❌ Error loading ANN model from {best_model_path}: {e}")
            best_model = None
            model_list = []
    elif best_model_name == 'LinearRegression':
        best_model_path = 'linear_regression_multi_geo_model.joblib'
        best_model = joblib.load(best_model_path)
        model_list = best_model.estimators_

    if best_model:
        print(f"✅ Successfully loaded {best_model_name} model.")

        # --- Generate Forecast Matrix using the Best Model ---
        print(f"\n--- 🚀 Starting Forecast Generation using {best_model_name} ---")

        if 'df_geog_clean' not in globals() or 'df_imputed' not in globals() or 'features' not in globals():
            print("❌ Error: Required dataframes (df_geog_clean, df_imputed) or features list not found.")
        elif not model_list:
            print("❌ Error: Best model estimators could not be loaded.")
        else:
            all_cities = np.sort(df_geog_clean['City'].unique())
            all_forecasts_best_model = {}

            print(f"⏳ Generating forecasts for {len(all_cities)} cities using {best_model_name}. This may take a moment...")

            if best_model_name == 'ANN':
                # The ANN model predicts all 7 days at once,
                # so it needs a different prediction function.
                print("🧠 Using adapted prediction function for ANN.")

                # Use the ANN-specific function (defined at top)
                for city in all_cities:
                    forecast = get_forecast_for_city_ann(
                        city,
                        best_model,  # Pass the single ANN model
                        df_geog_clean,
                        df_imputed,
                        features
                    )
                    all_forecasts_best_model[city] = forecast

            else:
                # Use the original get_forecast_for_city for MultiOutputRegressor models (RF, Poly, Linear)
                if 'get_forecast_for_city' in globals():
                    for city in all_cities:
                        forecast = get_forecast_for_city(
                            city,
                            model_list,  # Pass the list of estimators
                            df_geog_clean,
                            df_imputed,
                            features
                        )
                        all_forecasts_best_model[city] = forecast
                else:
                    print("❌ Error: get_forecast_for_city function not found. Cannot generate forecasts.")
                    all_forecasts_best_model = {}

            if all_forecasts_best_model:
                print("✅ ...All forecasts generated successfully.")

                # --- 4. Create the Forecast "Matrix" (DataFrame) ---
                df_forecast_best_model = pd.DataFrame.from_dict(all_forecasts_best_model, orient='index')
                df_forecast_best_model.columns = [f'Day +{i}' for i in range(1, 8)]

                # Store the forecast dataframe in a dictionary by model name
                if 'forecast_dataframes' not in globals():
                    forecast_dataframes = {}
                forecast_dataframes[best_model_name] = df_forecast_best_model.copy()

                # Drop any cities that failed prediction
                initial_city_count = len(df_forecast_best_model)
                df_forecast_best_model = df_forecast_best_model.dropna()
                dropped_city_count = initial_city_count - len(df_forecast_best_model)
                if dropped_city_count > 0:
                    print(f"⚠️ Dropped {dropped_city_count} cities from the forecast matrix due to missing data.")

                # Save the matrix to a CSV file with model name
                csv_filename = f"{best_model_name.replace(' ', '_').lower()}_7_day_forecast_matrix.csv"
                df_forecast_best_model.to_csv(csv_filename)
                print(f"💾 Forecast matrix saved to '{csv_filename}'")

                # --- 5. Generate Heatmap ---
                sns.set_theme(style="white")
                n_cities = len(df_forecast_best_model)
                fig_height = max(8, n_cities * 0.4)

                plt.figure(figsize=(12, fig_height))
                heatmap = sns.heatmap(
                    df_forecast_best_model,
                    annot=True,
                    fmt=".0f",
                    linewidths=.5,
                    cmap="YlOrRd",
                    cbar_kws={'label': 'Predicted AQI'},
                    annot_kws={"size": 8}
                )

                # Add model name to title
                plt.title(f'7-Day AQI Forecast Matrix for All Cities ({best_model_name})', fontsize=16, pad=15)
                plt.xlabel('Forecast Day', fontsize=10)
                plt.ylabel('City', fontsize=10)
                plt.yticks(rotation=0)
                plt.xticks(rotation=45, ha='right')

                # Save the plot to a file with model name
                plot_filename = f"{best_model_name.replace(' ', '_').lower()}_aqi_forecast_heatmap.png"
                plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
                print(f"🖼️ Heatmap saved to '{plot_filename}'")

                # Show the plot
                plt.show()

                print(f"\n--- 🎉 Forecast Generation and Heatmap for {best_model_name} Complete ---")

            else:
                print("❌ Forecast generation failed.")

print("\n--- 🎉 Process Complete ---")
# %%
# ===================================================================
# === High-Risk Pollution Report (RandomForest)
# ===================================================================
# This cell creates two reports:
# 1. All events in the Top 15% (85th percentile) or over 175 AQI, sorted by AQI.
# 2. The *same list* of events, but grouped and sorted by Forecast Day.

# Check if the get_aqi_bucket function exists
if 'get_aqi_bucket' not in globals():
    print("❌ Error: 'get_aqi_bucket' function not found.")
    print("🔄 Please re-run the 'Model Evaluation Utility' cell.")
    raise NameError("'get_aqi_bucket' not found")

print("\n--- 🔥 Identifying High-Risk Forecasts (RandomForest) ---")
try:
    # Check if the forecast_dataframes dictionary and the RandomForest forecast exist
    if 'forecast_dataframes' not in globals() or 'RandomForest' not in forecast_dataframes:
        print("❌ Error: RandomForest forecast data not found.")
        print("🔄 Please ensure the evaluation and forecast generation cell was run successfully.")
        raise NameError("RandomForest forecast data missing")

    # Get the forecast DataFrame for the RandomForest model
    df_forecast_rf = forecast_dataframes['RandomForest']

    if df_forecast_rf.empty:
        print("❌ Error: RandomForest forecast DataFrame is empty.")
        raise ValueError("Empty forecast data")

    # Convert the forecast DataFrame to long format
    # This DataFrame ('df_long_rf') is created here and used for both reports
    df_long_rf = df_forecast_rf.stack().reset_index()
    df_long_rf.columns = ['City', 'Forecast_Day', 'Predicted_AQI']

    # --- REPORT 1: High-Risk Events (Sorted by Highest AQI) ---
    
    # 1. Calculate the 85th percentile
    q_85 = df_long_rf['Predicted_AQI'].quantile(0.85)
    
    # 2. Define the two filter conditions
    condition_upper_percentile = (df_long_rf['Predicted_AQI'] > q_85)
    condition_above_175 = (df_long_rf['Predicted_AQI'] > 175)
    
    # 3. Filter for rows that meet *either* condition (OR logic)
    # This is our master list of high-risk events
    df_high_risk_rf = df_long_rf[condition_upper_percentile | condition_above_175]
    
    # 4. Sort the results from highest to lowest AQI
    df_high_risk_sorted_by_aqi = df_high_risk_rf.sort_values(by='Predicted_AQI', ascending=False)

    print(f"☣️ REPORT 1: High-Risk Events (Top 15% [> {q_85:.0f} AQI] or > 175 AQI) - Sorted by AQI:")
    print("--------------------------------------------------------------------------------------")
    
    if df_high_risk_sorted_by_aqi.empty:
        print("ℹ️ No high-risk forecasts found.")
    else:
        # Print all high-risk results, sorted by AQI
        for row in df_high_risk_sorted_by_aqi.itertuples():
            aqi_bucket = get_aqi_bucket(row.Predicted_AQI)
            print(f"    City: {row.City}")
            print(f"    Day: {row.Forecast_Day}")
            print(f"    Forecast: {row.Predicted_AQI:.0f} AQI ({aqi_bucket})  ")
            print("")

    # --- MODIFIED BLOCK ---
    # --- REPORT 2: High-Risk Events (Grouped by Forecast Day) ---
    
    print("\n--------------------------------------------------------------------------------------")
    print(f"☣️ REPORT 2: Same High-Risk Events as Report 1, but Grouped by Forecast Day:")
    print("--------------------------------------------------------------------------------------")

    if df_high_risk_rf.empty:
        print("ℹ️ No high-risk forecasts found.")
    else:
        # 1. Take the *same* high-risk DataFrame (df_high_risk_rf)
        # 2. Sort it by 'Forecast_Day' (primary) and then 'Predicted_AQI' (secondary)
        df_high_risk_sorted_by_day = df_high_risk_rf.sort_values(
            by=['Forecast_Day', 'Predicted_AQI'], 
            ascending=[True, False]
        )
        
        # 3. Loop and print with day headings
        current_day = "" # Variable to track the day being printed
        for row in df_high_risk_sorted_by_day.itertuples():
            
            # If this row is for a new day, print the day heading
            if row.Forecast_Day != current_day:
                current_day = row.Forecast_Day
                print(f"\n        --- {current_day} ---")
            
            # Print the city and forecast details, indented under the day
            aqi_bucket = get_aqi_bucket(row.Predicted_AQI)
            print(f"        City: {row.City:<15} |  Forecast: {row.Predicted_AQI:.0f} AQI ({aqi_bucket})  ")

except NameError as e:
    print(f"❌ A required object or function was not found: {e}")
except ValueError as e:
    print(f"❌ Data error: {e}")
except Exception as e:
    print(f"❌ An error occurred while identifying high-risk cases: {e}")
# %%
# ===================================================================
# === Past vs. Future Heatmap (Best Model - RandomForest)
# ===================================================================

print("\n--- 📅 Generating 30-Day Past vs. 7-Day Future Heatmap (RandomForest) ---")
try:
    # Check if required dataframes and forecast data exist
    if 'df_imputed' not in globals() or 'forecast_dataframes' not in globals() or 'RandomForest' not in forecast_dataframes:
        print("❌ Error: 'df_imputed' or RandomForest forecast data missing.")
        raise NameError("Missing required Dataframes or forecast data")

    # Get the forecast DataFrame for the RandomForest model
    df_forecast_rf = forecast_dataframes['RandomForest']

    if df_forecast_rf.empty:
        print("❌ Error: RandomForest forecast DataFrame is empty. Cannot generate heatmap.")
        raise ValueError("Empty forecast data for RandomForest")

    # Ensure we only process cities that are present in the RandomForest forecast
    cities_to_process = df_forecast_rf.index.tolist()

    print(f"🔍 Extracting last 30 days of actual data for {len(cities_to_process)} cities...")
    # 1. Extract last 30 days of actual data
    past_data = {}
    for city in cities_to_process:
        # Sort by date to ensure we get the *last* 30 days
        city_aqi_history = df_imputed[df_imputed['City'] == city].sort_values(by='Date')['AQI'].values
        if len(city_aqi_history) >= 30:
            past_data[city] = list(city_aqi_history[-30:])
        else:
            # Pad with NaNs if less than 30 days available
            pad_width = 30 - len(city_aqi_history)
            past_data[city] = [np.nan] * pad_width + list(city_aqi_history)

    df_past = pd.DataFrame.from_dict(past_data, orient='index')
    df_past.columns = [f'Day -{i}' for i in range(30, 0, -1)]

    # 2. Concatenate past data with the RandomForest forecast data
    df_combined = pd.concat([df_past, df_forecast_rf], axis=1)

    # Drop cities where past data extraction OR forecast data failed
    df_combined = df_combined.dropna(subset=df_forecast_rf.columns)
    df_combined = df_combined.dropna(subset=df_past.columns)

    if df_combined.empty:
        print("❌ Error: The combined DataFrame is empty. No cities had both past and future data.")
        raise ValueError("Empty combined DataFrame for plotting")

    print(f"✅ Combined matrix created with shape: {df_combined.shape}")

    # 3. Plot the combined heatmap
    sns.set_theme(style="white")
    n_cities = len(df_combined)
    fig_height = max(8, n_cities * 0.4) # Dynamic height

    plt.figure(figsize=(20, fig_height))

    sns.heatmap(
        df_combined,
        annot=True,
        fmt=".0f",
        linewidths=.5,
        cmap="YlOrRd",
        cbar_kws={'label': 'AQI Value (Actual & Predicted)'},
        annot_kws={"size": 7}
    )

    # 4. Add divider and labels
    plt.axvline(x=30, color='blue', linestyle='--', linewidth=2)
    plt.text(15, -0.5, 'PAST 30 DAYS (Actual)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='black')
    plt.text(33.5, -0.5, 'FUTURE 7 DAYS (Predicted - RandomForest)', ha='center', va='center',
             fontsize=12, fontweight='bold', color='blue')

    plt.title('30-Day Past (Actual) vs. 7-Day Future (Predicted) AQI Matrix (RandomForest)', fontsize=16, pad=40)
    plt.xlabel('Day Relative to Present', fontsize=10)
    plt.ylabel('City', fontsize=10)
    plt.yticks(rotation=0)

    # 5. Adjust x-tick labels for readability
    past_ticks_indices = np.arange(0, 30, 5)  # Ticks at -30, -25, ..., -5
    future_ticks_indices = np.arange(30, 37)  # Ticks at +1, +2, ..., +7

    all_cols = df_combined.columns.tolist()
    selected_labels = [all_cols[i] for i in np.concatenate([past_ticks_indices, future_ticks_indices])]
    selected_ticks = np.concatenate([past_ticks_indices + 0.5, future_ticks_indices + 0.5])  # Center ticks

    plt.xticks(selected_ticks, selected_labels, rotation=45, ha='right', fontsize=8)

    # Save the plot
    plot_filename = "aqi_past_future_heatmap_randomforest.png"
    plt.savefig(plot_filename, bbox_inches='tight', dpi=200)
    print(f"🖼️ Combined Past/Future heatmap saved to '{plot_filename}'")

    plt.show()

    print("\n--- 🎉 Combined Matrix Process Complete ---")

except NameError as e:
    print(f"❌ A required object was not found: {e}")
except ValueError as e:
    print(f"❌ Data error: {e}")
except Exception as e:
    print(f"❌ An error occurred during plotting: {e}")