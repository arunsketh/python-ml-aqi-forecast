Here is the `README.md` content rewritten to use `**` formatting for headers.

-----

**AQI 7-Day Forecasting Pipeline**

This project is an end-to-end machine learning pipeline to clean, process, and model Air Quality Index (AQI) data. It performs a comprehensive data imputation pipeline, then trains and evaluates multiple models (`RandomForest`, `PolynomialRegression`, `ANN`, `LinearRegression`) to find the best performer. Finally, it uses the best model to generate a 7-day forecast and visualizes the highest-risk areas on a geographical map of India.

**How to Run in Google Colab**

Follow these steps to run the complete pipeline in a Google Colab notebook.

**Step 1: Setup Environment**

Clone the repository and move into the new directory. This single command downloads all scripts and required CSV files (`city_day.csv`, `cities_geog.csv`).

```bash
!git clone https://github.com/arunsketh/python-ml-aqi-forecast.git
%cd python-ml-aqi-forecast
```

**Step 2: Install Dependencies**

Run this command in a Colab cell to install all required libraries for the three scripts.

```bash
!pip install pandas numpy scikit-learn matplotlib seaborn tensorflow altair cartopy joblib
```

**Step 3: Run the Pipeline (3 Scripts)**

The pipeline must be run in order.

**1. Run the Data Processing Script**

This script loads `city_day.csv`, performs advanced imputation (using `KNNImputer` for pollutants and `LinearRegression` for AQI), and saves the clean data.

```bash
!python 01_Data_Processing.py
```

  * **Input:** `city_day.csv`
  * **Output:** `city_day_imputed_aqi.csv` (plus 6 visualization images like `01_data_gaps_heatmap.png`)

**2. Run the Model Training & Forecasting Script**

This is the main script. It loads the cleaned data, trains all models, tunes them, selects the best one (RandomForest, based on the script's logic), and generates a 7-day forecast.

At the very end, it will print a **"High-Risk Pollution Report"** to your console.

```bash
!python 02_PY_ML_Assessment.py
```

  * **Input:** `city_day_imputed_aqi.csv`, `cities_geog.csv`
  * **Output:** Prints a text report to the console (see next step). Also saves model files (`.joblib`, `.h5`) and heatmap images (`randomforest_aqi_forecast_heatmap.png`).

**3. CRITICAL STEP: Update the Map Script**

This pipeline requires a **manual copy-paste** step.

1.  Look at the console output from running `02_PY_ML_Assessment.py`.

2.  Find the **"REPORT 2: ... Grouped by Forecast Day"** section and **copy the entire report**, including the `--- Day +1 ---` headers.

    *Your copied text will look something like this:*

    ```
            --- Day +1 ---
            City: Delhi            |  Forecast: 138 AQI (Moderate)
            City: Brajrajnagar     |  Forecast: 124 AQI (Moderate)
            ...
            --- Day +7 ---
            City: Brajrajnagar     |  Forecast: 142 AQI (Moderate)
            ...
    ```

3.  In the Colab file browser (left-hand sidebar), double-click the `03_Map_AQI.py` file to open it.

4.  Find the `forecast_text = """ ... """` variable (around line 20).

5.  **Delete** the old text inside the `"""` and **paste your new report** from the console.

**Before:**

```python
forecast_text = """
--- Day +1 ---
        City: Delhi            |  Forecast: 138 AQI (Moderate)
...
"""
```

**After (pasting your new output):**

```python
forecast_text = """
        --- Day +1 ---
        City: Ahmedabad        |  Forecast: 205 AQI (Poor)
        City: Delhi            |  Forecast: 198 AQI (Moderate)
        ... (and so on for all 7 days) ...
"""
```

6.  Save the file (Ctrl+S or File \> Save).

**4. Run the Map Generation Script**

Now that the script has your custom forecast data, you can run it to generate the maps.

```bash
!python 03_Map_AQI.py
```

  * **Input:** The `forecast_text` you just pasted.
  * **Output:** 7 map images (`aqi_map_Day_1.png`, `aqi_map_Day_2.png`, etc.) saved to your Colab folder.

-----

**Project Pipeline Explained**

**1. `01_Data_Processing.py`**

This script is responsible for cleaning and imputing the raw data.

  * **Loads:** `city_day.csv`.
  * **Visualizes Gaps:** Generates and saves 3 plots (`01_data_gaps_heatmap.png`, `02_missing_percentages_bar.png`, `03_missing_patterns.png`) to show missing data.
  * **Imputes Pollutants:** Uses `sklearn.impute.KNNImputer` to fill missing values for all 12 pollutant features (e.g., `PM2.5`, `NO2`, `O3`).
  * **Visualizes Imputation:** Generates and saves 3 more plots (`04_knn_clusters_before_after.png`, `05_knn_2d_scatter_pm25_pm10.png`, `06_distribution_comparison.png`) to prove the imputation was effective.
  * **Imputes Target (AQI):** Trains a `LinearRegression` model on the imputed pollutants to predict and fill any missing `AQI` values.
  * **Saves:** The fully cleaned and imputed dataset as `city_day_imputed_aqi.csv`.

**2. `02_PY_ML_Assessment.py`**

This is the core machine learning script that trains, tunes, evaluates, and forecasts.

  * **Loads:** `city_day_imputed_aqi.csv` and `cities_geog.csv`.
  * **Feature Engineering:** Merges the datasets and creates time-series features (e.g., `AQI_lag_1`, `AQI_lag_7`, `AQI_roll_mean_7`) and 7 target columns (`target_day_1`...`target_day_7`).
  * **Train/Test Split:** Splits the data 80% for training and 20% for testing.
  * **Model Training & Evaluation:** Trains four different models:
    1.  `LinearRegression` (as a baseline)
    2.  `RandomForestRegressor`
    3.  `PolynomialRegression`
    4.  A Keras/TensorFlow `ANN` (Artificial Neural Network)
  * **Hyperparameter Tuning:** Automatically tests a range of parameters for each model (e.g., `n_estimators` for RandomForest, `degree` for Polynomial) to find the best settings, saving plots of the tuning results.
  * **Model Selection:** Compares all tuned models on the test set and selects the best one based on the lowest average Root Mean Squared Error (RMSE). The script is designed to select `RandomForest` as the final model.
  * **Forecast Generation:** Uses the best model (`RandomForest`) to perform an autoregressive 7-day forecast for every city.
  * **Report Generation:** Prints two reports to the console: one sorted by highest AQI and one **grouped by day** (this is the one used by the next script).
  * **Saves:** Multiple heatmaps (`randomforest_aqi_forecast_heatmap.png`, `aqi_past_future_heatmap_randomforest.png`) and the final forecast data (`randomforest_7_day_forecast_matrix.csv`).

**3. `03_Map_AQI.py`**

This script is purely for visualizing the high-risk forecast generated by the previous step.

  * **Input:** Relies on the user manually pasting the console report into the `forecast_text` variable.
  * **Parses Data:** Reads the hardcoded string and converts it into a pandas DataFrame.
  * **Generates Maps:** Loops through each of the 7 forecast days.
  * **Plots:** Uses `Cartopy` and `Matplotlib` to plot the location of each high-risk city on a map of India, coloring each city by its predicted AQI level.
  * **Saves:** A separate map image for each day (`aqi_map_Day_1.png`, `aqi_map_Day_2.png`, etc.).
