# python-ml-aqi-forecast
A complete machine learning pipeline to clean, process, and model Air Quality Index (AQI) data. Trains multiple models (RandomForest, ANN) to generate a 7-day forecast and visualizes results on maps
# 7-Day AQI Forecasting Pipeline

This project is a complete machine learning pipeline to clean, process, and model Air Quality Index (AQI) data. It trains multiple models (RandomForest, Polynomial Regression, ANN), selects the best one, and generates a 7-day forecast visualized on heatmaps and geographical maps.

## Project Structure

This pipeline is broken into three main scripts, intended to be run in order:

1.  **`01_Data_Processing.py`**
    * Loads the raw `city_day.csv` dataset.
    * Performs imputation for missing pollutant data using `KNNImputer`.
    * Imputes the target `AQI` column using `LinearRegression`.
    * Saves the final clean dataset as `city_day_cleaned_imputed.csv`.

2.  **`02_PY_ML_Assessment.py`**
    * Loads the cleaned data from Step 1.
    * Engineers time-series features (lags, rolling averages) and geographical features.
    * Trains and evaluates four models: Linear Regression, Polynomial Regression, RandomForest, and a Keras ANN.
    * Hyperparameter tunes the models and selects the best performer (RandomForest).
    * Generates a 7-day forecast and prints a "High-Risk Pollution Report" to the console.

3.  **`03_Map_AQI.py`**
    * Uses a **hardcoded text block** (copied from the output of script 2).
    * Uses `Cartopy` to plot the high-risk cities on a map of India.
    * Generates a separate map image (`.png`) for each of the 7 forecast days.

---

## How to Run This Project

Follow these steps exactly to run the pipeline from start to finish.

### Step 1: Data Requirements

Before you begin, you must place the following two (2) data files in the same folder as the Python scripts:

1.  `city_day.csv` (Used by `01_Data_Processing.py`)
2.  `cities_geog.csv` (Used by `02_PY_ML_Assessment.py`)

### Step 2: Setup Python Environment

1.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  Create the `requirements.txt` file (as shown above) in your project folder.

3.  Install all required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Step 3: CRITICAL FIX (Filename Mismatch)

Your scripts have a filename mismatch. You must fix it before running.

* `01_Data_Processing.py` **saves** its output as: `city_day_cleaned_imputed.csv`
* `02_PY_ML_Assessment.py` **looks for** a file named: `city_day_imputed_aqi.csv`

**To fix this:**
Open `02_PY_ML_Assessment.py` and find this line (around line 655):

```python
aqi_data_path = os.path.join(base_path, "city_day_imputed_aqi.csv")
