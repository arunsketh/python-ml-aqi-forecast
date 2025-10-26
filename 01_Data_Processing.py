"""
Enhanced Data Cleaning and Imputation Pipeline with Visualizations
===================================================================

This script performs a data cleaning and imputation pipeline
on the 'city_day.csv' dataset with comprehensive visualizations.

The pipeline performs the following steps:
  - Loads the raw 'city_day.csv' data.
  - Visualizes initial data gaps and missing patterns.
  - Drops rows that are unrecoverable (i.e., missing 'AQI' AND all pollutant data).
  - Uses KNNImputer to fill in missing values for the pollutant features
    (e.g., PM2.5, NO2, etc.).
  - Visualizes KNN imputation effects with cluster analysis.
  - Uses Linear Regression to impute missing 'AQI' values based on the
    now-complete pollutant data.
  - Saves this fully cleaned dataset as 'city_day_imputed_aqi.csv'.
"""


# --- Import Necessary Libraries ---

# pandas: For data manipulation and analysis (DataFrames)
import pandas as pd
# numpy: For numerical operations (often used by pandas)
import numpy as np
# KNNImputer: For imputing missing pollutant values
from sklearn.impute import KNNImputer
# LinearRegression: For imputing missing AQI values
from sklearn.linear_model import LinearRegression
# PCA: For dimensionality reduction to visualize clusters
from sklearn.decomposition import PCA
# matplotlib and seaborn: For visualization
import matplotlib.pyplot as plt
import seaborn as sns
# warnings: To suppress unnecessary warnings and keep the output clean
import warnings


# --- Global Configuration ---

# Suppress warnings that are not critical, improving output readability.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set random seed for reproducibility
np.random.seed(42)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# ==============================================================================
# === PART 1: LOAD AND EXPLORE DATA
# ==============================================================================
print("=" * 80)
print("ENHANCED DATA CLEANING AND IMPUTATION PIPELINE")
print("=" * 80)
print("\n--- Starting: Data Cleaning and Imputation with Visualizations ---\n")


# --- 1.1: Loading Data ---
print("--- Step 1.1: Loading Data ---")
try:
    # Attempt to read the raw dataset from the CSV file
    df = pd.read_csv("city_day.csv")
    print(f"✓ Successfully loaded 'city_day.csv'")
    print(f"  Original shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
except FileNotFoundError:
    # Handle the case where the file is not in the same directory
    print("✗ Error: city_day.csv not found.")
    print("  Please make sure the file is in the same directory as this script.")
    exit()
except Exception as e:
    # Handle other potential read errors
    print(f"✗ An error occurred while reading the file: {e}")
    exit()


# --- 1.2: Define Column Groups ---

# These are the columns that represent different chemical pollutants
pollutant_cols = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]

# This is the 'Air Quality Index'
aqi_col = 'AQI'

print(f"\n  Pollutant columns to impute: {pollutant_cols}")
print(f"  Target column: {aqi_col}")


# ==============================================================================
# === PART 2: VISUALIZE INITIAL DATA GAPS
# ==============================================================================
print("\n" + "=" * 80)
print("--- Step 2: Visualizing Initial Data Gaps ---")
print("=" * 80)

# Calculate missing value statistics
missing_counts = df[pollutant_cols].isnull().sum()
missing_pct = (missing_counts / len(df)) * 100

print("\nMissing Value Statistics:")
print("-" * 50)
for col in pollutant_cols:
    print(f"  {col:12s}: {missing_counts[col]:6d} missing ({missing_pct[col]:5.2f}%)")
print("-" * 50)
print(f"  Total rows: {len(df)}")

# --- Visualization 2.1: Missing Values Heatmap ---
print("\n→ Creating missing values heatmap...")

# Create a binary mask for missing values (1 for missing, 0 for present)
missing_mask = df[pollutant_cols].isna().astype(int)

# Plot heatmap for a sample of rows
sample_rows = min(1000, len(df))
plt.figure(figsize=(12, 8))
sns.heatmap(missing_mask.head(sample_rows),
            yticklabels=False,
            xticklabels=pollutant_cols,
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Missing (1) / Present (0)'})
plt.title(f'Data Gaps in Pollutant Columns (First {sample_rows} Rows)',
          fontsize=14, fontweight='bold')
plt.xlabel('Pollutant Features', fontsize=12)
plt.ylabel('Row Index', fontsize=12)
plt.tight_layout()
plt.savefig('01_data_gaps_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 01_data_gaps_heatmap.png")
plt.close()

# --- Visualization 2.2: Missing Percentages Bar Chart ---
print("→ Creating missing percentages bar chart...")

plt.figure(figsize=(12, 6))
colors = plt.cm.RdYlGn_r(missing_pct / 100)
bars = plt.bar(range(len(pollutant_cols)), missing_pct, color=colors, edgecolor='black', linewidth=1.5)
plt.xticks(range(len(pollutant_cols)), pollutant_cols, rotation=45, ha='right')
plt.title('Percentage of Missing Values per Pollutant Column',
          fontsize=14, fontweight='bold')
plt.xlabel('Pollutant Features', fontsize=12)
plt.ylabel('Missing Percentage (%)', fontsize=12)
plt.ylim(0, max(missing_pct) * 1.1)

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, missing_pct)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('02_missing_percentages_bar.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 02_missing_percentages_bar.png")
plt.close()

# --- Visualization 2.3: Missing Value Patterns ---
print("→ Creating missing value pattern analysis...")

# Count combinations of missing patterns
missing_pattern = df[pollutant_cols].isnull().astype(int)
pattern_counts = missing_pattern.groupby(pollutant_cols).size().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(range(len(pattern_counts)), pattern_counts.values, color='coral', edgecolor='black')
plt.yticks(range(len(pattern_counts)), [f'Pattern {i+1}' for i in range(len(pattern_counts))])
plt.xlabel('Number of Rows', fontsize=12)
plt.ylabel('Missing Pattern', fontsize=12)
plt.title('Top 10 Most Common Missing Value Patterns', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_missing_patterns.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 03_missing_patterns.png")
plt.close()


# ==============================================================================
# === PART 3: DROP UNRECOVERABLE ROWS
# ==============================================================================
print("\n" + "=" * 80)
print("--- Step 3: Dropping Unrecoverable Rows ---")
print("=" * 80)

# We identify rows that are impossible to use or impute meaningfully.
# These are rows where we have NO target (AQI) AND NO features (pollutants).

# Condition 1: The 'AQI' value is missing (NaN)
condition1 = df[aqi_col].isna()

# Condition 2: ALL of the pollutant columns are also missing
condition2 = df[pollutant_cols].isna().all(axis=1)

# We want rows where *both* conditions are true
rows_to_drop = df[condition1 & condition2].index

print(f"\n  Found {len(rows_to_drop)} unrecoverable rows")
print(f"  (where 'AQI' AND all pollutant columns are missing)")

# Drop these rows from the DataFrame
df_cleaned = df.drop(rows_to_drop).copy()

print(f"\n  Shape before: {df.shape}")
print(f"  Shape after:  {df_cleaned.shape}")
print(f"  Rows dropped: {len(rows_to_drop)}")


# ==============================================================================
# === PART 4: STORE PRE-IMPUTATION DATA FOR COMPARISON
# ==============================================================================
print("\n" + "=" * 80)
print("--- Step 4: Storing Pre-Imputation State ---")
print("=" * 80)

# Store original data before imputation for visualization
df_before_imputation = df_cleaned[pollutant_cols].copy()
print(f"  ✓ Stored original pollutant data for comparison")


# ==============================================================================
# === PART 5: IMPUTE MISSING POLLUTANT DATA (KNN)
# ==============================================================================
print("\n" + "=" * 80)
print("--- Step 5: Imputing Missing Pollutant Data (KNNImputer) ---")
print("=" * 80)

print("\n→ Initializing KNNImputer with n_neighbors=5...")
print("  (This looks at the 5 most similar rows to fill missing values)")

# Initialize the imputer
imputer = KNNImputer(n_neighbors=5)

# Apply the imputer
print("\n→ Running KNN imputation on pollutant columns...")
df_cleaned[pollutant_cols] = imputer.fit_transform(df_cleaned[pollutant_cols])

print("  ✓ Imputation of pollutant columns complete")
print(f"  ✓ All pollutant columns now have 0 missing values")


# ==============================================================================
# === PART 6: VISUALIZE KNN IMPUTATION EFFECTS
# ==============================================================================
print("\n" + "=" * 80)
print("--- Step 6: Visualizing KNN Imputation Effects ---")
print("=" * 80)

# Use a subset for visualization efficiency
subset_size = min(2000, len(df_cleaned))
subset_idx = np.random.choice(df_cleaned.index, subset_size, replace=False)

df_subset_before = df_before_imputation.loc[subset_idx].copy()
df_subset_after = df_cleaned.loc[subset_idx, pollutant_cols].copy()

print(f"\n→ Using subset of {subset_size} rows for visualization")

# --- Visualization 6.1: PCA Cluster Analysis ---
print("\n→ Creating PCA cluster visualization...")

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)

# For "before" data, temporarily fill NaNs with column mean for PCA
df_subset_before_filled = df_subset_before.fillna(df_subset_before.mean())
pca_before = pca.fit_transform(df_subset_before_filled)

# For "after" data (no NaNs)
pca_after = pca.transform(df_subset_after)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Before imputation
has_missing = df_subset_before.isnull().any(axis=1)
axes[0].scatter(pca_before[~has_missing, 0], pca_before[~has_missing, 1],
                c='#2E86AB', alpha=0.6, label='Complete Rows', s=30, edgecolors='black', linewidth=0.5)
axes[0].scatter(pca_before[has_missing, 0], pca_before[has_missing, 1],
                c='#E63946', alpha=0.8, marker='x', s=80, linewidth=2, label='Rows with Missing Values')
axes[0].set_title('Before KNN Imputation (PCA 2D Projection)', fontsize=13, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Subplot 2: After imputation
axes[1].scatter(pca_after[:, 0], pca_after[:, 1],
                c='#06A77D', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
axes[1].set_title('After KNN Imputation (n_neighbors=5)', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_knn_clusters_before_after.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 04_knn_clusters_before_after.png")
plt.close()

# --- Visualization 6.2: 2D Pollutant Scatter (PM2.5 vs PM10) ---
print("→ Creating PM2.5 vs PM10 scatter comparison...")

pm25_col = 'PM2.5'
pm10_col = 'PM10'

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before imputation
complete_mask = df_subset_before[[pm25_col, pm10_col]].notna().all(axis=1)
axes[0].scatter(df_subset_before.loc[complete_mask, pm25_col],
                df_subset_before.loc[complete_mask, pm10_col],
                c='#2E86AB', alpha=0.6, label='Complete Data', s=40, edgecolors='black', linewidth=0.5)

# Mark rows with missing values
incomplete_mask = ~complete_mask
missing_pm25 = df_subset_before[pm25_col].isna()
missing_pm10 = df_subset_before[pm10_col].isna()

if missing_pm25.any():
    axes[0].scatter(df_subset_before.loc[missing_pm25 & ~missing_pm10, pm10_col],
                    [0] * sum(missing_pm25 & ~missing_pm10),
                    c='#E63946', marker='x', s=100, linewidth=2,
                    label=f'{pm25_col} Missing', alpha=0.8)

if missing_pm10.any():
    axes[0].scatter([0] * sum(missing_pm10 & ~missing_pm25),
                    df_subset_before.loc[missing_pm10 & ~missing_pm25, pm25_col],
                    c='#F77F00', marker='+', s=100, linewidth=2,
                    label=f'{pm10_col} Missing', alpha=0.8)

axes[0].set_title(f'Before Imputation: {pm25_col} vs {pm10_col}', fontsize=13, fontweight='bold')
axes[0].set_xlabel(pm25_col, fontsize=11)
axes[0].set_ylabel(pm10_col, fontsize=11)
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(True, alpha=0.3)

# After imputation
axes[1].scatter(df_subset_after[pm25_col], df_subset_after[pm10_col],
                c='#06A77D', alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
axes[1].set_title(f'After KNN Imputation: {pm25_col} vs {pm10_col}', fontsize=13, fontweight='bold')
axes[1].set_xlabel(pm25_col, fontsize=11)
axes[1].set_ylabel(pm10_col, fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_knn_2d_scatter_pm25_pm10.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 05_knn_2d_scatter_pm25_pm10.png")
plt.close()

# --- Visualization 6.3: Distribution Comparison ---
print("→ Creating distribution comparison plots...")

# Select 4 key pollutants for distribution comparison
key_pollutants = ['PM2.5', 'PM10', 'NO2', 'O3']

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Distribution Comparison: Before vs After KNN Imputation',
             fontsize=15, fontweight='bold', y=1.00)

for idx, pollutant in enumerate(key_pollutants):
    # Before imputation - histogram
    axes[0, idx].hist(df_before_imputation[pollutant].dropna(), bins=50,
                      color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, idx].set_title(f'{pollutant} - Before', fontsize=11, fontweight='bold')
    axes[0, idx].set_xlabel('Value', fontsize=10)
    axes[0, idx].set_ylabel('Frequency', fontsize=10)
    axes[0, idx].grid(True, alpha=0.3)

    # After imputation - histogram
    axes[1, idx].hist(df_cleaned[pollutant], bins=50,
                      color='#06A77D', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, idx].set_title(f'{pollutant} - After', fontsize=11, fontweight='bold')
    axes[1, idx].set_xlabel('Value', fontsize=10)
    axes[1, idx].set_ylabel('Frequency', fontsize=10)
    axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_distribution_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: 06_distribution_comparison.png")
plt.close()


# ==============================================================================
# === PART 7: IMPUTE MISSING AQI DATA (LINEAR REGRESSION)
# ==============================================================================
print("\n" + "=" * 80)
print("--- Step 7: Imputing Missing 'AQI' Data (Linear Regression) ---")
print("=" * 80)

# Separate the data into two parts
data_with_aqi = df_cleaned[df_cleaned[aqi_col].notna()]
data_missing_aqi = df_cleaned[df_cleaned[aqi_col].isna()]

print(f"\n  Rows with AQI (for training):    {data_with_aqi.shape[0]:,}")
print(f"  Rows missing AQI (for prediction): {data_missing_aqi.shape[0]:,}")

# We can only proceed if we have rows to predict and rows to train on
if not data_missing_aqi.empty and not data_with_aqi.empty:
    # Define the features (X) and target (y) for the training data
    X_train = data_with_aqi[pollutant_cols]
    y_train = data_with_aqi[aqi_col]

    # Initialize and train the Linear Regression model
    print("\n→ Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate and display model performance
    train_score = model.score(X_train, y_train)
    print(f"  ✓ Model R² score on training data: {train_score:.4f}")

    # Define the features (X) for the data we want to predict
    X_predict = data_missing_aqi[pollutant_cols]

    # Predict the missing AQI values
    predicted_aqi = model.predict(X_predict)

    print(f"\n→ Predicting {len(predicted_aqi):,} missing AQI values...")

    # Fill the NaNs in the original 'df_cleaned' DataFrame
    df_cleaned.loc[df_cleaned[aqi_col].isna(), aqi_col] = predicted_aqi

    print(f"  ✓ Imputation of 'AQI' complete")
    print(f"  ✓ Predicted AQI range: [{predicted_aqi.min():.2f}, {predicted_aqi.max():.2f}]")

elif data_missing_aqi.empty:
    print("\n  ℹ No missing 'AQI' values found. Skipping 'AQI' imputation.")
else:
    print("\n  ⚠ Warning: No 'AQI' data available to train the imputation model.")
    print("  'AQI' column will remain with NaN values.")


# ==============================================================================
# === PART 8: SAVE CLEANED FILE
# ==============================================================================
print("\n" + "=" * 80)
print("--- Step 8: Saving Cleaned and Imputed Data ---")
print("=" * 80)

output_filename = "city_day_imputed_aqi.csv"

# Create a copy of the DataFrame for formatting before saving
df_cleaned_formatted = df_cleaned.copy()

# Apply formatting to pollutant columns (1 decimal place)
for col in pollutant_cols:
    df_cleaned_formatted[col] = df_cleaned_formatted[col].round(1)

# Apply formatting to AQI column (no decimal places)
df_cleaned_formatted[aqi_col] = df_cleaned_formatted[aqi_col].round(0)

try:
    # Save the formatted DataFrame to a new CSV file.
    # index=False prevents pandas from writing the DataFrame's row index (0, 1, 2...)
    # as a new column in the CSV.
    df_cleaned_formatted.to_csv(output_filename, index=False)
    print(f"\n  ✓ Successfully saved to '{output_filename}'")
    print(f"  ✓ Final shape: {df_cleaned_formatted.shape}")
    print(f"  ✓ File size: ~{df_cleaned_formatted.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
except PermissionError:
    print(f"\n  ✗ Error: Permission denied. Could not write to '{output_filename}'.")
except Exception as e:
    print(f"\n  ✗ An error occurred while saving the file: {e}")


# ==============================================================================
# === PART 9: FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("--- PIPELINE COMPLETED SUCCESSFULLY ---")
print("=" * 80)

print("\n📊 SUMMARY OF CHANGES:")
print("-" * 50)
print(f"  Original dataset:        {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Unrecoverable rows:      {len(rows_to_drop):,} removed")
print(f"  Final dataset:           {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns")
print(f"  Pollutant columns:       All {len(pollutant_cols)} fully imputed (KNN)")
print(f"  AQI column:              Imputed via Linear Regression")
print("-" * 50)

print("\n📁 GENERATED FILES:")
print("-" * 50)
print("  Data:")
print(f"    • {output_filename}")
print("\n  Visualizations:")
print("    • 01_data_gaps_heatmap.png")
print("    • 02_missing_percentages_bar.png")
print("    • 03_missing_patterns.png")
print("    • 04_knn_clusters_before_after.png")
print("    • 05_knn_2d_scatter_pm25_pm10.png")
print("    • 06_distribution_comparison.png")
print("-" * 50)

print("\n📖 INTERPRETATION GUIDE:")
print("-" * 50)
print("  • Red markers in PCA plots = rows with missing values before imputation")
print("  • Green clusters = all data after KNN imputation (n=5 neighbors)")
print("  • PCA captures major variance patterns in 12-dimensional pollutant space")
print("  • KNN preserves cluster structure while filling gaps intelligently")
print("  • Distribution plots show imputed values blend naturally with originals")
print("-" * 50)

print("\n✓ All operations completed successfully!")
print("\n" + "=" * 80)

# Display sample of cleaned data
print("\n🔍 SAMPLE OF CLEANED DATA (First 5 rows):")
print("=" * 80)
# Format the output to show pollutants with one decimal and AQI with no decimals
format_dict = {col: '{:.1f}'.format for col in pollutant_cols}
format_dict[aqi_col] = '{:.0f}'.format
print(df_cleaned.head().to_string(formatters=format_dict))

print("\n" + "=" * 80)

