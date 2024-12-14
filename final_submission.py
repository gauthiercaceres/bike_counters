# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime, date
import warnings
import gc
import importlib
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.preprocessing import LabelEncoder
import pickle
import sys
sys.path.append('/kaggle/input/ext-dataset/external_data')
import importlib
import example_estimator
importlib.reload(example_estimator)
pd.pandas.set_option('display.max_columns', None)
from sklearn.cluster import KMeans


# %%
#!pip install geopy
from geopy.distance import distance

# %% [markdown]
# We are starting by loading the train and test data.
# We then use functions from example_estimator file to merge with weather data and encode the dates.

# %%
# Data paths
train_data = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")
test_data = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")

train_data = example_estimator._encode_dates(train_data)
train_data = example_estimator._merge_external_data(train_data)
print(train_data.columns)

# %% [markdown]
# Now,
# Let us create a set of features that can be derived from the data that we have

# %%

def preprocess_and_engineer_features(data, target):
    """
    Preprocesses the input DataFrame by engineering features for bike usage prediction.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing raw data.
        target : bolean

    Returns:
        pd.DataFrame: The processed DataFrame with engineered features.
    """
    # Feature: Weekend indicator
    data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)

    # Feature: Distance to Paris city center
    city_center = (48.8566, 2.3522)
    data['distance_to_center'] = data.apply(
        lambda row: distance(city_center, (row['latitude'], row['longitude'])).km,
        axis=1
    )

    # Weather-related features
    data['is_rainy'] = (data['rr1'] > 0).astype(int)
    data['is_windy'] = (data['ff'] > 8).astype(int)
    data['low_visibility'] = (data['vv'] < 1000).astype(int)

    # Temperature-related features
    data['temperature_c'] = data['t'] - 273.15  # Convert Kelvin to Celsius
    data['is_hot'] = (data['temperature_c'] > 20).astype(int)
    data['is_cold'] = (data['temperature_c'] < 5).astype(int)
    data['is_snowy'] = (data['temperature_c'] < 0).astype(int)

    # Day period based on hour
    data['day_period'] = pd.cut(
        data['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        right=False
    )

    # Rename columns for consistency
    data.rename(columns={
        'workday': 'workingday',
        'temperature_c': 'temp',
        'u': 'humidity',
        'ff': 'windspeed'
    }, inplace=True)

    # Feature: Apparent temperature
    if 'atemp' not in data.columns:
        data['atemp'] = np.where(data['is_windy'] == 1, data['temp'] - 2, data['temp'])

    # Feature: Week number
    if 'week' not in data.columns:
        data['week'] = pd.to_datetime(data['date']).dt.isocalendar().week

    # Feature: Weather category
    if 'weather' not in data.columns:
        conditions = [
            (data['is_rainy'] == 1),
            (data['is_snowy'] == 1)
        ]
        choices = ['Rainy', 'Snowy']
        data['weather'] = np.select(conditions, choices, default='Clear')

    # Cyclical encoding for hour and month
    data['sin_hour'] = np.sin(data['hour'] * (2 * np.pi / 24))
    data['cos_hour'] = np.cos(data['hour'] * (2 * np.pi / 24))
    data['sin_month'] = np.sin(data['month'] * (2 * np.pi / 12))
    data['cos_month'] = np.cos(data['month'] * (2 * np.pi / 12))

    # Select final columns
    targets = ['log_bike_count', 'bike_count']

    selected_features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    'humidity', 'windspeed', 'year', 'hour', 'week', 'distance_to_center', 
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'day_period', 'date'
    ]

    if target == True: 
        selected_features = selected_features + targets
        return data[selected_features]
    
    else:
        return data[selected_features]


final_train_data = preprocess_and_engineer_features(train_data, True)
print(final_train_data.head())
print(final_train_data.info())


# %% [markdown]
# **External data**
# 
# Vacations in Paris can be a very crucial factor in determining the bike count
# We have used an external dataset vacances_paris.csv
# 
# **Source and Liscence:**
# 
# Source: https://www.data.gouv.fr/fr/datasets/le-calendrier-scolaire/
# 
# Liscence: Licence Ouverte / Open Licence version 2.0
# 

# %%
holidays = pd.read_csv("/kaggle/input/vacances-paris/vacances_paris.csv")

# %%
holidays.head(5)

# %%
def add_school_holidays(data, holidays_file):
    """
    Incorporates school holiday information into the input DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing bike usage data.
        holidays_file (str): Path to the CSV file containing holiday data.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'school_holiday' feature.
    """
    # Load holidays dataset
    holidays = pd.read_csv(holidays_file)

    # Convert holiday start and end dates to datetime
    holidays["Start_Date"] = pd.to_datetime(holidays["Date de début"]).dt.tz_localize(None)
    holidays["End_Date"] = pd.to_datetime(holidays["Date de fin"]).dt.tz_localize(None)

    # Ensure the 'date' column in the main dataset is in datetime format
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)

    # Vectorized computation of school holiday flag
    def is_school_holiday(dates, holidays):
        """
        Determines whether given dates fall within school holiday periods.

        Parameters:
            dates (pd.Series): A series of dates to check.
            holidays (pd.DataFrame): A DataFrame with 'Start_Date' and 'End_Date' columns.

        Returns:
            pd.Series: A series with 1 if the date is a school holiday, else 0.
        """
        is_holiday = pd.Series(0, index=dates.index, dtype=int)
        for _, row in holidays.iterrows():
            is_holiday |= ((dates >= row["Start_Date"]) & (dates <= row["End_Date"])).astype(int)
        return is_holiday

    # Add the school holiday feature
    data["school_holiday"] = is_school_holiday(data["date"], holidays)

    return data

final_train_data = add_school_holidays(final_train_data, "/kaggle/input/vacances-paris/vacances_paris.csv")

print(final_train_data.head())
print(final_train_data.info())


# %%

def add_lockdown_status(data):
    """
    Incorporates lockdown status information into the input DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing data with a 'date' column.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'lockdown_status' feature.
    """
    lockdown_df = pd.DataFrame(pd.date_range(start='2020-01-01', end='2021-12-31', freq='D'), columns=['date'])
    lockdown_df['lockdown_status'] = 'no_lockdown'  

    # Defining the Paris lockdown periods
    lockdown_periods = [
        {'start': '2020-03-17', 'end': '2020-05-11', 'type': 'lockdown'},
        {'start': '2020-10-30', 'end': '2020-12-15', 'type': 'lockdown'},
        {'start': '2021-04-03', 'end': '2021-05-02', 'type': 'lockdown'},
        {'start': '2020-05-12', 'end': '2020-06-11', 'type': 'partial'},
        {'start': '2020-10-17', 'end': '2020-10-29', 'type': 'partial'},
        {'start': '2020-12-16', 'end': '2020-12-31', 'type': 'partial'},
        {'start': '2021-05-03', 'end': '2021-05-31', 'type': 'partial'}
    ]

    for period in lockdown_periods:
        lockdown_df.loc[(lockdown_df['date'] >= period['start']) & (lockdown_df['date'] <= period['end']), 'lockdown_status'] = period['type']

    data['date'] = pd.to_datetime(data['date']).dt.normalize()

    lockdown_df['date'] = lockdown_df['date'].dt.normalize()

    merged_data = pd.merge(data, lockdown_df, on='date', how='left')

    return merged_data

final_train_data['date'] = pd.to_datetime(final_train_data['date']).dt.normalize()  
final_train_data = add_lockdown_status(final_train_data)




# %% [markdown]
# From all the data we had, we have selected these required features.

# %% [markdown]
# 
# 1. **season**: Captures seasonal demand patterns.
# 2. **holiday**: Identifies demand differences on holidays.
# 3. **workingday**: Highlights workday versus non-workday variations.
# 4. **weather**: Reflects impact of weather conditions on bike use.
# 5. **temp**: Accounts for the effect of temperature on ridership.
# 6. **atemp**: Considers perceived temperature's influence.
# 7. **school_holiday**: Indicates increased demand due to school closures.
# 8. **humidity**: Captures demand affected by air moisture levels.
# 9. **windspeed**: Measures impact of wind on biking conditions.
# 10. **year**: Accounts for trend-based changes over years.
# 11. **hour**: Reflects time-of-day ridership peaks.
# 12. **week**: Considers weekly ridership patterns.
# 13. **distance_to_center**: Highlights spatial accessibility and urban density effects.
# 14. **sin_hour/cos_hour**: Encodes cyclical nature of hours for demand.
# 15. **sin_month/cos_month**: Encodes cyclical nature of months for seasonality.
# 16. **day_period**: Groups hours into broader day segments for ridership trends.
# 17. **lockdown_period**: Whether there was partial/complete/no lockdown

# %%
final_train_data = final_train_data.drop(columns=["date"])

# %% [markdown]
# Now let us Explore the datset and do an EDA

# %%
final_train_data.columns

# %%
numerical_cols = [
    col for col in final_train_data.columns
    if pd.api.types.is_numeric_dtype(final_train_data[col])
]

categorical_cols = [
    col for col in final_train_data.columns
    if final_train_data[col].dtype == 'object'
]

# %%
def detect_outliers(data, numerical_cols):
    """
    Analyse des outliers dans les colonnes numériques à l'aide de la méthode IQR.

    Parameters:
        data (pd.DataFrame): Le DataFrame à analyser.
        numerical_cols (list): Liste des colonnes numériques.

    Returns:
        dict: Dictionnaire contenant les détails des outliers pour chaque colonne.
    """
    outlier_info = {}
    print("Outlier Analysis:")
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_info[col] = {
            "outliers_count": len(outliers),
            "percentage": len(outliers) / len(data) * 100,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        print(f"{col} Outliers: {len(outliers)} ({outlier_info[col]['percentage']:.2f}%)")
        print(f"  Lower bound: {lower_bound}, Upper bound: {upper_bound}\n")
    return outlier_info

outlier_info = detect_outliers(final_train_data, numerical_cols)


# %% [markdown]
# We can see that there are outliers in columns like holiday, windspeed and temperature. Note: Holiday has only few values other than 0, hence those values are treated as outliers. We should handle that while dealing with outliers

# %%
#target variable distribution
plt.figure(figsize=(5, 3))
sns.histplot(final_train_data['log_bike_count'], kde=True)
plt.title('Distribution of log_bike_count')
plt.show()

# Distribution par bins
final_train_data_plt = final_train_data.copy()
final_train_data_plt['target_bins'] = pd.qcut(final_train_data_plt['log_bike_count'], q=4)
sns.countplot(x='target_bins', data=final_train_data_plt)
plt.title('Distribution of log_bike_count by bins')
plt.show()

# %% [markdown]
# The above 2 plots analysed our target variable. The distribution as we can see is not properly even across all bins

# %%
#correlation matrix
plt.figure(figsize=(8, 8))
correlation_matrix = final_train_data[numerical_cols + ['log_bike_count']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# We look for Correlations. 
# 1. We see that cos_hour has high negetive Correlation with target. This seems like a useful information
# 2. atemp and temp are very correlated. It is expected as both are temperature and one is derived from another.
#    For non windy days they would be equal.
# 3. Several other weather factors are correlated to temperature

# %%
#impact of categorical data on the target
for cat_col in categorical_cols:
    final_train_data.boxplot(column='log_bike_count', by=cat_col)
    plt.title(f'log_bike_count by {cat_col}')
    plt.xticks(rotation=45)
    plt.show()

#Violin Plots for Weather
plt.figure(figsize=(20, 15))
for i, cat_col in enumerate(categorical_cols, 1):
    plt.subplot(2, 3, i)
    sns.violinplot(x=cat_col, y='log_bike_count', data=final_train_data)
    plt.title(f'log_bike_count Distribution by {cat_col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# We are analysing weather and its impact on target.
# We had derived rainy, snowy and clear from the weather data
# Weather has impact on target variable and different conditions have different impacts

# %%
# Identifying time-based columns
time_cols = [
    col for col in ['hour', 'week', 'sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    if col in final_train_data.columns
]

print("Time-based Columns:", time_cols)

# Analysis of time-based features against the target variable
plt.figure(figsize=(20, 10))
for i, col in enumerate(time_cols, 1):
    plt.subplot(2, 3, i)
    sns.lineplot(x=final_train_data[col], y=final_train_data['log_bike_count'])
    plt.title(f'{col} vs log_bike_count')
    plt.xlabel(col)
    plt.ylabel('log_bike_count')
    plt.xticks(rotation=45)
    plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# 
# 1. **hour vs log_bike_count**: Bike usage peaks around the afternoon and decreases sharply at night.
# 2. **week vs log_bike_count**: There are weekly variations in bike usage, with peaks in certain weeks and dips towards the end of the year.
# 3. **sin_hour vs log_bike_count**: The sine-transformed hour cyclically aligns with bike usage, capturing hourly periodicity.
# 4. **cos_hour vs log_bike_count**: The cosine-transformed hour inversely correlates with bike usage periodicity.
# 5. **sin_month vs log_bike_count**: The sine-transformed months capture seasonality, showing periodic variation.
# 6. **cos_month vs log_bike_count**: The cosine-transformed months complement sine transformation, highlighting periodic trends in bike usage.

# %%
# Pairwise relationships between numerical features and the target variable
sns.pairplot(final_train_data[numerical_cols], diag_kind='kde', corner=True)
plt.suptitle("Pairplot for Numerical Features", y=1.02)
plt.show()

# %%
# Calculating skewness and kurtosis for numerical columns
print("\nSkewness and Kurtosis:")
for col in numerical_cols:
    skewness = final_train_data[col].skew()
    kurtosis = final_train_data[col].kurt()
    print(f"{col}: Skewness={skewness:.2f}, Kurtosis={kurtosis:.2f}")

# %% [markdown]
# 1. Features like holiday and workingday are categorical, so their skewness and kurtosis reflect their limited values or imbalanced frequency distribution.
# 2. Continuous features like temp, atemp, and distance_to_center show well-behaved distributions with minimal skewness or kurtosis.
# 3. Outliers are most pronounced in holiday, but we had explained why this is so when we saw box plots.

# %%
# Univariate analysis for selected features
features_to_plot = [
    col for col in ['temp', 'humidity', 'windspeed']
    if col in final_train_data.columns
]

plt.figure(figsize=(12, 10))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(3, 2, i)
    sns.histplot(final_train_data[feature], bins=50, kde=True, color='green')
    plt.axvline(final_train_data[feature].mean(), color='red', linestyle='--', label='Mean')
    plt.axvline(final_train_data[feature].median(), color='blue', linestyle='--', label='Median')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# We can see that humidity is slightly skewed

# %% [markdown]
# Now, these numbers give a sense of more numerical understanding of outliers.
# Note: Holiday has only few values other than 0, hence those values are treated as outliers.
# We should handle that while dealing with outliers

# %%
# Identify and drop columns with more than a threshold of missing values
def drop_high_missing_columns(df, threshold=40):
    """
    Drops columns from the dataframe that have more than the specified percentage of missing values.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        threshold (float): Percentage threshold for missing values.

    Returns:
        pd.DataFrame: Dataframe with high-missing columns removed.
        list: Names of the dropped columns.
    """
    missing_percentage = df.isna().mean() * 100
    cols_to_drop = missing_percentage[missing_percentage > threshold].index
    df = df.drop(columns=cols_to_drop)
    print(f"Columns dropped (>{threshold}% missing): {list(cols_to_drop)}")
    return df

# Handle missing and infinite values
def handle_missing_values(df):
    """
    Fills missing and infinite values in the dataframe:
        - Numeric columns: Linear interpolation for NaNs, replace infinities with NaNs.
        - Categorical columns: Replace NaNs with the mode.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with missing and infinite values handled.
    """
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:  # Colonnes numériques
            df[column].replace([np.inf, -np.inf], np.nan, inplace=True)
            df[column].interpolate(method='linear', inplace=True)
        elif isinstance(df[column].dtype, pd.CategoricalDtype) or df[column].dtype == 'object':  # Colonnes catégorielles
            if df[column].isna().sum() > 0:
                df[column].fillna(df[column].mode()[0], inplace=True)
    return df

# Check for remaining missing values
def check_missing_values(df):
    """
    Prints and returns the count of missing values per column in the dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with counts of missing values per column.
    """
    missing_values = pd.DataFrame(df.isna().sum(), columns=["missing_count"])
    columns_with_missing = missing_values[missing_values["missing_count"] > 0]
    print("Columns with missing values:")
    print(columns_with_missing)
    return columns_with_missing

# Apply the missing value handling
print("Initial DataFrame shape:", final_train_data.shape)
train_data_cleaned = drop_high_missing_columns(final_train_data, threshold=40)

print("\nHandling missing and infinite values...")
train_data_cleaned = handle_missing_values(train_data_cleaned)

# Verify remaining missing values
remaining_missing = check_missing_values(train_data_cleaned)

# Summary
if remaining_missing.empty:
    print("\nNo missing values remain after handling.")
else:
    print(f"\nColumns with remaining missing values:\n{remaining_missing}")
print("Final DataFrame shape:", train_data_cleaned.shape)


# %%
from sklearn.preprocessing import OneHotEncoder
import pickle

def drop_high_cardinality_columns(df, threshold=50):
    """
    Drops columns with unique categories greater than the specified threshold.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        threshold (int): Maximum number of unique categories allowed for categorical columns.

    Returns:
        pd.DataFrame: Dataframe with high-cardinality columns dropped.
        list: Names of dropped columns.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > threshold]

    print(f"Dropping columns with >{threshold} unique categories: {high_cardinality_cols}")
    df = df.drop(columns=high_cardinality_cols)
    return df, high_cardinality_cols

def one_hot_encode_columns(df, encoders_dict):
    """
    One-hot encodes categorical columns with a reasonable number of unique categories.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        encoders_dict (dict): Dictionary to store fitted OneHotEncoders.

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded columns.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"Encoding column '{col}' with {unique_count} unique categories using OneHotEncoder.")

        # Fit OneHotEncoder and transform the column
        ohe = OneHotEncoder(sparse_output=False, drop=None)
        transformed = ohe.fit_transform(df[[col]])

        # Create a dataframe for the encoded categories
        encoded_df = pd.DataFrame(
            transformed,
            columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
            index=df.index
        )

        # Store the encoder for future use
        encoders_dict[col] = ohe

        # Replace the original column with the encoded columns
        df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

    return df

def save_encoders(encoders_dict, file_path='onehot_encoders.pkl'):
    """
    Saves the OneHotEncoders dictionary to a file using pickle.

    Parameters:
        encoders_dict (dict): Dictionary of OneHotEncoders.
        file_path (str): File path for saving the dictionary.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(encoders_dict, f)
    print(f"Encoders saved to '{file_path}'.")

# Apply the functions to clean and encode the dataset
print("Initial dataset shape:", train_data_cleaned.shape)

# Drop high-cardinality columns
train_data_cleaned, dropped_columns = drop_high_cardinality_columns(train_data_cleaned, threshold=50)

# One-hot encode remaining categorical columns
one_hot_encoders = {}
train_data_cleaned = one_hot_encode_columns(train_data_cleaned, one_hot_encoders)

# Save the encoders for future use
save_encoders(one_hot_encoders)

# Final summary
print("Final dataset shape:", train_data_cleaned.shape)
print("Sample of the updated dataset:")
print(train_data_cleaned.head())

# %%
def cap_outliers(df, threshold=4, exclude_columns=None):
    """
    Caps numerical columns to a range defined by mean +/- (threshold * standard deviation).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The number of standard deviations to use for capping.
        exclude_columns (list): List of column names to exclude from capping. Default is None.

    Returns:
        pd.DataFrame: The DataFrame with capped values.
    """
    if exclude_columns is None:
        exclude_columns = []

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        if col in exclude_columns:
            print(f"Skipping column '{col}' (excluded).")
            continue

        # Calculate the capping limits
        mean = df[col].mean()
        std_dev = df[col].std()
        lower_limit = mean - threshold * std_dev
        upper_limit = mean + threshold * std_dev

        # Apply capping
        print(f"Capping outliers in column '{col}' to [{lower_limit:.2f}, {upper_limit:.2f}].")
        df[col] = np.clip(df[col], lower_limit, upper_limit)

    return df


train_data_cleaned = cap_outliers(train_data_cleaned, threshold=4, exclude_columns=['holiday'])
print("Outliers capped. Preview of updated dataset:")
print(train_data_cleaned.head())


# %%
from sklearn.preprocessing import RobustScaler
import pandas as pd

def scale_features(data, features):
    """Scales specified features using RobustScaler and updates the original columns.

    Args:
    data (pd.DataFrame): The data containing the features to scale.
    features (list): List of feature column names to scale.

    Returns:
    pd.DataFrame: The data with scaled features.
    """
    scaler = RobustScaler()
    data[features] = scaler.fit_transform(data[features])

    return data

train_data_cleaned = scale_features(train_data_cleaned, ['humidity', 'windspeed'])

print("Updated dataset with scaled features:")
print(train_data_cleaned.head())


# %%
train_data_cleaned.columns

# %%
cols_to_drop = ['log_bike_count','bike_count']
X = train_data_cleaned.drop(columns=cols_to_drop)
y = train_data_cleaned['log_bike_count']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 17)

# %%
# Trying out diff models and seeing which one works the best for our dataset

models = [LinearRegression(), Lasso(), Ridge(), DecisionTreeRegressor(), RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), KNeighborsRegressor(), XGBRegressor(),LGBMRegressor()]
model_names = ['LinearRegression', 'Lasso', 'Ridge', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'KNeighborsRegressor', 'XGBRegressor','LightGBM']
r2_train = []
r2_val = []
i = 0
for model in models:
    i=i+1
    mod = model
    mod.fit(X_train, y_train)
    y_pred_train = mod.predict(X_train)
    y_pred_train = y_pred_train.clip(0)
    y_pred_val = mod.predict(X_val)
    y_pred_val = y_pred_val.clip(0)
    r2_train.append(r2_score(y_train, y_pred_train))
    r2_val.append(r2_score(y_val, y_pred_val))
data = {'Modelling Algorithm' : model_names, 'Train R2' : r2_train, 'Validation R2' : r2_val}
data = pd.DataFrame(data)
data['Difference'] = ((np.abs(data['Train R2'] - data['Validation R2'])) * 100)/(data['Train R2'])
data.sort_values(by = 'Validation R2', ascending = False)

# %% [markdown]
# 1. **XGBoost**: It has a high training R² (0.8959) and validation R² (0.8720), with a small difference of 0.4427, indicating a good balance between model complexity and performance without overfitting.
# 2. **LightGBM**: It also demonstrates strong performance, with a training R² (0.8775) and validation R² (0.8756), and a very small difference of 0.2141, which highlights its reliability and efficiency for generalization.
# 
# Both models are candidate models for our pipeline because of their ability to provide robust predictions while maintaining low overfitting compared to models like Random Forest or Decision Tree, which show larger differences.

# %%

# Use RandomizedSearch Cross Validation to find the Optimal Hyperparameters for the LightGBM Model

param_dist = {
    'num_leaves': np.arange(20, 300, step=5),
    'learning_rate': np.logspace(-3, 0, num=100),
    'n_estimators': np.arange(100, 1000, step=50),
    'max_bin': np.arange(50, 255, step=5),
    'min_child_samples': np.arange(5, 50, step=5),
    'reg_alpha': np.logspace(-4, 0, num=50),
    'reg_lambda': np.logspace(-4, 0, num=50),
    'colsample_bytree': np.linspace(0.4, 1.0, num=50)
}

lgbm = LGBMRegressor(n_jobs=-1, verbose=-1)
tscv = TimeSeriesSplit(n_splits=3)
random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=tscv,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

# Best Parameters Found
print("Best parameters found: ", random_search.best_params_)
print("Best score found: ", -random_search.best_score_)


# %%
# Evaluate the model on the test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print("Best Parameters:", random_search.best_params_)
print("Test RMSE:", rmse)

# %%
# Characteristics of the final model
best_model

# %%
# Feature Importances
if hasattr(best_model, "feature_importances_"):
    feature_importances = best_model.feature_importances_
    features = X_val.columns if hasattr(X_val, "columns") else np.arange(len(feature_importances))
    feature_importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importances:")
    print(feature_importance_df)
else:
    print("\nThe best model does not support feature importances.")


# %% [markdown]
# Note: We have tried using XGBoost model, but the final rmse on validation was higher when compared to LightGBM. We had also tried a Stacking regressor approach with GradientBoosting Regressor as Meta model combining our XGBoost and LightGBM models. Even though rmse improved slightly, considering the time complexity, we still chose to keep the LightGBM model as our final choice

# %% [markdown]
# Preprocessing our test dataset

# %%
# Load test dataset
print("Initial test data columns:")
print(test_data.columns)

# Step 1: Encode dates and merge external data
test_data = example_estimator._encode_dates(test_data)
test_data = example_estimator._merge_external_data(test_data)

# Step 2: Preprocess and engineer features (similar to train_data)
test_data = preprocess_and_engineer_features(test_data, False)

# Step 3: Add school holiday information and Covid related information
test_data = add_school_holidays(test_data, "/kaggle/input/vacances-paris/vacances_paris.csv")
test_data['date'] = pd.to_datetime(test_data['date']).dt.normalize()  
test_data = add_lockdown_status(test_data)

# Step 4: Handle missing values
print("\nHandling missing and infinite values in test dataset...")
test_data = drop_high_missing_columns(test_data, threshold=40)  
test_data = handle_missing_values(test_data)  

# Verify remaining missing values
remaining_missing_test = check_missing_values(test_data)

if remaining_missing_test.empty:
    print("\nNo missing values remain in the test dataset after handling.")
else:
    print(f"\nColumns with remaining missing values in test dataset:\n{remaining_missing_test}")

# Step 5: Drop high cardinality columns
test_data, dropped_columns_test = drop_high_cardinality_columns(test_data, threshold=50)

# Step 6: Apply saved one-hot encoders to test data
def apply_one_hot_encoding(test_data_cleaned, encoder_path):
    """
    Applies one-hot encoding to the specified columns in a test dataset using pre-fitted encoders.
    """
    with open(encoder_path, 'rb') as f:
        one_hot_encoders = pickle.load(f)

    for col, ohe in one_hot_encoders.items():
        if col in test_data_cleaned.columns:
            transformed = ohe.transform(test_data_cleaned[[col]])
            ohe_df = pd.DataFrame(
                transformed,
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                index=test_data_cleaned.index
            )

            test_data_cleaned = pd.concat([test_data_cleaned.drop(columns=[col]), ohe_df], axis=1)


    return test_data_cleaned

test_data = apply_one_hot_encoding(test_data, 'onehot_encoders.pkl')

# Step 7: Cap outliers
test_data = cap_outliers(test_data, threshold=4, exclude_columns=['holiday'])
print("Outliers capped. Preview of updated test dataset:")
print(test_data.head())

# Step 8: Scale numerical features
test_data = scale_features(test_data, ['humidity', 'windspeed'])

print("Updated test dataset with scaled features:")
print(test_data.head())

# Final dataset summary
print("Final test dataset shape:", test_data.shape)
print("Sample of the cleaned and preprocessed test dataset:")
print(test_data.head())


# %%
def prepare_submission(predictions):
        """
        Save predictions
        """

        submission_df = pd.DataFrame({'id': range(len(predictions)), 'log_bike_count': predictions})
        submission_df.to_csv('submission.csv', index=False)
        print("Submission created.")

# %%
assert all(col in test_data.columns for col in X_train.columns), "Test data must have the same columns as the training data."


predictions = best_model.predict(test_data[X_train.columns])

print("Predictions on test data:")
prepare_submission(predictions)
