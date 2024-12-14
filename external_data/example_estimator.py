from pathlib import Path
import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge



def add_holiday_column(X, year=2024):
    # Get the list of French holidays for the specified year
    french_holidays = holidays.FR(years=year)
    
    # Create a DataFrame with the holiday dates (using only the month and day)
    df_holidays = pd.DataFrame(list(french_holidays.keys()), columns=['date'])
    df_holidays['date'] = pd.to_datetime(df_holidays['date'])
    
    # Create the 'holiday' column in the X DataFrame
    X['holiday'] = (X['date'].dt.month.astype(str) + '-' + X['date'].dt.day.astype(str))
    X['holiday'] = X['holiday'].isin(df_holidays['date'].dt.month.astype(str) + '-' + df_holidays['date'].dt.day.astype(str)).astype(int)



def _encode_dates(X):
    X = X.copy()  # Work on a copy of the dataset

    # Extract basic date components
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    add_holiday_column(X, year=2024)
    X.loc[:, "week_of_month"] = (X["date"].dt.day - 1) // 7 + 1
    X.loc[:, "workday"] = (~X["date"].dt.weekday.isin([5, 6])).astype(int)

    # Suggestion: Add season encoding
    X.loc[:, "season"] = (X["date"].dt.month - 1) // 3 + 1 # 1: Winter, 2: Spring, 3: Summer, 4: Autumn

    return X


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    df_ext["date"] = pd.to_datetime(df_ext["date"])
    X = X.copy()
    X["date"] = X["date"].astype("datetime64[us]")
    df_ext["date"] = df_ext["date"].astype("datetime64[us]")

    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext.sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X



def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    regressor = Ridge()

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
