import pandas as pd
import numpy as np
from datetime import timedelta
from pylab import rcParams
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def pivot_data(data, target, binary_target_value=None):
    X_original = data.drop([target], axis=1)
    y_original = data[target]
    y = y_original.apply(lambda x: 1 if x==binary_target_value else 0)

    X = pd.DataFrame()
    for col in X_original.columns:
        if type(X_original[col][0]) == str:
            col_pivoted = pd.get_dummies(X_original[col], prefix=col)
            X = pd.concat([X, col_pivoted], axis=1)
        else:
            X = pd.concat([X, X_original[col]], axis=1)

    return X, y


def resample_data(X_train, y_train, method, random_state=None):
    if method == "upsample":
        X_train_balanced, y_train_balanced = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    elif method == "downsample":
        X_train_balanced, y_train_balanced = RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)
    
    return X_train_balanced, y_train_balanced


def normalize_scale(X_train, X_test, method="standard", exclude_column=None,
                    min=0, max=1):
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=(min, max))
    X_train_normalized = pd.DataFrame(scaler.fit_transform(X_train),
                                      index=X_train.index, columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler.transform(X_test),
                                     index=X_test.index, columns=X_test.columns)
    
    if exclude_column != None:
        X_train_normalized[exclude_column] = X_train[exclude_column]
        X_test_normalized[exclude_column] = X_test[exclude_column]

    return X_train_normalized, X_test_normalized, scaler


def numeric_to_interval(data, column, n_intervals):
    col_vals = data[column]
    min_val = np.min(col_vals)
    max_val = np.max(col_vals)
    interval = (max_val - min_val) / n_intervals
    data[column] = pd.cut(col_vals, bins=np.arange(min_val, max_val, interval), right=True)
    
    return data


def concat_counts_df(df1, df1_name, df2, df2_name, column):
    counts_df1 = df1[column].value_counts(sort=False)
    counts_df1.name = f"{df1_name}_{column}"
    counts_df2 = df2[column].value_counts(sort=False)
    counts_df2.name = f"{df2_name}_{column}"
    return pd.concat([
        pd.DataFrame(counts_df1/ sum(counts_df1)).T,
        pd.DataFrame(counts_df2/ sum(counts_df2)).T
        ]).round(2)


def get_numeric_columns(data, cols_to_exclude=None):
    numeric_columns = []
    non_numeric_columns = []
    for col in data.drop(cols_to_exclude, axis=1).columns:
        if type(data[col][0]) == str:
            non_numeric_columns.append(col)
        else:
            numeric_columns.append(col)

    return numeric_columns, non_numeric_columns


def add_zero_score_col(data, fit_columns):
    has_zero_scores = []
    for i, row in data.iterrows():
        has_zero_score = 0
        for fit in data.iloc[i][fit_columns]:
            if fit == 0:
                has_zero_score = 1
        
        has_zero_scores.append(has_zero_score)

    data['has_zero_scores'] = has_zero_scores

    return data


def update_ranks(y_train_updated, y_test_updated, ideal_candidates):
    y_train_updated += 1
    y_test_updated += 1

    ideal_rank = 1
    for id in ideal_candidates:
        if id in y_train_updated.index:
            y_train_updated[id] = ideal_rank
            print(f"Rank of candidate {id} in y_train updated to {ideal_rank}.")
        elif id in y_test_updated.index:
            y_test_updated[id] = ideal_rank
            print(f"Rank of candidate {id} in y_test updated to {ideal_rank}.")
        else:
            print(f"Candidate {id} not found!")

    return y_train_updated, y_test_updated


def convert_vol_to_float(string):
    if string == '-':
        return np.nan
    elif "M" in string:
        return float(string.split("M")[0]) * 1000000
    elif "K" in string:
        return float(string.split("K")[0]) * 1000
    

def add_moving_average(df):
    weekly_window = 7
    monthly_window = int(365/12)
    quarterly_window = int(365/4)

    df['Weekly SMA'] = df['Close'].rolling(window=weekly_window).mean()
    df['Monthly SMA'] = df['Close'].rolling(window=monthly_window).mean()
    df['Quarterly SMA'] = df['Close'].rolling(window=quarterly_window).mean()

    df['Weekly EMA'] = df['Close'].ewm(span=weekly_window).mean()
    df['Monthly EMA'] = df['Close'].ewm(span=monthly_window).mean()
    df['Quarterly EMA'] = df['Close'].ewm(span=quarterly_window).mean()

    return df


def seasonal_decomposition(df, frequency, column, add_to_df=True, plot=True):    
    result = seasonal_decompose(df.asfreq(frequency).ffill()[[column]])

    if plot:
        rcParams['figure.figsize'] = 12, 8

        print('Plotting seasonal componets:')
        result.plot()

    if add_to_df:
        df['Trend'] = result.trend
        df['Seasonal'] = result.seasonal
        df['Residual'] = result.resid

        return df
    

def add_datetime_features(df, year, month, day, weekday):
    if year:
        df["Year"] = df.index.year
    if month:
        df["Month"] = df.index.month
    if day:
        df["Day"] = df.index.day
    if weekday:
        df["Weekday"] = df.index.weekday

    return df


def add_data_from_past(df):
    df = df.asfreq('D').ffill()

    df_prev_day = df.loc[ (df.index - timedelta(days=1))[1:] ].rename(
        columns = {column: column + " (-1 day)" for column in df.columns})
    df_prev_day.index += timedelta(days=1)

    df_prev_week = df.loc[ (df.index - timedelta(days=7))[7:] ].rename(
        columns = {column: column + " (-1 week)" for column in df.columns})
    df_prev_week.index += timedelta(days=7)

    df_prev_month = df.loc[ (df.index - timedelta(days=30))[30:] ].rename(
        columns = {column: column + " (-1 month)" for column in df.columns})
    df_prev_month.index += timedelta(days=30)

    concat_df = pd.concat([df, df_prev_day, df_prev_week, df_prev_month], axis=1)
    
    return concat_df


def add_features_from_previous_dates(df, previous_date_range=[2, 3, 4, 5, 6, 7]):
    # df['Trend'] = df['Change %'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df = df.drop(['Vol.', 'Change %'], axis=1)
    # df = df.drop(['Change %'], axis=1)

    price_previous_day = df[:-1]
    price_previous_day.index = df.index[1:]
    price_previous_day = price_previous_day.rename(columns={col: f'{col} (Day-1)' for col in df.columns})

    price_sma_dfs = []
    # price_ema_dfs = []
    # max_high_dfs = []
    # min_low_dfs = []
    # max_min_diff_dfs = []
    # diff_dfs = []
    # trend_dfs = []
    # vol_sma_dfs = []
    # vol_ema_dfs = []
    # max_vol_dfs = []
    # min_vol_dfs = []
    for num_day in previous_date_range:
        index = df.index[num_day : ]

        price_sma_df = df['Price'].rolling(window=num_day).mean()[ (num_day-1) : -1]
        price_sma_df.index = index
        price_sma_df.name = f'{num_day} Day SMA Price (Day-1)'
        price_sma_dfs.append(price_sma_df)

        # price_ema_df = df['Price'].ewm(span=num_day).mean()[ (num_day-1) : -1]
        # price_ema_df.index = index
        # price_ema_df.name = f'{num_day} Day EMA Price (Day-1)'
        # price_ema_dfs.append(price_ema_df)

        # max_high_df = df['High'].rolling(window=num_day).max()[ (num_day-1) : -1]
        # max_high_df.index = index
        # max_high_df.name = f'{num_day} Day Max High (Day-1)'
        # max_high_dfs.append(max_high_df)

        # min_low_df = df['Low'].rolling(window=num_day).min()[ (num_day-1) : -1]
        # min_low_df.index = index
        # min_low_df.name = f'{num_day} Day Min Low (Day-1)'
        # min_low_dfs.append(min_low_df)

        # max_min_diff_df = max_high_df - min_low_df
        # max_min_diff_df.name = f'{num_day} Day MaxHigh-MinLow Difference (Day-1)'
        # max_min_diff_dfs.append(max_min_diff_df)

        # diff_df = df['Price'].diff(periods=num_day)[ (num_day-1) : -1]
        # diff_df.index = index
        # diff_df.name = f'{num_day} Day Price Difference (Day-1)'
        # diff_dfs.append(diff_df)

        # trend_df = df['Trend'].rolling(window=num_day).sum()[ (num_day-1) : -1]
        # trend_df.index = index
        # trend_df.name = f'{num_day} Day Trend (Day-1)'
        # trend_dfs.append(trend_df)

        # Add Vol. features
        # vol_sma_df = df['Vol.'].rolling(window=num_day).mean()[ (num_day-1) : -1]
        # vol_sma_df.index = index
        # vol_sma_df.name = f'{num_day} Day SMA Vol. (Day-1)'
        # vol_sma_dfs.append(vol_sma_df)

        # vol_ema_df = df['Vol.'].ewm(span=num_day).mean()[ (num_day-1) : -1]
        # vol_ema_df.index = index
        # vol_ema_df.name = f'{num_day} Day EMA Vol. (Day-1)'
        # vol_ema_dfs.append(vol_ema_df)

        # max_vol_df = df['Vol.'].rolling(window=num_day).max()[ (num_day-1) : -1]
        # max_vol_df.index = index
        # max_vol_df.name = f'{num_day} Day Max Vol. (Day-1)'
        # max_vol_dfs.append(max_vol_df)

        # min_vol_df = df['Vol.'].rolling(window=num_day).min()[ (num_day-1) : -1]
        # min_vol_df.index = index
        # min_vol_df.name = f'{num_day} Day Min Vol. (Day-1)'
        # min_vol_dfs.append(min_vol_df)

    df = df.drop(['Open', 'High', 'Low'], axis=1)

    concat_df = pd.concat(
        [df, price_previous_day]
        + price_sma_dfs
        # + price_ema_dfs
        # + max_high_dfs
        # + min_low_dfs
        # + max_min_diff_dfs
        # + diff_dfs
        # + trend_dfs
        # + vol_sma_dfs
        # + vol_ema_dfs
        # + max_vol_dfs
        # + min_vol_dfs
        ,
        axis=1).dropna()

    return concat_df


def create_bollinger_band(y, y_test, bollinger_band_window=20, bollinger_band_std=2):
    """
    Short term: 10 day moving average, bands at 1.5 standard deviations.
    Medium term: 20 day moving average, bands at 2 standard deviations.
    Long term: 50 day moving average, bands at 2.5 standard deviations.
    """
    
    rolling_mean = y.rolling(window=bollinger_band_window).mean().loc[y_test.index]
    rolling_std = y.rolling(window=bollinger_band_window).std().loc[y_test.index]

    upper_band = (rolling_mean + (rolling_std * bollinger_band_std)).rename('Upper Band')
    lower_band = (rolling_mean - (rolling_std * bollinger_band_std)).rename('Lower Band')

    return rolling_mean, upper_band, lower_band