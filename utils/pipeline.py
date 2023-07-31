# Classifiers
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM, LinearSVC, NuSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,\
                            GradientBoostingClassifier, BaggingClassifier

# Transformers
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer

# Decomposers
from sklearn.decomposition import PCA, KernelPCA, FastICA, SparsePCA, IncrementalPCA, TruncatedSVD, MiniBatchSparsePCA
from sklearn.cluster import FeatureAgglomeration

# Time series mmodel
from statsmodels.tsa.statespace.sarimax import SARIMAX

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

from utils.transform import convert_vol_to_float, add_features_from_previous_dates, normalize_scale\
                            , create_bollinger_band
from utils.plot import plot_histograms, plot_bollinger_band
from utils.predict import get_trading_decision_and_results, train_models_and_make_predictions


def process_df(df, sheet_name,
               bollinger_band_windows, bollinger_band_stds,
               plot_histogram=False, plot_bollinger=None,
               normalize_X=False, previous_date_range=[2, 3, 4, 5, 6, 7],
               initial_balance=0, initial_no_stock=0, max_no_stock_to_trade=1,
               print_result=True, random_state=5):
    
    # Transform data types
    df = df.set_index('Date').sort_index()
    df['Vol.'] = df['Vol.'].apply(convert_vol_to_float)

    if plot_histogram:
        plot_histograms(data=df,
                        target='Price', target_figsize=(2,2),
                        dependent_layout=(2,9), dependent_figsize=(12, 4),
                        include_boxplots=True)

    # Feature selection and engineering
    concat_df = add_features_from_previous_dates(df, previous_date_range=previous_date_range)

    # Split into X and y
    X = concat_df.drop(['Price'], axis=1)
    y = concat_df['Price']

    # Split into train and test
    X_train = X[X.index.year == 2020]
    X_test = X[X.index.year == 2021]

    y_train = y[y.index.year == 2020]
    y_test = y[y.index.year == 2021]

    if normalize_X:
        X_train, X_test, scaler = normalize_scale(X_train, X_test, method="standard", exclude_column=None)

    preds_df = train_models_and_make_predictions(X_train, y_train, X_test, y_test, random_state, models=['SARIMAX'])
    sarimax_pred = preds_df['SARIMAX']
    sarimax_pred.name = 'Predicted'
    
    sarimax_rmse = sqrt(mean_squared_error(y_test, sarimax_pred))
    sarimax_mape = mean_absolute_percentage_error(y_test, sarimax_pred)
    sarimax_r2 = r2_score(y_test, sarimax_pred)
    # print(f'''SARIMAX model:
    # Mean Squared Error: {round(sarimax_mse, 4)}
    # Mean Absolute Percentage Error: {round(sarimax_mape, 4)}
    # R2 Score: {round(sarimax_r2, 4)}
    # ''')
    eval_df = pd.DataFrame({
        'Root Mean Squared Error': round(sarimax_rmse, 4),
        'Mean Absolute Percentage Error': round(sarimax_mape, 4),
        'R2 Score': round(sarimax_r2, 4)
    }, index=[sheet_name])
    # print(eval_df)

    # Create Bollinger Bands and compare against the SARIMAX predictions to make trading decisions.
    if plot_bollinger != None:
        if plot_bollinger == 'Daily':
            window = bollinger_band_windows[0]
            std = bollinger_band_stds[0]
        elif plot_bollinger == 'Weekly':
            window = bollinger_band_windows[1]
            std = bollinger_band_stds[1]
        elif plot_bollinger == 'Weekly':
            window = bollinger_band_windows[2]
            std = bollinger_band_stds[2]
        else:
            print(f'Unexpected input {plot_bollinger}! Please input one of the following values:')
            print('Daily, Weekly, or Monthly')

        rolling_mean, upper_band, lower_band = create_bollinger_band(y, y_test, window, std)

        plot_bollinger_band(rolling_mean, window, std,
                            upper_band, lower_band, y_test,
                            additional_df=pd.concat([y_test, sarimax_pred] ,axis=1), xlabel='Date', ylabel='Price')

    # Get trading dates with different trading intervals    
    daily_trading_dates = y_test.index # Trade every 1 trading day
    weekly_trading_dates = y_test.index[np.arange(5, len(y_test), 5)] # Trade every 5 trading days
    monthly_trading_dates = y_test.index[np.arange(20, len(y_test), 20)] # Trade every 20 trading days

    # Evaluate results of the training decisions based solely on Bollinger Band and on the SARIMAX predictions
    results_based_on_bollinger = get_trading_decision_and_results(
        y, y_test, sarimax_pred,
        bollinger_band_windows, bollinger_band_stds,
        df, initial_balance, initial_no_stock, max_no_stock_to_trade,
        daily_trading_dates, weekly_trading_dates, monthly_trading_dates,
        use_pred=False, print_result=print_result)
    # print(f'Results based on Bollinger Band: {results_based_on_bollinger}')
    # print()
    results_based_on_predictions = get_trading_decision_and_results(
        y, y_test, sarimax_pred,
        bollinger_band_windows, bollinger_band_stds,
        df, initial_balance, initial_no_stock, max_no_stock_to_trade,
        daily_trading_dates, weekly_trading_dates, monthly_trading_dates,
        use_pred=True, print_result=print_result)
    # print(f'Results based on SARIMAX predictions: {results_based_on_predictions}')

    results_based_on_bollinger.index = [
        'Bollinger - Daily',
        'Bollinger - Weekly',
        'Bollinger - Monthly'
        ]
    results_based_on_predictions.index = [
        'SARIMAX - Daily',
        'SARIMAX - Weekly',
        'SARIMAX - Monthly'
        ]
    capital_return_df = pd.concat([results_based_on_bollinger, results_based_on_predictions]).T
    capital_return_df.index = [sheet_name]
    capital_return_df['Daily Diff'] = capital_return_df['SARIMAX - Daily'] - capital_return_df['Bollinger - Daily']
    capital_return_df['Weekly Diff'] = capital_return_df['SARIMAX - Weekly'] - capital_return_df['Bollinger - Weekly']
    capital_return_df['Monthly Diff'] = capital_return_df['SARIMAX - Monthly'] - capital_return_df['Bollinger - Monthly']
    
    # print(capital_return_df)
    return eval_df, capital_return_df