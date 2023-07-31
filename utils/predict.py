import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.transform import resample_data, normalize_scale, create_bollinger_band
from utils.evaluate import get_capital_returns


def transform_fit_predict(X_train, y_train, X_test, y_test,
                          resample, resampling_method,
                          normalize, normalize_method,
                          model,
                          cv_n_splits, cv_n_repeats,
                          method,
                          random_state=None,
                          save_model=False, path=None):
    if resample:
        X_train, y_train = resample_data(X_train, y_train, method=resampling_method, random_state=random_state)

    if normalize:
        X_train, X_test = normalize_scale(X_train, X_test, method=normalize_method)

    model.fit(X_train, y_train)
    if save_model:
        if path == None:
            path = Path(os.getcwd())
        with open(path / "trained_model.sav", 'wb') as f:
            pickle.dump(model, f)
        print(f"Trained model saved: {path}")
            
    scores = cross_val_score(model, X_test, y_test, scoring='roc_auc',
                            cv=RepeatedStratifiedKFold(n_splits=cv_n_splits, n_repeats=cv_n_repeats))
    print(f"{method}: {np.mean(scores)}")


def get_rank_predictions(X, y, ranker, target_column,
                         target="data points",
                         top_n=5,
                         round_precision=4):
    results = pd.DataFrame(y)

    pred = ranker.predict(X)
    results['pred_rank'] = pd.Series(pred).rank(method='dense', ascending=True).values
    results['abs_diff'] = abs( results['rank'] - results['pred_rank'] )

    selected_mean_rank = results[results[target_column] <= top_n]['pred_rank'].mean()
    print(f"Mean rank of top {top_n} {target} based on predictions: {round(selected_mean_rank, round_precision)}")
    print(f"Mean rank of all {target} based on predictions: {round(results['pred_rank'].mean(), round_precision)}")
    print(f"Std rank of all {target} based on predictions: {round(results['pred_rank'].std(), round_precision)}")
    print(f"Mean absolute difference between each pair of rank and predicted rank:",
          f"{round(results['abs_diff'].mean(), round_precision)}")
    print(results.sort_values(['pred_rank', target_column]).head(), "\n")
    
    return results.sort_values(['pred_rank', target_column])


def get_stats(X_train, X_test, y_train, y_test, model,
              target_column="rank", target="candidates",
              updated=False):
    stats_columns = ["y_train", "y_test"]
    print_statements = [
        "Ground truth stats:",
        "Train stats:",
        "Test stats:"
    ]
    if updated:
        stats_columns = [stats_column + "_updated" for stats_column in stats_columns]
        print_statements = ["(Updated) " + print_statement for print_statement in print_statements]
    
    stats_df = pd.DataFrame(
        index=["Mean (Top 5 rankers)", "Mean", "Std"],
        columns=stats_columns,
        data=[
            [round(y_train[y_train<=5].mean(), 4),
            round(y_test[y_test<=5].mean(), 4)],
            [round(y_train.mean(), 4),
            round(y_test.mean(), 4)],
            [round(y_train.std(), 4),
            round(y_test.std(), 4)]
        ]
    )

    print(print_statements[0])
    print(stats_df,"\n")

    print(print_statements[1])
    train_result = get_rank_predictions(X_train, y_train, model, target_column=target_column, target=target)

    print(print_statements[2])
    test_result = get_rank_predictions(X_test, y_test, model, target_column=target_column, target=target)

    return stats_df, train_result, test_result


def get_trading_decision_and_results(y, y_test, y_pred, bollinger_band_windows, bollinger_band_stds,
                                     df, initial_balance, initial_no_stock, max_no_stock_to_trade,
                                     daily_trading_dates, weekly_trading_dates, monthly_trading_dates,
                                     use_pred=False, pred_error_rate=0.01, pred_col='Predicted',
                                     print_result=True):

    """
    Short term: 10 day moving average, bands at 1.5 standard deviations.
    Medium term: 20 day moving average, bands at 2 standard deviations.
    Long term: 50 day moving average, bands at 2.5 standard deviations.
    """

    temp_dfs = {}
    for interval, window, std in zip(['Daily', 'Weekly', 'Monthly'],
                                     bollinger_band_windows, bollinger_band_stds):
        rolling_mean, upper_band, lower_band = create_bollinger_band(y, y_test, window, std)
        temp_df = pd.concat([
            df[['Price', 'Open', 'High', 'Low']],
            y_pred,
            rolling_mean.rename(f'{window}-day SMA'),
            upper_band,
            lower_band
            ],
            axis=1).dropna()
        temp_dfs[interval] = temp_df

    results_dfs = {}
    for interval, trading_dates in zip(['Daily', 'Weekly', 'Monthly'],
                                        [daily_trading_dates, weekly_trading_dates, monthly_trading_dates]):
        df = temp_dfs[interval].loc[trading_dates]
        df['Balance'] = initial_balance
        df['No. of Stock'] = initial_no_stock
        df['Trading Decision'] = 'Hold' # Default trading decision

        index_list = df.index
        last_index = index_list[-1]
        for i in range(len(df)):
            current_index = index_list[i]
            previous_index = index_list[max(0, i-1)]

            # price = df.loc[current_index, 'Price'] 
            trading_price = df.loc[current_index, 'Open']

            predicted_price = df.loc[previous_index, pred_col]
            previous_high = df.loc[previous_index, 'High'] 
            previous_low = df.loc[previous_index, 'Low']

            if use_pred:
                previous_high = predicted_price * (1 + pred_error_rate)
                previous_low = predicted_price * (1 - pred_error_rate)

            previous_upper = df.loc[previous_index, 'Upper Band']
            previous_lower = df.loc[previous_index, 'Lower Band']

            previous_balance = df.loc[previous_index, 'Balance']
            previous_no_stock = df.loc[previous_index, 'No. of Stock']

            if (((previous_high > previous_upper) or (current_index == last_index))\
                and (previous_no_stock > 0)):
                    
                no_stock_to_sell = min( max_no_stock_to_trade, previous_no_stock )
                # Sell all stocks on the last trading date
                if current_index == last_index:
                    no_stock_to_sell = previous_no_stock

                # Only sell when we won't lose money (or when it's the last trading date)
                if (((no_stock_to_sell * trading_price) >= 0) or (current_index == last_index)):
                    if print_result:
                        print(f'Sell {no_stock_to_sell} stock(s) at {trading_price}.')
                    df.loc[current_index, 'Trading Decision'] = 'Sell'
                    df.loc[current_index, 'Balance'] = previous_balance + (no_stock_to_sell * trading_price)
                    df.loc[current_index, 'No. of Stock'] = previous_no_stock - no_stock_to_sell
                else:
                    if print_result:
                        print(f'Hold. Selling stocks at {trading_price} will create a loss.')
                    df.loc[current_index, 'Trading Decision'] = 'Hold'
                    df.loc[current_index, 'Balance'] = previous_balance
                    df.loc[current_index, 'No. of Stock'] = previous_no_stock

            elif previous_low < previous_lower:

                max_no_stock_to_buy = previous_balance // trading_price
                no_stock_to_buy = min( max_no_stock_to_trade, max_no_stock_to_buy )

                if no_stock_to_buy > 0:
                    # Only buy when the balance won't go negative
                    if previous_balance >= (trading_price * no_stock_to_buy):
                        if print_result:
                            print(f'Buy {no_stock_to_buy} stock(s) at {trading_price}.')
                        df.loc[current_index, 'Trading Decision'] = 'Buy'
                        df.loc[current_index, 'Balance'] = previous_balance - (trading_price * no_stock_to_buy)
                        df.loc[current_index, 'No. of Stock'] = previous_no_stock + no_stock_to_buy
                    else:
                        if print_result:
                            print(f'Hold. Buying stocks at {trading_price} will create a loss.')
                        df.loc[current_index, 'Trading Decision'] = 'Hold'
                        df.loc[current_index, 'Balance'] = previous_balance
                        df.loc[current_index, 'No. of Stock'] = previous_no_stock

                else:
                    if print_result:
                        print(f'Hold. Not enough balance to buy stocks at {trading_price}.')
                    df.loc[current_index, 'Trading Decision'] = 'Hold'
                    df.loc[current_index, 'Balance'] = previous_balance
                    df.loc[current_index, 'No. of Stock'] = previous_no_stock

            else:
                if print_result:
                    print(f'Hold. The price range is within the Bollinger Band.')
                df.loc[current_index, 'Trading Decision'] = 'Hold'
                df.loc[current_index, 'Balance'] = previous_balance
                df.loc[current_index, 'No. of Stock'] = previous_no_stock

        last_row = df.iloc[-1]
        if print_result:
            print(f'\nTrading interval: {interval}')
            print(f'{last_row["No. of Stock"]} remaining stock at last closing price of {last_row["Price"]}')
            print('-----------------------------------------------------------------------------------')
        results_dfs[interval] = df
    
    capital_return_df = get_capital_returns(results_dfs, initial_balance)

    return capital_return_df


def train_models_and_make_predictions(X_train, y_train, X_test, y_test, random_state,
                                      models=['SARIMAX',
                                              'RandomForestRegressor',
                                              'XGBRegressor']):
    result_dfs = []
    for i in tqdm(range(0, len(X_test))):
        new_X_train = pd.concat([X_train, X_test[:i]])
        new_y_train = pd.concat([y_train, y_test[:i]])
        new_X_test = X_test[i:i+1]
        new_y_test = y_test[i:i+1]

        result_df = pd.DataFrame(new_y_test)

        # Train models and get predictions only for 1 day.
        if 'SARIMAX' in models:
            sarimax_model = SARIMAX(
                endog = new_y_train,
                exog = new_X_train,
                order = (0, 1, 0),
                seasonal_order = (0, 0, 0, 0)
                )
            results = sarimax_model.fit(disp=False)
            pred = results.get_prediction(start=new_X_train.shape[0],
                                        end=new_X_train.shape[0] + new_X_test.shape[0] - 1,
                                        exog=new_X_test)
            pred_mean = pred.predicted_mean
            pred_mean.index = new_X_test.index
            result_df['SARIMAX'] = pred_mean.values

        if 'RandomForestRegressor' in models:
            rf_reg = RandomForestRegressor(max_depth=2, random_state=random_state)
            rf_reg.fit(new_X_train, new_y_train)
            rf_pred = rf_reg.predict(new_X_test)
            rf_pred = pd.Series(rf_pred, index=new_X_test.index)
            result_df['RandomForestRegressor'] = rf_pred.values

        if 'XGBRegressor' in models:
            xgb_reg = XGBRegressor(random_state=random_state)
            xgb_reg.fit(new_X_train, new_y_train)
            xgb_pred = xgb_reg.predict(new_X_test)
            xgb_pred = pd.Series(xgb_pred, index=new_X_test.index)
            result_df['XGBRegressor'] = xgb_pred.values

        result_dfs.append(result_df)

    preds_df = pd.concat(result_dfs)

    return preds_df