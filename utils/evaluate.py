# Classifiers
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Transformers
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures

# Decomposers
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, TruncatedSVD
from sklearn.cluster import FeatureAgglomeration

# Model selection, evaluation, and other stats functions
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error

# Others
import pandas as pd
import numpy as np
import time
import operator
from math import sqrt

# Custom function
from utils.plot import plot_scores


def update_dicts(clf, X, y, n_splits, n, eval_dict, scores_dict, key, random_state, _type):
    start_time = time.time()
    scores = cross_val_score(clf, X, y, scoring='roc_auc', # roc_auc, accuracy, f1
                             cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n, random_state=random_state))
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    
    eval_dict[key] = {
        "Mean": round(np.mean(scores), 4),
        "Std": round(np.std(scores), 4),
        "Max": round(np.max(scores), 4),
        "75th Percentile": round(np.percentile(scores, 75), 4),
        "Median": round(np.median(scores), 4),
        "25th Percentile": round(np.percentile(scores, 25), 4),
        "Min": round(np.min(scores), 4),
        "Time elapsed": time_elapsed}
    scores_dict[key] = scores

    print(f"{_type}: {key}")
    print(eval_dict[key], "\n")

    return eval_dict, scores_dict


def get_dicts(X, y, n_splits, n, random_state,
              classifiers=None, classifier=None,
              transformers=None, transformer=None,
              decomposers=None, decomposer=None,
              combinations=None):
    eval_dict = {}
    scores_dict = {}
    if classifiers != None:
        for classifier in classifiers:
            key = type(classifier).__name__
            eval_dict, scores_dict = update_dicts(classifier, X, y, n_splits, n,
                                                  eval_dict, scores_dict, key, random_state,
                                                  "Classifier")

    elif combinations != None:
        if classifier == None:
            raise Exception("You need to specify a classifier!")
        for features in combinations:
            key = ", ".join(features)
            eval_dict, scores_dict = update_dicts(classifier, X[features], y, n_splits, n,
                                                  eval_dict, scores_dict, key, random_state,
                                                  "Features")
            
    elif transformers != None:
        if classifier == None:
            raise Exception("You need to specify a classifier!")
        for transformer in transformers:
            clf = make_pipeline(transformer, classifier)
            eval_dict, scores_dict = update_dicts(clf, X, y, n_splits, n,
                                                  eval_dict, scores_dict, transformer, random_state,
                                                  "Transformer")

    elif decomposers != None:
        if classifier == None:
            raise Exception("You need to specify a classifier!")
        for decomposer in decomposers:
            clf = make_pipeline(decomposer, classifier)
            eval_dict, scores_dict = update_dicts(clf, X, y, n_splits, n,
                                                  eval_dict, scores_dict, decomposer, random_state,
                                                  "Decomposer")
            
    else:
        raise Exception("Failed to determine what to evaluate.")

    scores_dict_sorted = dict(sorted(scores_dict.items(), key=lambda x: np.mean(x[1]), reverse=True))

    return eval_dict, scores_dict_sorted


def eval_models(X, y, n_splits, n, random_state):
    classifiers = [
        LogisticRegression(random_state=random_state), # One of the most basic classifiers
        RandomForestClassifier(random_state=random_state, max_depth=2), # decision tree ensemble model where each tree is built independently
        XGBClassifier(random_state=random_state) # eXtreme Gradient Boosting; decision tree ensemble model where each tree is built one after another
    ]

    eval_dict, scores_dict = get_dicts(X, y, n_splits, n, random_state,
                                       classifiers=classifiers)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_classifier_str = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_classifier_str} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


def eval_feature_combinations(X, y, n_splits, n, random_state, classifier):
    combinations = [
        ["X1", "X3"],
        ["X1", "X5"],
        ["X1", "X6"],
        ["X1", "X3", "X5"],
        ["X1", "X3", "X6"],
        ["X1", "X5", "X6"]
    ]
    
    eval_dict, scores_dict = get_dicts(X, y, n_splits, n, random_state,
                                       classifier=classifier,
                                       combinations=combinations)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_features = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_features} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


def eval_transformers(X, y, n_splits, n, random_state, classifier, transformers=None):
    if transformers==None:
        transformers = [
            SkewedChi2Sampler(random_state=random_state),
            PolynomialCountSketch(random_state=random_state),
            AdditiveChi2Sampler(),
            RBFSampler(random_state=random_state),
            PolynomialFeatures()
        ]
    eval_dict, scores_dict = get_dicts(X, y, n_splits, n, random_state,
                                       classifier=classifier,
                                       transformers=transformers)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_transformer = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_transformer} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


def eval_decomposers(X, y, n_splits, n, random_state, classifier, decomposers=None):
    if decomposers==None:
        decomposers = [
            # PCA(n_components=0.95, svd_solver='full'),
            PCA(random_state=random_state),
            KernelPCA(random_state=random_state),
            IncrementalPCA(),
            TruncatedSVD(),
            FeatureAgglomeration()
        ]
    
    eval_dict, scores_dict = get_dicts(X, y, n_splits, n, random_state,
                                       classifier=classifier,
                                       decomposers=decomposers)

    plot_scores(scores_dict)

    eval_df = pd.DataFrame(eval_dict).T.sort_values(
        ["Mean", "Max", "Min", "Std", "Time elapsed"],
        ascending=[False, False, True, True, True])
    
    best_decomposer = eval_df.index[0]
    best_accuracy = eval_df.iloc[0, 0]
    print(f"{best_decomposer} yielded the best mean accuracy score of {best_accuracy}")
    return eval_df


def hyperparameter_tuning_and_evaluation(n_iter, cv, random_state, X_train, y_train, X_test, y_test, params):
    classifier = LogisticRegression(random_state=random_state)

    best_search = RandomizedSearchCV(estimator=classifier,
                                    param_distributions=params,
                                    scoring="f1",
                                    n_iter=n_iter, # n_iter=10 by default
                                    random_state=random_state,
                                    cv=cv)
    best_search.fit(X_train, y_train)
    # print(best_search.best_estimator_, "\n", best_search.best_score_, "\n", best_search.best_params_)

    score = accuracy_score(y_test, best_search.best_estimator_.predict(X_test))
    print(f"n_iter: {n_iter}, cv: {cv}, best_score: {round(best_search.best_score_, 2)}, test_score: {round(score, 2)}")
    
    return best_search


def get_sorted_dict(keys, values):
    sorted_dict = {}
    for k, v in zip(keys, values):
        sorted_dict[k] = v

    return dict(sorted(sorted_dict.items(), key=operator.itemgetter(1), reverse=True))


def group_items(_dict, group_keys):
    grouped_dict = {}
    for group_key in group_keys:
        for k, v in _dict.items():
            if group_key in k:
                if group_key in grouped_dict:
                    grouped_dict[group_key] += v
                else:
                    grouped_dict[group_key] = v
    return dict(sorted(grouped_dict.items(), key=operator.itemgetter(1), reverse=True))


def get_mean_stats(xgb_train_result, xgb_train_result_updated,
                   xgb_test_result, xgb_test_result_updated,
                   lgbm_train_result, lgbm_train_result_updated,
                   lgbm_test_result, lgbm_test_result_updated):
    overall_mean_dict = {}

    overall_mean_dict["xgb_train"] = round(xgb_train_result.mean(), 4).to_dict()
    overall_mean_dict["xgb_train_updated"] = round(xgb_train_result_updated.mean(), 4).to_dict()
    overall_mean_dict["xgb_test"] = round(xgb_test_result.mean(), 4).to_dict()
    overall_mean_dict["xgb_test_updated"] = round(xgb_test_result_updated.mean(), 4).to_dict()

    overall_mean_dict["lgbm_train"] = round(lgbm_train_result.mean(), 4).to_dict()
    overall_mean_dict["lgbm_train_updated"] = round(lgbm_train_result_updated.mean(), 4).to_dict()
    overall_mean_dict["lgbm_test"] = round(lgbm_test_result.mean(), 4).to_dict()
    overall_mean_dict["lgbm_test_updated"] = round(lgbm_test_result_updated.mean(), 4).to_dict()

    return pd.DataFrame(overall_mean_dict)


def return_result_table(results):    
    scores_list = []
    for result in results:
        scores = [round(np.mean(result), 4), round(np.std(result), 4), round(np.max(result), 4), round(np.min(result), 4)]
        scores_list.append(scores)

    result_table = pd.DataFrame(scores_list,                
                                columns=['Mean', 'Std', 'Max', 'Min'],
                                index=['Train loss', 'Validate loss', 'Test loss', 'Train f1', 'Validate f1', 'Test f1']).T

    return result_table


def get_capital_returns(results_dfs, initial_balance=0):
    capital_return_dict = {}
    for interval, df in results_dfs.items():
        last_row = df.iloc[-1]
        capital_return = round(last_row['Balance'] - initial_balance, 4)
        # remaining_stock = last_row['No. of Stock']
        # last_closing_price = last_row['Price']
        
        capital_return_dict[f'{interval} Trading'] = {'Capital Return': capital_return,
                                                        # 'Remaining Stock': remaining_stock,
                                                        # 'Last Closing Price': last_closing_price
                                                        }

    capital_return_df = pd.DataFrame.from_dict(capital_return_dict).T

    return capital_return_df


def evaluate_predictions(y_test, preds_df, keys, plot_df=True):
    if plot_df:
        preds_df.plot(figsize=(12,4));
    eval_list = [[key,
                  sqrt(mean_squared_error(y_test, preds_df[key].values)),
                  mean_absolute_percentage_error(y_test, preds_df[key].values),
                  r2_score(y_test, preds_df[key].values)] for key in keys]
    eval_df = pd.DataFrame(eval_list,
                           columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Percentage Error', 'R2 Score'])\
                            .sort_values(['Root Mean Squared Error', 'Mean Absolute Percentage Error', 'R2 Score'],
                                         ascending=[True, True, False])
    
    return eval_df