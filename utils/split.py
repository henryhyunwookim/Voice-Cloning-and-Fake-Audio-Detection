from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from sklearn.utils import shuffle


def split_data(X, y, test_size, stratify=None, random_state=None, oversampling=False):
    if random_state != None:
        if oversampling:
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            X_test, y_test = SMOTE().fit_resample(X_test, y_test)
            return X_train, X_test, y_train, y_test
        else:
            return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, stratify=stratify)
    

def custom_train_test_split(df, train_size, label_col='label'):
    train_idx = []
    test_idx = []
    label_counts = defaultdict(lambda: 0)
    for idx, row in shuffle(df).iterrows(): # df.sort_values('label').iterrows()
        label = row[label_col]
        
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

        if label_counts[label] <= (10 * train_size):
            train_idx.append(idx)
        else:
            test_idx.append(idx)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    train_X = train_df.drop([label_col], axis=1)
    train_y = train_df[label_col]

    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_X = test_df.drop([label_col], axis=1)
    test_y = test_df[label_col]

    return train_df, train_X, train_y, test_df, test_X, test_y