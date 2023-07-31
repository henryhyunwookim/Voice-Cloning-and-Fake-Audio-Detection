from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


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