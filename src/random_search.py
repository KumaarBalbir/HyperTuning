import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics

if __name__ == "__main__":
    df = pd.read_csv('../input/train.csv')
    X = df.drop('price_range', axis=1).values  # features numpy array
    y = df['price_range'].values  # target numpy array

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        # 100 to 1500 no of estimators with step size of 100
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 20),
        "criterion": ["gini", "entropy"]
    }
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        verbose=1,
        n_jobs=1,
        cv=5
    )
    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
