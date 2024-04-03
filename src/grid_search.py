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
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [1, 3, 5, 7],
        "criterion": ["gini", "entropy"]
    }
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=1,
        n_jobs=1,
        cv=5
    )
    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
