# Model Hyperparameter Tuning Techniques Comparison

This project compares various hyperparameter tuning techniques for optimizing machine learning models. It includes implementations of different tuning methods using popular libraries such as scikit-learn, scikit-optimize (skopt), hyperopt, Optuna, and manual grid search/random search.

## Techniques Implemented

1. **Bayesian Optimization with skopt (`bayesian_search.py`)**:
   - Utilizes Gaussian process-based Bayesian optimization for hyperparameter tuning.
   - Defines a custom optimization function using `gp_minimize` from skopt.
   - Searches the parameter space for RandomForestClassifier hyperparameters.
   - Uses 5-fold cross-validation for evaluation.

2. **Custom Scoring with scikit-learn (`custom_scoring.py`)**:
   - Implements hyperparameter tuning using scikit-learn's `RandomizedSearchCV`.
   - Defines a custom parameter grid and scoring metric (`accuracy`) for RandomForestClassifier.
   - Performs randomized search over the parameter grid using cross-validation.

3. **Grid Search with scikit-learn (`grid_search.py`)**:
   - Implements hyperparameter tuning using scikit-learn's `GridSearchCV`.
   - Searches through a predefined grid of hyperparameters for RandomForestClassifier.
   - Evaluates model performance using 5-fold cross-validation.

4. **Hyperopt Optimization (`hyperopt_search.py`)**:
   - Utilizes tree-structured Parzen Estimator (TPE) algorithm for hyperparameter optimization.
   - Defines a search space and optimization function using Hyperopt library.
   - Searches for optimal hyperparameters for RandomForestClassifier using 5-fold cross-validation.

5. **Optuna Optimization (`optuna_search.py`)**:
   - Implements hyperparameter optimization using Optuna library.
   - Defines an objective function and search space for RandomForestClassifier hyperparameters.
   - Searches for optimal hyperparameters using Bayesian optimization with TPE sampler.

6. **Random Search with scikit-learn (`random_search.py`)**:
   - Performs hyperparameter tuning using scikit-learn's `RandomizedSearchCV`.
   - Searches randomly across the parameter space for RandomForestClassifier.
   - Utilizes 5-fold cross-validation for evaluation.

## Dataset
- The project utilizes the train.csv dataset for training machine learning models.
- The dataset contains features and target variable ('price_range').

## Getting Started
1. Clone the repository.
2. Ensure you have the required dependencies installed (`pandas`, `numpy`, `scikit-learn`, `scikit-optimize`, `hyperopt`, `optuna`, `dlib`).
3. Run each Python script to see the implementation of different hyperparameter tuning techniques.
4. Compare the results obtained by different methods to understand their performance.

## Conclusion
- Compare the performance and efficiency of each hyperparameter tuning method based on their results.
- Consider factors such as computational resources, tuning time, and optimization effectiveness when choosing a method for model tuning in practical applications.


🙂 Feel free to contribute, provide feedback, or suggest improvements to the project!
