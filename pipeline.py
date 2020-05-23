'''
Helper functions for analysis.py
'''
### I've commented out the modules/functions I don't think we're actually using,
### but we should clean these up before we submit.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import GaussianNB
    #, accuracy_score, mean_absolute_error,
# import datetime


def get_summary_stats(df):
    '''
    Print out general summary statistics on a Pandas DataFrame

    Input:
        df: a Pandas DataFrame
        corr (bool): an optional parameter to print out correlations between
                     variables

    Returns:
        Nothing, prints summary statistics
    '''

    print(df.describe())
    print(df.corr())


def split_data(df, random_state=246012349, test_size=0.2):
    '''
    Split a DataFrame into Training and Testing using Scikit Learn

    Input:
        df: a Pandas DataFrame
        random_state (int): a random seed
        test_size (float): the percentage of the data for the testing data

    Returns:
        train, test (pandas DataFrames) with imputed values
    '''

    train, test = train_test_split(df, test_size=test_size,
                                   random_state=random_state)

    print("Training set contains {} observations".format(len(train)))
    print("Testing set contains {} observation\n".format(len(test)))

    return train, test


def find_outliers(df, col, min_value, max_value):
    '''
    Identify the outliers in a particular column for spotchecking values

    Input:
        df: a dataframe
        col: the column to inspect
        min_value: the lowest reasonable value for the column
        max_value: the highest reasonable value for the column

    Returns:
        the number of records falling outside the specified bounds,
        what those values are, and those records
    '''
    rv = df[(df[col] < min_value) | (df[col] > max_value)]
    print("Number of outliers found: {}".format(len(rv)))
    print("Outlier values found: {}".format(rv[col].unique()))

    return rv


def impute_missing(train, test, num_cols):
    '''
    Impute missing values of continuous variables using the median value from
    the training data

    Input:
        train_df: the training DataFrame
        test_df: the testing DataFrame
        num_cols (list): a list of column names to impute missing values on

    Returns:
        train, test (pandas DataFrames) with imputed values
    '''

    for col in num_cols:
        med = train[col].median()
        if train[col].isnull().values.any():
            print('Imputing {} missing values with median {}'.format(col, med))
            train.loc[:, col] = train[col].fillna(med)
            test.loc[:, col] = test[col].fillna(med)

    return train, test


def normalize(train, test, atts):
    '''
    Normalize continuous variables based on the training data
    Input:
        train (pandas dataframe): dataframe of features for train observations
        test (pandas dataframe): dataframe of features for test observations
        atts (list of strings): list of attribute names to normalize
    Returns:
        train, test (pandas DataFrames) with normalized values
    '''
    scaler = StandardScaler()
    train[atts] = scaler.fit_transform(train[atts])
    test[atts] = scaler.transform(test[atts])
    return train, test


def hot_encode(df, cols):
    '''
    Perform one-hot encoding of categorical variables

    Input:
        df: a Pandas DataFrame
        cols (list): a list of categorical columns on which to perform
                     one-hot encoding

    Returns:
        pandas DataFrame with one-hot encoding performed
    '''

    return pd.get_dummies(df, columns=cols)


def evaluate_model(model, test_features, test_target):
    '''
    Calculates mean squared error, R-squared, and adjusted R-squared

    Input:
        model (GridSearch CV object)
        test_features (pandas DataFrame)
        test_target (pandas DataFrame)

    Returns:
        mse, r2, adj (floats): evaluation metrics (also prints)
    '''

    predicted = model.predict(test_features)
    mse = mean_squared_error(test_target, predicted)
    r2 = r2_score(test_target, predicted)

    n, p = test_features.shape
    adj = calc_adjr2(r2, n, p)

    print('Mean Squared Error: {}\nR-Squared: {}\nAdjusted R-Squared: {}'\
        .format(mse, r2, adj))

    return mse, r2, adj


def rank_coefs(model_result, labels):
    '''
    Ranks ceofficients from largest absolute value

    Input:
        model_result (GridSearch CV object)
        lables (list-like attribute names in order of DataFrame)

    Returns:
        ranked_coeffs (pandas DataFrame): feature label, coefficient, and
            absolute value of coefficient
    '''

    features = np.array(labels).reshape(-1, 1)
    coefficients = model_result.best_estimator_.steps[-1][1].coef_.reshape(
        -1, 1)

    ranked_coeffs = pd.DataFrame(np.concatenate((features, coefficients),
                                                axis=1))
    ranked_coeffs['abs'] = [abs(x) for x in ranked_coeffs[1]]

    ranked_coeffs = ranked_coeffs.sort_values('abs', ascending=False)
    ranked_coeffs.columns = ['feature', 'coefficient', 'coefficient (absolute)']

    return ranked_coeffs


def calc_adjr2(r2, n, p):
    '''
    Calculates adjusted R-squared value

    Input:
        r2 (float)
        n (int): number of observations
        p (int): number of features

    Return:
        adjusted R-2 (float)
    '''

    comp_r2 = 1 - r2
    num = n - 1
    denom = n - p - 1

    return 1 - (comp_r2 * (num/denom))



### I BELIEVE THESE INFECTIONS WE DON'T NEED; JUST COMMENTING OUT FOR NOW
# def discretize(df, col, bins):
#     '''
#     Discretize a variable into a specified number of bins

#     Input:
#         df: a Pandas DataFrame
#         col (str): the name of the column to discretize
#         bins (int): the number of bins to split along

#     Return:
#         A series of discretized values
#     '''

#     return pd.cut(df[col], bins)


# def grid_search(train_f, train_t, test_f, test_t, models, grid):
#     '''
#     Build and evaluate classifiers for input models and grid

#     Input:
#         train_f (df): the training features
#         train_t (series): the training target
#         test_f (df): the testing features
#         test_t (series): the testing target
#         models (dict): a dictionary of models
#         grid (dict): a dictionary of parameters for grid search

#     Return:
#     '''

#     # Begin timer
#     start = datetime.datetime.now()

#     # Initialize results data frame
#     results = pd.DataFrame(columns=['model', 'parameters', 'accuracy_score',
#                                     'mean_squared_error',
#                                     'mean_absolute_error'])

#     # Loop over models
#     for model_key in models.keys():

#         # Loop over parameters
#         for params in grid[model_key]:
#             print("Training model:", model_key, "|", params)

#             # Create model
#             model = models[model_key]
#             model.set_params(**params)

#             # Fit model on training set
#             model.fit(train_f, train_t)

#             # Predict on testing set
#             target_predict = model.predict(test_f)

#             # Evaluate predictions
#             evaluation = evaluate_classifiers(model_key, str(params), test_t,
#                                               target_predict)

#             # Store results in your results data frame
#             results = results.append(evaluation, ignore_index=True)

#     # End timer
#     stop = datetime.datetime.now()
#     print("Time Elapsed:", stop - start)

#     return results


# def evaluate_classifiers(model_key, params, test_t, test_p):
#     '''
#     Return the accuracy score, MSE, MAE, and R^2 for a regression

#     Input:
#         model:
#         params:
#         test_t: the test target
#         test_p: the test prediction

#     Return:
#         a tuple of evaluation metrics
#     '''

#     acc = accuracy_score(test_t, test_p)
#     mse = mean_squared_error(test_t, test_p)
#     mae = mean_absolute_error(test_t, test_p)

#     return {'model': model_key, 'parameters': params,
#             'accuracy_score': acc, 'mean_squared_error': mse,
#             'mean_absolute_error': mae}




# def top_pred(model, pred_names, num_preds=5):
#     '''
#     Determine the most important features in a logistic regression model.
#     Input:
#         model: the logistic regression model
#         pred_names: (list-like) the names of the predictors
#         num_preds: how many predictors to include
#     Return:
#         the top predictors
#     '''
#     #code source: adapted from a suggestion on
#     #https://stackoverflow.com/questions/43576614/logistic-regression-how-to-find-top-three-feature-that-have-highest-weights
#     coefs = model.coef_[0]
#     top = np.argpartition(coefs, -1 * num_preds)[-1* num_preds:]

#     return pred_names[top]
