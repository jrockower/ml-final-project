'''
Helper functions for analysis.py
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


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


def run_gridsearch(pl, params, train_features, train_target, head=False,
                   verbose=1):
    '''
    Runs a grid search on a pipeline object with specified parameters

    Inputs:
        pl: a pipeline object
        params: a dictionary consisting of parameters for a grid search
        train_features: a Pandas DataFrame including the training features
        train_target: a Pandas DataFrame including the training target
        head: a boolean specifying whether or not to return the first 5 rows
              of the ranking of the various models
        verbose: an integer specifying the level of progress output

    Returns:
        a tuple consisting of the model, the resulting model, and the ranked
        models based on how well they score
    '''

    grid_model = GridSearchCV(estimator=pl,
                              param_grid=params,
                              cv=5,
                              return_train_score=True,
                              scoring='neg_mean_squared_error',
                              iid=True,
                              verbose=verbose)

    grid_model_result = grid_model.fit(train_features, train_target)
    cv_results = pd.DataFrame(grid_model_result.cv_results_)

    cols = []

    for key in params:
        cols.append('param_' + key)

    cols.extend(['rank_test_score', 'mean_test_score'])
    cv_results = cv_results.sort_values('rank_test_score')[cols]

    if head:
        return cv_results.head()

    return grid_model, grid_model_result, cv_results


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

