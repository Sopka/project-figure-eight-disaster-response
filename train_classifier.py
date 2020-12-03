
import sys
import os

import pandas as pd
import pickle

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from tokenizer import tokenizer

def load_data(database_filepath):
    '''
    Description: load figure eight data from an sqlite3 database

    Arguments:
        database_filepath (str):   path to an sqlite3 database

    Returns:
        X (pandas.DataFrame): features
        Y (pandas.DataFrame): targets
        columns (...): column names of targets
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine)

    X = df['message']
    feature_columns = [c for c in df.columns.values.tolist(
    ) if c not in ['id', 'message', 'original', 'genre']]
    Y = df[feature_columns]
    return X, Y, Y.columns


def build_model():
    '''
    Description: build a scikit learn pipeline model consisting of 
                 several transformers and one estimator

    Returns:
        model: scikit learn model
    '''
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenizer.tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_df': [0.75, 1],
        'vect__max_features': [None, 2000, 5000],
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2,4]
    }

    model = GridSearchCV(model, param_grid=parameters, n_jobs=-1, verbose=10)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Description: prints the macro average f1-score, precision and recall values
    for the targets

    Arguments:
        model: scikit learn model
        X_test: test feature matrix
        Y_test: test target matrix
        category_names: target names

    Returns:
        report (pandas.DataFrame): the evaluation result as a dataframe
    '''
    Y_pred = model.predict(X_test)
    assert len(Y_pred[0]) == len(
        category_names),     "lenght of category names does not match number of targets"
    reportlist = []
    reportcolumns = ['f1-score', 'precision', 'recall']
    for i in range(len(category_names)):
        cr = classification_report(
            Y_test.iloc[:, i], Y_pred[:, i], output_dict=True, zero_division=0)
        report_row = [cr['macro avg'][crv] for crv in reportcolumns]
        reportlist.append(report_row)

    report = pd.DataFrame(
        reportlist, index=Y_test.columns, columns=reportcolumns)
    return report


def save_model(model, model_filepath):
    '''
    Description: serialize model in pickle format

    Arguments:
        model: scikit learn model
        model_filepath (str): filename for serialized model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def save_most_counted_tokens(model, X_train, database_filepath):
    '''
    Description: store the most counted tokens of the training data set

    Arguments:
        model: scikit learn model
        X_train ([str]): the training messages
        database_filepath (str):   path to an sqlite3 database
    '''
    # get the countvectorize from the best estimator of the gridsearch model
    best_countvectorizer = model.get_params()['estimator__steps'][0][1]
    #best_countvectorizer = model.get_params()['vect']
    # transform again to reproduce the count matrix
    wordcounts = best_countvectorizer.fit_transform(X_train)
    # compute the mean occurance and take the 20 largest values
    most_counted_tokens = pd.DataFrame(
        wordcounts.toarray()).mean().nlargest(20)
    # extract the numbered token names from the training phase/step
    numbered_featurenames = pd.DataFrame(
        best_countvectorizer.get_feature_names(),  columns=['tokens'])
    # select only the 20 tokens that occured most
    most_counted_featurenames = numbered_featurenames.loc[most_counted_tokens.index.values]
    # combine the token name and occurance in one table to export it
    token_occurance = pd.concat([most_counted_featurenames, pd.DataFrame(
        most_counted_tokens, columns=['occurance'])], axis=1)
    # export/store in the same database as the model data
    # will be used later in the web app for visualization
    engine = create_engine('sqlite:///' + database_filepath)
    token_occurance.to_sql('most_counted_tokens', engine,
                           index=False,  if_exists='replace')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        report = evaluate_model(model, X_test, Y_test, category_names)
        print(report)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('save most counted tokens to database...')
        save_most_counted_tokens(model, X_train, database_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
