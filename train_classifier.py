
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
        model (sklearn.pipeline): scikit learn pipeline model
    '''
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenizer.tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Description: prints the macro average f1-score, precision and recall values
    for the targets

    Arguments:
        model (sklearn.pipeline): scikit learn pipeline model
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
        model_filepath (str): filename for serialized model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        assert os.environ['NLTK_DATA'], "please export NLTK_DATA environment variable"
        print("NLTK_DATA:", os.environ['NLTK_DATA'])
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

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
