import json
import plotly
import pandas as pd

from tokenizer import tokenizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)
most_counted_tokens = pd.read_sql_table('most_counted_tokens', engine)

# load model
model = joblib.load("./models/model.pkl")
print("loaded model:", model)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    classifiers_mean = df[df.columns[4:]].mean()
    classifiers_mean.index = [ c.replace('_', ' ').title() for c in classifiers_mean.index.values]
    classifiers_mean.sort_values(ascending=False, inplace=True)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=classifiers_mean.index,
                    y=classifiers_mean.values
                )
            ],

            'layout': {
                'title': 'Distribution of Training Classifiers',
                'yaxis': {
                    'title': "Occurance of Classifier in percentage"
                },
                'xaxis': {
                    'title': "Classifier",
                    'automargins': True,
                    'tickangle': -30
                }
            }
        },
        {
            'data': [
                Bar(
                    x=most_counted_tokens.tokens,
                    y=most_counted_tokens.occurance
                )
            ],

            'layout': {
                'title': 'Most counted Tokens',
                'yaxis': {
                    'title': "Occurance of Tokens in percentage"
                },
                'xaxis': {
                    'title': "Token name",
                    'automargins': True,
                    'tickangle': -30
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
