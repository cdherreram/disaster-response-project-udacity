import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    graphs = []


    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    graph_1 = {
		'data': [
                	Bar(
	                    	x=genre_names,
        	            	y=genre_counts
                	)
            	],

            	'layout': {
                	'title': 'Distribution of Message Genres',
	                'yaxis': {
	                    'title': "Count"
	                },
	                'xaxis': {
	                    'title': "Genre"
	                }
        	    }
	        }
    # add to the graphs list
    graphs.append(graph_1)

    request_mean = df.groupby('request').mean()['search_and_rescue']
    request_names = list(request_mean.index)

    graph_2 = {
                'data': [
                        Bar(
                	        x=request_names,
                        	y=request_mean
                        )
                ],
                'layout': {
                        'title': 'Distribution of Search and Rescue by Request',
                        'yaxis': {
                            'title': "Mean"
                        },
                        'xaxis': {
                            'title': "Request"
                        }
                    }
                }

    graphs.append(graph_2)


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
