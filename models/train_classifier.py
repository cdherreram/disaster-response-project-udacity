import sys
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump

from sqlalchemy import create_engine

def load_data(database_filepath):
    """
        Load the data from a file to train the model.
        
        INPUT - 
            database_filepath - string - path to file
            
        OUTPUT -
            X,y - pandas dataframe - data to 
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('MessageCategories',engine)
    X = df['message']
    y = df.loc[:,'related':'direct_report']
    category_names = y.columns.tolist()
    return X, y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    return stemmed


def build_model():
    
    # Build pipeline
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    # Establish parameters
    parameters = {
        'vect__ngram_range': [(1, 1)],
        'vect__max_df': [0.5, 0.75, 1.0],
        'vect__max_features': [2500, 5000],
        'tfidf__use_idf': (True, False),
        'clf__estimator__max_depth': [64,128,256,512]
    }

    # BUild a grid with cross-validation
    return GridSearchCV(pipe, parameters, verbose=1)
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print('REPORT BY COLUMN:')
    for i in range(Y_test.shape[1]):
        print(f'For column \'{category_names[i]}\':')
        print(
            classification_report(
            Y_test.iloc[:,i],
            Y_pred[:,i]
            )
        )
    
def save_model(model, model_filepath):
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()