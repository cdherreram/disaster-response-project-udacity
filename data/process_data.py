import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
	Load the data from .csv files (messages and categories) and return a
	merged table.

	INPUTS -
	    messages_filepath - string - Path file to messsages file with .csv extension
	    categories_filepath - string - Path file to categories file with .csv extension
    """

    # Load the data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge the info on id column
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    """
	Clean the categories column and report each value for category in a final dataframe

	INPUTS -
	    
    """

    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[1,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.number)

        categories.loc[categories[column] > 1, column] = 1

    df.drop('categories', axis = 1, inplace=True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessageCategories', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
