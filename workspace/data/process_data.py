import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Description: load messages and categories cvs files and merge them

    Arguments:
        messages_filepath (str):   path to messages csv file
        categories_filepath (str): path to categories csv file

    Returns:
        df (pandas.DataFrame):     a pandas DataFrame populated with messages   
                                   and categories from figure eight cvs files 
    '''
    # load csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Description: clean the messages and categories data

    Arguments:
        df (pandas.DataFrame): a pandas DataFrame populated with messages   
                                and categories from figure eight cvs files 

    Returns:
        df (pandas.DataFrame):  a pandas DataFrame populated with messages   
                                and categories from figure eight cvs files 
    '''

    # Split the values in the categories column on the ; character so that each
    # value becomes a separate column.
    #
    # dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # Use the first row of categories dataframe to create column names for the categories data.
    row = categories.iloc[[0]]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda s: s.str[0:-2]).values.tolist()[0]
    # Rename columns of categories with new column names.
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.

    # Iterate through the category columns in df to keep only the last
    # character of each string (the 1 or 0). For example, related-0 becomes 0,
    # related-1 becomes 1. Convert the string to a numeric value.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda s: s[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    if "categories" in df.columns:
        # Drop the categories column from the df dataframe since it is no
        # longer needed.
        df.drop(columns=["categories"], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    print("check for duplicates:")
    num_df_cols = df.shape[0]
    num_duplicates = num_df_cols - \
        df.duplicated(subset=['message']).value_counts()[0]
    if num_duplicates > 0:
        print('duplicates comparing only the message column', num_duplicates)
        # drop duplicates
        df.drop_duplicates(subset=['message'], inplace=True)
        print('check for duplicates after removal:')
        num_df_cols = df.shape[0]
        num_duplicates = num_df_cols - \
            df.duplicated(subset=['message']).value_counts()[0]
        assert num_duplicates == 0, "could not remove all duplicates"
        print('duplicates comparing only the message column', num_duplicates)

    return df


def save_data(df, database_filename):
    '''
    Description: store the data in an sqlite3 database

    Arguments:
        df (pandas.DataFrame):   a pandas DataFrame populated with messages
                                 and categories from figure eight cvs files
        database_filename (str): filename for the database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
