import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # 1. Load two csv files.
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # 2. Merge datasets.
    # Don't use merge here since the two datasets have duplicated rows
    df = pd.concat([messages, categories], axis=1)

    return df


def clean_data(df):
    # 3. Split categories into separate category columns
    categories = df.categories.str.split(';', expand=True)
    # 3.1 Rename categories' column names from the first row.
    row = categories.loc[0, :]
    category_colnames = row.str.split('-', expand=True)[0]
    categories.columns = category_colnames

    # 4. Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # 5. Replace categories column in df with new category columns.
    # 5.1 Drop the old categories column from 'df'
    df.drop(columns=['categories'], inplace=True)
    # 5.2 Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # 6. Remove duplicates.
    # use ignore_index=True to reset index
    df.drop_duplicates(inplace=True, ignore_index=True)

    return df


def save_data(df, database, table):
    # 7. Save the clean dataset into an sqlite database.
    try:
        engine = create_engine(f'sqlite:///{database}.db')
        df.to_sql(table, con=engine, index=False)
        print('\nCleaned data saved to database!')
    except ValueError:
        print(f"\nValueError: Table '{table}' already exists.")
        print(f"Try another table name. Program exit.")


def main():
    """The code in process_data.py is from '../../../../code/ETL Pipeline Preparation.ipynb'.
    """
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database, table = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database}.db\n    TABLE: {table}')
        save_data(df, database, table)
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database and table name to save the cleaned data '\
              'to as the third argument. \n\nExample: python3 process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'disaster_response disaster_response')


if __name__ == '__main__':
    main()
