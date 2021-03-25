#version 02 by Alex

#import python libraries
import pandas as pd
import sys
from sqlalchemy import create_engine


#load data

def load_data(messages_filepath, categories_filepath):
    
    """
    - Input out of two CSV files 
    - Import data as pandas dataframe 
    - Merge data into one combined dataframe
    
    Attributes / Parameters:
    messages_filepath str: Messages CSV file
    categories_filepath str: Categories CSV file
    
    Returns:
    df pandas_dataframe: Dataframe with combined data out of the two input dataframes
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id')
    
    return df   
    


def clean_data(df):
    
    """
    - Clean the data in the combined dataframe 
    - Target is to prepare data for ML model 
    
    Attributes / Parameters:
    df pandas_dataframe: dataframe from load_data() function
    
    Returns:
    df pandas_dataframe: data cleaned for the ML model
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:].values
    
    # use this row to extract a list of new column names for categories
    new_columns = [r[:-2] for r in row]

    # rename the columns of `categories`
    categories.columns = new_columns

    # Convert category values to just numbers 0 or 1.
    
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    # For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    # You can perform normal string actions on Pandas Series, like indexing, by including .str after the Series. 
    # You may need to first convert the Series to be of type string, which you can do with astype(str).
    
    for column in categories:

        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from df
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new categories dataframe
    df[categories.columns] = categories

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df



def save_data(df, database_filename):
    
    """
    - Save cleaned data to an SQL database
    
    Attributes / Parameters:
    df pandas_dataframe: data from clean_data() function
    database_filename str: File path of SQL db into which cleaned data will be saved
    
    Returns:
    None
    """
    
    #engine = create_engine('sqlite:///{}'.format(database_file_name)) 
    #db_filename = database_filename.split("/")[-1] # extract file name from the file path
    #table_name = db_filename.split(".")[0]
    #df.to_sql(table_name, engine, index=False, if_exists = 'replace')
    
    database_filename = "DisasterResponse.db"
    engine = create_engine('sqlite:///{}'.format(database_filename)) 
    db_filename = database_filename.split("/")[-1] # extract file name from \the file path

    table_name = db_filename.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')
     


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