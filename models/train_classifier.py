# version 03 by Alex

# import python libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re #normalize text

# import libraries for natural language toolkit (nltk)
from nltk.tokenize import word_tokenize #tokenize text
from nltk.corpus import stopwords #non-englisch words; https://stackoverflow.com/questions/41967511/removing-non-english-words-from-corpus
from nltk.stem import WordNetLemmatizer

# import libraries for ML model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split

# import libraries to test ML model
from sklearn.metrics import classification_report

# import libraries to save ML model
import pickle


# import other libraries
#from pprint import pprint
#from sklearn.ensemble import RandomForestClassifier

#import time
#import warnings
#warnings.filterwarnings('ignore')


#solve problems with nltk in anaconda

#import nltk
#nltk.download()


def load_data(database_filepath):
    
    """
    - Loads data from SQL db
    
    Attributes / Parameters:
    database_filepath: SQL db file
    
    Returns:
    X pandas dataframe: dataframe of features
    Y pandas dataframe: dataframe of target values
    category_names list: Target labels 
    """
    
    # load data from SQL db
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con = engine)
    X = df['message']
    Y = df.iloc[:,4:]

    # Y['related'] contains three distinct values; mapping extra values 2 to 1
    
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    
    """
    - Tokenizes text data
    
    Attributes / Parameters:
    text str: Messages as text data
    
    Returns:
    words: list of words: Processed text after normalized & lemmatization & tokenization
    """
    
    #Tokenizes text data
    #Input: text str for Messages as text data
    #Output: words list for processed text (normalized & lemmatization & tokenization)
    #Info in GER: Lemmatisierung ist die Reduktion der Wortform auf ihre Grundform und wird auch lexikonbasiertes Stemming genannt. 
    #Info in GER: Dies wird dadurch erreicht, dass die erreichte Grundform in einem elektronischen Wörterbuch nachgeschlagen wird. 


    # normalize text
    # Help: https://stackoverflow.com/questions/6323296/python-remove-anything-that-is-not-a-letter-or-number
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    
    # tokenize text
    words = word_tokenize(text)
    
    
    # remove stop words
    # Help: https://stackoverflow.com/questions/41967511/removing-non-english-words-from-corpus
    stopwords_non_engl = stopwords.words("english")
    words = [word for word in words if word not in stopwords_non_engl]
    
    # lemmatization of words (output: root form of word)
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words


def build_model():
    
    # ML pipeline should take in the message column as input and output classification results on the other 36 categories
    
    # Build ML model with GridSearchCV
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    
    """
    - Build ML model with GridSearchCV
    
    Attributes / Parameters:
    non
    
    Returns:
    Trained model after performing GridSearchCV
    """
    
    # ML model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))])

    # parameters
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create ML model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):#
    
    """
    - Evaluate ML model performance on test data
    
    Attributes / Parameters:
    model: trained ML model
    X_test: test features
    Y_test: test targets
    category_names: target labels
    
    Returns: 
    non
    """

    # predict
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    
    """
    - Save the model in a pickle file    
    
    Attributes / Parameters:
    model: trained ML model
    model_filepath: Filepath to save the ML model
    """
    #Help: https://stackoverflow.com/questions/13906623/using-pickle-dump-typeerror-must-be-str-not-bytes
    
    # export and save ML model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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