# import packages
import sys
import re
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from sqlalchemy import create_engine

def load_data(data_file):
    # read in file
    engine = create_engine('sqlite:///disaster_response.db')
    df = pd.read_sql('select * from disaster_response', con=engine)

    # define features and label arrays
    X = df.message
    Y = df.loc[:, 'related':]
    target_names = Y.columns
    y = Y.to_numpy()

    return X, y

def tokenize(message, stem='lemm'):
    """Text processing.
    
    Args:
        stem(str): stem or lemm.
        
    Returns:
        list: Cleaned tokens.
    """
    # 1. Cleaning
    
    # 2. Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", message.lower())
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stop Word Removal
    stop_words = stopwords.words("english")
    tokens = list(filter(lambda w: w not in stop_words, tokens))
    
    # 5. Part of Speech Tagging / Named Entity Recognition
    
    # 6. Stemming or Lemmatization
    # Because the targets are not roots, we should use Lemmatization
    
    clean_tokens = []
    if stem == 'stem':
        stemmer = PorterStemmer()
        for tok in tokens:
            clean_tok = stemmer.stem(tok).strip()
            clean_tokens.append(clean_tok)
    else:
        lemmatizer = WordNetLemmatizer()
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    # text processing and model pipeline
    # Neural Network needs much more time for modeling
    # mlp = MLPClassifier(random_state=42, max_iter=200, verbose=True, early_stopping=True)
    forest = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('multi_clf', MultiOutputClassifier(forest))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'vect__smooth_idf': (True, False),
        'vect__use_idf': (True, False),
        'multi_clf__estimator__max_depth': (None, 300, 500),
        'multi_clf__estimator__n_estimators': [50, 100, 200, 300],
        'multi_clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy', verbose=0, n_jobs=-1)

    return cv


def train(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit model
    model = build_model()
    model.fit(X_trian, y_train)

    # output model test results
    

    return model


def export_model(model):
    # Export model as a pickle file
    pass


def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline

sys.argv[1]