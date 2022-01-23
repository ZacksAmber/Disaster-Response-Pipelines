#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################################
# File Name: model_tuning.py                                                   #
# File Path: /model_tuning.py                                                  #
# Created Date: 2022-01-19                                                     #
# -----                                                                        #
# Company: Zacks Shen                                                          #
# Author: Zacks Shen                                                           #
# Blog: https://zacks.one                                                      #
# Email: <zacks.shen@gmail.com>                                                #
# Github: https://github.com/ZacksAmber                                        #
# -----                                                                        #
# Last Modified: 2022-01-19 6:29:34 pm                                         #
# Modified By: Zacks Shen <zacks.shen@gmail.com>                               #
# -----                                                                        #
# Copyright (c) 2022 Zacks Shen                                                #
################################################################################

# import packages
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
import re
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(data_file):
    # read in file
    engine = create_engine(f'sqlite:///{data_file}.db')
    df = pd.read_sql(f'select * from {data_file}', con=engine)

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
    vect = TfidfVectorizer(tokenizer=tokenize, use_idf=True, smooth_idf=True)
    svd = TruncatedSVD(random_state=42)
    forest = RandomForestClassifier(random_state=42)
    multi_label_clf = MultiOutputClassifier(forest)

    pipeline = Pipeline([
        ('vect', vect),
        ('svd', svd),
        ('multi_clf', MultiOutputClassifier(forest))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'svd__n_components': [500, 1000, 3000],
        'multi_clf__estimator__max_depth': (None, 300, 500),
        'multi_clf__estimator__n_estimators': [50, 150, 300],
        'multi_clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid=parameters,
                      scoring='f1_macro', verbose=2)

    return cv


def tuning(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with open('best_params.txt', 'w') as f:
        f.write(model.best_params_)

def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()
    tuning(X, y, model)  # tuning


if __name__ == '__main__':
    data_file = 'disaster_response'  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline

