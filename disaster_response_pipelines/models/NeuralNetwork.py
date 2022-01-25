from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
import json
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


class NeuralNetwork:
    def load_data(self, database, table):
        engine = create_engine(f'sqlite:///../data/{database}')
        df = pd.read_sql_table(table, engine)

        # define features and label arrays
        X = df.message
        Y = df.loc[:, 'related':]
        category_names = Y.columns
        y = Y.to_numpy()

        return X, y, category_names

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    def load_params(self, jsonfile='best_params.json'):
        # The best params from hyperparameters tuning.
        with open('best_params.json') as f:
            best_params = json.load(f)

        return best_params

    def tokenize(self, message, stem='lemm'):
        """Text processing.

        Args:
            message(str): Message content.
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

    def build_model(self, params):
        # text processing and model pipeline
        vect = TfidfVectorizer(tokenizer=self.tokenize, use_idf=True, smooth_idf=True)
        svd = TruncatedSVD(random_state=42)
        mlp = MLPClassifier(random_state=42, early_stopping=True,
                            learning_rate='adaptive')

        multi_clf = MultiOutputClassifier(mlp)

        pipeline = Pipeline([
            ('vect', vect),
            # ('svd', svd),
            ('multi_clf', multi_clf)
        ])

        pipeline.set_params(**params)

        return pipeline

    def evaluate_model(self, model, X_test, y_test, category_names):
        y_pred = model.predict(X_test)

        scores = []
        for i in range(len(category_names)):
            scores.append(recall_score(
                y_test[:, i], y_pred[:, i], average='weighted'))

        print('Average Recall Score:', np.mean(scores))

    def export_model(self, database, table):
        print(f'\nLoading data...\n    DATABASE: {database}')
        X, y, category_names = self.load_data(database, table)
        X_train, X_test, y_train, y_test = self.train_test_split(
            X, y, test_size=0.2, random_state=42)

        print('Building model...')
        best_params = self.load_params()
        model = self.build_model(best_params)

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        self.evaluate_model(model, X_test, y_test, category_names)

        return model
