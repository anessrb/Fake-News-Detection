from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Télécharger les ressources nltk nécessaires
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))



# Définir la fonction de nettoyage de texte
def MyCleanText(X, lowercase=True, removestopwords=False, removedigit=False, getstemmer=False, getlemmatisation=False):
    sentence = str(X)
    sentence = re.sub(r'[^\w\s]', ' ', sentence)
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)
    tokens = word_tokenize(sentence)

    if lowercase:
        tokens = [token.lower() for token in tokens]

    table = str.maketrans('', '', string.punctuation)
    words = [token.translate(table) for token in tokens]
    words = [word for word in words if word.isalnum()]

    if removedigit:
        words = [word for word in words if not word.isdigit()]

    if removestopwords:
        words = [word for word in words if not word in stop_words]

    if getlemmatisation:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

    if getstemmer:
        ps = PorterStemmer()
        words = [ps.stem(word) for word in words]

    sentence = ' '.join(words)
    return sentence


# Définir la classe de normalisation de texte
class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, removestopwords=False, lowercase=False, removedigit=False, getstemmer=False,
                 getlemmatisation=False):
        self.lowercase = lowercase
        self.getstemmer = getstemmer
        self.removestopwords = removestopwords
        self.getlemmatisation = getlemmatisation
        self.removedigit = removedigit

    def transform(self, X, **transform_params):
        X = X.copy()
        return [self.MyCleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def MyCleanText(self, X):
        sentence = str(X)
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)
        tokens = word_tokenize(sentence)

        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        table = str.maketrans('', '', string.punctuation)
        words = [token.translate(table) for token in tokens]
        words = [word for word in words if word.isalnum()]

        if self.removedigit:
            words = [word for word in words if not word.isdigit()]

        if self.removestopwords:
            words = [word for word in words if not word in stop_words]

        if self.getlemmatisation:
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]

        if self.getstemmer:
            ps = PorterStemmer()
            words = [ps.stem(word) for word in words]

        sentence = ' '.join(words)
        return sentence




app = Flask(__name__)

with open('svc_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']
    input_data = title + " " + text
    prediction = model.predict([input_data])
    prediction_label = 'True' if prediction[0] == 1 else 'False'
    return render_template('index.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
