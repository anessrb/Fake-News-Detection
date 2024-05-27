from flask import Flask, request, render_template
import pickle
from text_normalizer import TextNormalizer  # Importer TextNormalizer

# Télécharger les ressources nltk nécessaires
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

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
