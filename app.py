from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        text_input = request.form['text_input']
        vectorized = vectorizer.transform([text_input])
        prediction = model.predict(vectorized)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)