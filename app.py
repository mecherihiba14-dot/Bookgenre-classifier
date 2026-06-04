from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def classify():
    prediction = None
    confidence = None
    mode = None
    title = None
    description = None

    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()

        # Combine exactly as training did: title + space + description
        if description:
            text_input = title + ' ' + description
            mode = 'Title + description'
        else:
            text_input = title
            mode = 'Title only'

        vectorized = vectorizer.transform([text_input])
        prediction = model.predict(vectorized)[0]

        # Confidence score (if your model supports it)
        try:
            proba = model.predict_proba(vectorized)[0]
            confidence = round(max(proba) * 100)
        except:
            confidence = None

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           mode=mode,
                           title=title,
                           description=description)

if __name__ == '__main__':
    app.run(debug=True)