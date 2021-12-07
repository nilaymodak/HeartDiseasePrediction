from flask import Flask, render_template, request
from pickle import load
import numpy as np

app = Flask(__name__)

#loading the model and the scaler
model = load(open('RandomForestClassifier.pkl','rb'))
scaler = load(open('minmaxscaler.pkl','rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['Age'])
        sex = float(request.form['Gender'])
        cp = float(request.form['ChestPain'])
        trestbps = float(request.form['RestingBP'])
        chol = float(request.form['Chol'])
        fbs = float(request.form['FstBldSug'])
        restecg = float(request.form['RestingECG'])
        thalach = float(request.form['MaxHR'])
        exang = float(request.form['ExerAngina'])
        oldpeak =  float(request.form['STDep'])
        slope = float(request.form['STSlope'])
        ca = float(request.form['MjrFlo'])
        thal = float(request.form['Defect'])

        data = np.array([age,sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,thal])
        reshaped_data = data.reshape(1,-1)

        scaler.transform(reshaped_data)
        prediction = model.predict(reshaped_data)
        if prediction==1:
            return render_template('index.html',prediction_text="This patient has heart disease.")
        else:
            return render_template('index.html', prediction_text ="This patient does not have heart disease.")

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
