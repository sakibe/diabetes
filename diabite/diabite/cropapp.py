from flask import Flask, render_template, request
import numpy as np

from sklearn.neighbors    import KNeighborsClassifier 
import joblib

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        model = joblib.load('cropmodel.pkl')
        model1 = joblib.load('cropmodeldd.pkl')
        testdata = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = model.predict(testdata)
        prediction1 = model1.predict(testdata)

        result=f"Result of knn = {prediction[0]} & Result of Decision Tree = {prediction1[0]}"

        return  render_template('index.html', crop_prediction=result)
    else : 
        return  render_template('index.html')   


if __name__ == "__main__":
    app.run()
