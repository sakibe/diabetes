from flask import Flask, render_template, request
import numpy as np

from sklearn.neighbors    import KNeighborsClassifier 
import joblib

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def predict():
    # Extract the form data
    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetespedigreefunction = float(request.form['diabetespedigreefunction'])
        age = float(request.form['age'])

        # Load the trained modelz
        model = joblib.load('diabmodel.pkl')
        model1 = joblib.load('diabmodeldd.pkl')
        # Make a prediction on the input data
        testdata = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])
        prediction = model.predict(testdata)
        prediction1 = model1.predict(testdata)

        result=f"Result of knn = {prediction[0]}, Result of Decision Tree = {prediction1[0]}"
        if prediction[0] & prediction1[0]==0:
         result = "no diabetes"
        else:
         result = "yes u have diabetes"

        return  render_template('diab.html', diabetes_prediction=result)
    else : 
        return  render_template('diab.html')   



if __name__ == "__main__":
    app.run(debug=True)
