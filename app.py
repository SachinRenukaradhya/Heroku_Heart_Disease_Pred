from flask import Flask, render_template,request,jsonify,url_for
import pickle
import numpy as np
from sklearn import feature_extraction
app=Flask(__name__)
pipe=pickle.load(open('heart_disease.pkl', 'rb'))
@app.route('/')
def Home():
   return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    feature_extraction=[float(x) for x in request.form.values()]
    features=np.array([feature_extraction],dtype='object').reshape(1,13)
    prediction=pipe.predict(features)
    return render_template('index.html', Prediction_text="The Report Prediction is 1-Yes, 0-No {}".format(prediction))

if (__name__)=='__main__':
    app.run(debug=True)