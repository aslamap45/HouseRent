import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
df=pd.read_csv('C:/Users/ASLAM/Desktop/RentPrediction/pune_file.csv')
pipe = pickle.load(open("C:/Users/ASLAM/Desktop/RentPrediction/Model.pkl","rb"))



@app.route('/')
def index():

    locations = sorted(df['locality'].unique())
    return render_template('index.html', locality=locations)

@app.route('/predict', methods=['POST'])
def predict():
    locality = request.form.get('locality')
    bedroom = request.form.get('bedroom')
    bathroom = request.form.get('bathroom')
    area = request.form.get('area')

    print(locality, bedroom, bathroom, area)
    input = pd.DataFrame([[locality,bedroom,bathroom,area]],columns=['locality', 'bedroom', 'bathroom', 'area'])
    prediction = pipe.predict(input)[0]

    return str(np.round(prediction))

if __name__=="__main__":
    app.run(debug=True, port=5000)

    df.dtypes()