import numpy as np
from flask import Flask, request, render_template
import pickle

# Create an app object using the Flask class
app = Flask(__name__)

# load the trained model (Pickle)
model = pickle.load(open('RF_classifier_model', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The probability of getting endorsed as Church Planter is  {}'.format(output))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
    # '''
    # For direct API calls trought request
    # '''
    # data = request.get_json(force=True)
    # prediction = model.predict([np.array(list(data.values()))])

    # output = prediction[0]
    # return jsonify(output)

if __name__ == "__main__":
    app.run()
