"""Code for our app"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

from .data_wrangling import wrangle

def create_app():
    app = Flask(__name__)

    # load model
    model = pickle.load(open('model.pkl','rb'))

    # prevent errors
    CORS(app)

    # List of features to use in request
    PARAMETERS = [
        'neighbourhood_group_cleansed', 'room_type', 'accommodates',
        'bathrooms', 'bedrooms', 'beds', 'bed_type', 'security_deposit',
        'cleaning_fee', 'minimum_nights', 'amenities'
    ]

    # routes
    @app.route('/', methods=['GET'])
    def predict():

    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
