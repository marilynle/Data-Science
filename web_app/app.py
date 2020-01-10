""" Code for our app """

# Standard imports
import pickle

# 3rd party imports
import numpy as np
import pandas as pd
import xgboost
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Local imports
from .data_wrangling import wrangle

def create_app():
    app = Flask(__name__)

    # load pickled model
    model = pickle.load(open('model.pkl','rb'))

    # prevent web errors
    CORS(app)

    @app.route('/')
    def index():
        return render_template('example_form.html', optimal_price="")

    @app.route('/predict', methods=['GET'])
    def predict():

        print('\n\n\n\nDEBUG TESTING\n\n\n\n')

        # defining a dictionary to store data in
        data = {}

        # List of features to use in request
        PARAMETERS = [
            'neighbourhood_group_cleansed', 'room_type', 'accommodates',
            'bathrooms', 'bedrooms', 'beds', 'bed_type', 'security_deposit',
            'cleaning_fee', 'minimum_nights'
            ]

        AMENITIES = [
           'Washer', 'Hair dryer', 'Laptop friendly workspace', 'Hangers',
           'Iron', 'Shampoo', 'TV', 'Hot water', 'Family/kid friendly', 'Internet',
           'Host greets you', 'Smoke detector', 'Buzzer/wireless intercom',
           'Lock on bedroom door', 'Free street parking', 'Elevator', 'Bed linens',
           'Smoking allowed', 'First aid kit', 'Cable TV' 
            ]

        print('\n\nGetting the request data\n\n')

        # load the data
        for param in PARAMETERS:
            print(f'{param} type:', type(request.args[param]))
            print(f'{param}:', request.args[param])
            try:
                data[param] = [int(request.args[param])]
            except:
                data[param] = [request.args[param]]


        print('\n\nAmenities:\n')

        for amenity in AMENITIES:
            if amenity in request.args.keys():

                print(f'{amenity} present! Value: {request.args[amenity]}')
                data[amenity.replace(' ', '_')] = 1
            else:
                data[amenity.replace(' ', '_')] = 0

        for arg in request.args.keys():
            print(f'{arg}: {request.args[arg]}')

        print('\n\nConverting to dataframe\n\n')

        # convert data into dataframe to be passed through the model
        data_df = pd.DataFrame.from_dict(data)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print('\n\n', data_df, '\n\n')

        # WRANGLING TIME!!!
        #data_df = wrangle(data_df)

        print('\n\nmaking a prediction\n\n')

        # making a prediction by passing a dataframe through the model
        result = int(model.predict(data_df)[0])

        # storing the result as a dict
        #output = {'results': int(result[0])}

        # create a string response to display
        response = f'The optimal nightly price is {result}'

        print('\n\n' + response + '\n\n')

        # convert the dict to a JSON object and return it
        return render_template('example_form.html', optimal_price=response)

    @app.route('/json', methods=['GET'])
    def json():

        # get JSON object from the GET request
        data = request.get_json(force=True)

        # encapsulate value y in a list for each value x
        data.update((x, [y]) for x, y in data.items())

        # convert data into dataframe to be passed through the model
        data_df = pd.DataFrame.from_dict(data)

        # WRANGLING TIME!!!
        data_df = wrangle(data_df)

        # making a prediction by passing a dataframe through the model
        result = model.predict(data_df)

        # storing the result as a dict
        output = {'results': int(result[0])}

        # convert the dict to a JSON object and return it
        return jsonify(output)

    # return the app object to be run
    return app

# if this file is main, run a development server
if __name__ == '__main__':
    create_app().run(debug=True)


