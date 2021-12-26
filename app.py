from flask import Flask, jsonify, render_template
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
api = Api(app)

model = pickle.load(open('models/rfr_model.pkl', 'rb'))
model_cols = pickle.load(open('models/rfr_model_columns.pkl', 'rb'))

data_arg = reqparse.RequestParser()
data_arg.add_argument("id", type=str)


class predict(Resource):

    def __init__(self):
        self.model1 = model

    def post(self):

        #####################
        # Load Input Values #
        #####################
        args = data_arg.parse_args()
        input_features = args.id.strip('][').split(',')

        num_feats = [float(i) for i in input_features[:7]]
        cat_feats = list(input_features)[7:]

        ##############################
        # Handle Categorical Columns #
        ##############################

        cat_feats = [x.lower().strip() for x in cat_feats]

        l1 = cat_feats[0]  # extract l1 value from input
        l1_cols = pd.Series(model_cols)
        # extract l1 values in training data
        l1_cols = list(l1_cols[l1_cols.str.startswith('l1')].values)
        # take off prefixes and standardize strings
        l1_cols = [x.split('_')[-1].lower() for x in l1_cols]
        l1_ohe = [1 if x == l1 else 0 for x in l1_cols]  # ohe value
        # if unseen value is entered, add it to 'other' column
        if sum(l1_ohe) == 0:
            l1_ohe[-1] = 1

        l2 = cat_feats[1]  # extract l2 value from input
        l2_cols = pd.Series(model_cols)
        # extract l2 values in training data
        l2_cols = list(l2_cols[l2_cols.str.startswith('l2')].values)
        # take off prefixes and standardize strings
        l2_cols = [x.split('_')[-1].lower() for x in l2_cols]
        l2_ohe = [1 if x == l2 else 0 for x in l2_cols]  # ohe value
        # if unseen value is entered, add it to 'other' column
        if sum(l2_ohe) == 0:
            l2_ohe[-1] = 1

        prop = cat_feats[2]  # extract property type value from input
        prop_cols = pd.Series(model_cols)
        # extract property type values in training data
        prop_cols = list(
            prop_cols[prop_cols.str.startswith('property_type')].values)
        # take off prefixes and standardize strings
        prop_cols = [x.split('_')[-1].lower() for x in prop_cols]
        prop_ohe = [1 if x == prop else 0 for x in prop_cols]  # ohe value
        # if unseen value is entered, add it to 'other' column
        if sum(prop_ohe) == 0:
            prop_ohe[6] = 1

        op = cat_feats[3]  # extract operation type value from input
        op_cols = pd.Series(model_cols)
        # extract operation type values in training data
        op_cols = list(
            op_cols[op_cols.str.startswith('operation_type')].values)
        # take off prefixes and standardize strings
        op_cols = [x.split('_')[-1].lower() for x in op_cols]
        op_ohe = [1 if x == op else 0 for x in op_cols]  # ohe value

        ###############################
        # Finalize Features & Predict #
        ###############################

        transformed_categoricals = l1_ohe + l2_ohe + prop_ohe + op_ohe
        final_features = num_feats + transformed_categoricals

        prediction = model.predict([final_features])

        output = prediction[0]
        print("prediction:", output)

        return jsonify({"Predicted House Price:": "$" + str(output)})


api.add_resource(predict, '/')


if __name__ == "__main__":
    app.run(debug=True)
