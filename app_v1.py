from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('models/rfr_model.pkl', 'rb'))
model_cols = pickle.load(open('models/rfr_model_columns.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    render prediction results.
    must first one hot encode categorical values. 
    11: bedrooms, bathrooms, surface_total, surface_covered, rooms, lat, lon <--- doubles 
                    l1, l2, property type, operation type <--- strings
    """
    #####################
    # Load Input Values #
    #####################

    input_features = request.form.values()
    print(input_features)
    num_feats = list(input_features)[:7]
    cat_feats = list(input_features)[7:]

    ##############################
    # Handle Categorical Columns #
    ##############################

    cat_feats = [x.lower() for x in cat_feats]

    l1 = num_feats[0]  # extract l1 value from input
    l1_cols = pd.Series(model_cols)
    # extract l1 values in training data
    l1_cols = list(l1_cols[l1_cols.str.startswith('l1')].values)
    # take off prefixes and standardize strings
    l1_cols = [x.split('_')[-1].lower() for x in l1_cols]
    l1_ohe = [1 if x == l1 else 0 for x in l1_cols]  # ohe value
    # if unseen value is entered, add it to 'other' column
    if sum(l1_ohe) == 0:
        l1_ohe[-1] = 1

    l2 = num_feats[1]  # extract l2 value from input
    l2_cols = pd.Series(model_cols)
    # extract l2 values in training data
    l2_cols = list(l2_cols[l2_cols.str.startswith('l2')].values)
    # take off prefixes and standardize strings
    l2_cols = [x.split('_')[-1].lower() for x in l2_cols]
    l2_ohe = [1 if x == l2 else 0 for x in l2_cols]  # ohe value
    # if unseen value is entered, add it to 'other' column
    if sum(l2_ohe) == 0:
        l2_ohe[-1] = 1

    prop = num_feats[2]  # extract property type value from input
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

    op = num_feats[3]  # extract operation type value from input
    op_cols = pd.Series(model_cols)
    # extract operation type values in training data
    op_cols = list(op_cols[op_cols.str.startswith('operation_type')].values)
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

    return render_template('index.html', prediction_text='House price should be ${}'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
