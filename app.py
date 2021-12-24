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
    num_feats = list(input_features)[:7]
    cat_feats = list(input_features)[7:]
    
    ##############################
    # Handle Categorical Columns #
    ##############################
    
    cat_feats = [x.lower() for x in cat_feats]
    
    l1 = num_feats[0] #extract l1 value from input
    l1_cols = pd.Series(model_cols)
    l1_cols = list(l1_cols[l1_cols.str.startswith('l1')].values) #extract l1 values in training data
    l1_cols = [x.split('_')[-1].lower() for x in l1_cols] #take off prefixes and standardize strings
    l1_ohe = [1 if x==l1 else 0 for x in l1_cols] #ohe value
    # if unseen value is entered, add it to 'other' column
    if sum(l1_ohe) == 0:
        l1_ohe[-1] = 1
    
    l2 = num_feats[1] #extract l2 value from input
    l2_cols = pd.Series(model_cols)
    l2_cols = list(l2_cols[l2_cols.str.startswith('l2')].values) #extract l2 values in training data
    l2_cols = [x.split('_')[-1].lower() for x in l2_cols] #take off prefixes and standardize strings
    l2_ohe = [1 if x==l2 else 0 for x in l2_cols] #ohe value
    # if unseen value is entered, add it to 'other' column
    if sum(l2_ohe) == 0:
        l2_ohe[-1] = 1
        
    prop = num_feats[2] #extract property type value from input
    prop_cols = pd.Series(model_cols)
    prop_cols = list(prop_cols[prop_cols.str.startswith('property_type')].values) #extract property type values in training data
    prop_cols = [x.split('_')[-1].lower() for x in prop_cols] #take off prefixes and standardize strings
    prop_ohe = [1 if x==prop else 0 for x in prop_cols] #ohe value
    # if unseen value is entered, add it to 'other' column
    if sum(prop_ohe) == 0:
        prop_ohe[6] = 1
        
    op = num_feats[3] #extract operation type value from input
    op_cols = pd.Series(model_cols)
    op_cols = list(op_cols[op_cols.str.startswith('operation_type')].values) #extract operation type values in training data
    op_cols = [x.split('_')[-1].lower() for x in op_cols] #take off prefixes and standardize strings
    op_ohe = [1 if x==op else 0 for x in op_cols] #ohe value
    
    ###############################
    # Finalize Features & Predict #
    ###############################
    
    transformed_categoricals = l1_cols + l2_cols + prop_cols + op_cols
    final_features = num_feats + transformed_categoricals
    
    prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    return


if __name__ == "__main__":
    app.run(debug=True)

