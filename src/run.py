###################
# IMPORT PACKAGES #
###################

import pandas as pd
import pickle
import sys

from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

####################
# CREATE FUNCTIONS #
####################


def train_model():

    ### LOAD DATA ###
    df = pd.read_csv('../data/argentina_final.csv')
    df.drop(['Unnamed: 0', 'price', 'start_date',
            'end_date', 'created_on'], axis=1, inplace=True)

    ### SPLIT DATA ###
    x = df.drop(['price'], axis=1)
    y = df.price

    ### FIT MODEL ###
    print("fitting model...")
    rfr_final = RandomForestRegressor(n_jobs=-1)
    rfr_final.fit(x, y)

    ### PICKLE MODEL ###
    filename = 'models/rfr_model.pkl'
    pickle.dump(rfr_final, open(filename, 'wb'))

    ### SAVE COLUMNS MODEL USES ###
    model_cols = list(x.columns)
    filename2 = 'models/rfr_model_columns.pkl'
    pickle.dump(model_cols, open(filename2, 'wb'))

    return "Model Trained and Saved"


def predict(data):

    print('loading data...')
    ### LOAD DATA ###
    X_test = pd.read_csv(data)

    ### LOAD MODEL ###
    filename = 'models/rfr_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))

    print('selecting columns...')
    ### LOAD COLUMNS ###
    filename2 = 'models/rfr_model_columns.pkl'
    loaded_cols = pickle.load(open(filename2, 'rb'))
    #X_test = X_test[loaded_cols]
    X_test = X_test[X_test.columns[X_test.columns.isin(loaded_cols)]]

    print('making predictions...')
    ### MAKE PREDICTIONS ###
    predictions = loaded_model.predict(X_test)
    preds = pd.DataFrame()
    preds['predictions'] = predictions
    preds.to_csv('predictions/predictions.csv')
    return print('predictions saved!')


if __name__ == "__main__":

    train_model()

# If user wants to predict a csv of data, enter csv location as second command line argument.
    if sys.argv[1] == 'True':
        predict(sys.argv[1])
