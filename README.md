# house-price-predict
The overall project workflow can be seen in the diagram "Project Flow". This diagram shows the steps taken for this project with general descriptions of what those steps consisted of.

## Data

The data for this project was found from Kaggle:
- https://www.kaggle.com/msorondo/argentina-venta-de-propiedades
  - 908MB
  - 2 million rows
  - 25 columns
  - data was collected from https://www.properati.com.ar which is a website used to find properties for sale, rent and ventures.

### Data Cleaning & Preprocessing
To start, I needed to handle some data type conversions. I explored null data values for the columns and dropped any columns that were missing 50% or more data. Then, I imputed the rest of the null values for numerical columns with the mean and the mode for the categorical columns. Next, I translated the categorical data into English since most of the data was in Spanish. I then had to standardize the currencies to all be of the same currency by converting them all to US dollars. For the price column, I transformed the values to all be the cost of the house monthly.

### Data Visualizations & Aggregations
For exploratory data analysis, I evaluated some descriptive statistics of the data along with a correlation matrix and heatmap. I then observed the distributions of many of the numerical columns along with creating some bivariate analysis graphs to see the relationships between specific features. This gave me insight to remove some extra columns.

### Final Model Data Input
The columns used and the structure for inputting data for this model, is as follows:
Column | Type | Description
------ | ---- | -----------
bedrooms | int | number of bedrooms
bathrooms | int | number of bathrooms, 
surface_total | int | total area in m^2, 
surface_covered | int | total covered in m^2, 
rooms | int | number of rooms, 
lat | int | latitude, 
lon | int | longitude, 
l1 | str | Administrative level 1, usually country, 
l2 | str | Administrative level 2, usually province, 
property_type | str | type of property, ex. House, Apartment, Penthouse, Office, etc. 
operation_type | str | Type of offer, ex. For Rent, For Sale, or For Sublease

After all the data cleaning and feature selection, the size of the data had been substantially reduced. The combinations and aggregations done in this step allowed me to remove many column that no longer offered any value.

## Modeling
Once I completed the necessary data cleaning and preprocessing, the data is now ready to be fed into the models. Before feeding the data to the models, however, one-hot encoding is done to the necessary categorical columns. I used scikit learn's mean and median dummy regressor models as a baseline for model performance and picked a handful of models to try out for my final model. I used Root Mean Squared Error as my evaluation metric in order to test the performance of my model. I also used Mean Absolute Error and R2 as supporting metrics. In the end, scikit learn's Random Forest Regressor performed the best and therefore was chosen as my final model.

After selecting the final model, I carried out hyperparameter tuning using scikit learn's gridsearchCV and evaluated performance with K-fold cross validation. After that analysis, I saved the best performing model along with the names of the columns the model uses and created a .py where one can recreate the model by training it yourself, then saving it to use for deployment. This file is run.py.

## Deployment
I used Flask's Restful API in order to deploy this model. In order to do this, I load the saved model itself and the model columns. The model columns are necessary for the categorical features that will be input by the user. To build this API, I first create a "Predict" class and more importanty, create the post function that will take in the user's input that they wish to get a price prediction for. On this server side, it will take in the data for prediction and transform the data into the necessary format for the model to use it. This makes it easier for the user by not making them input an excessive number of features.

On the client side, the user will simply have to first run the api by running app.py within one terminal, then they can choose to either modify the request.py with their own data and run that in a separate terminal to get the predictions, or should they choose to open up the index.html file, they can also input the values in question there and the prediction result will show up in the terminal running the api in both cases. The user will not have to worry about categorical values such as the country, province, or property type as the server side will handle all of that.

Once the data has been transformed properly, it is fed into the model where it will output a house price prediction. That value is then displayed for the user in the running terminal.

## Project Structure

This project has four major parts :

1. run.py - This will create, train, and save the model based on the training data.
2. app.py - This contains Flask APIs that receives house information and computes the predicted value based on this model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. index.html - This is an interface you can so that you can input data to sent to the API

## Running the project

Ensure that you are in the project home directory. Create the machine learning model by running below command

`python run.py`

This would create a serialized version of our model into a file model.pkl

Run app.py using below command to start Flask API

`python app.py`

Edit data in request.py with the information on the house that you'd like to predict the price of.
Open up another terminal and run the below command to get the prediction

`python request.py`

Or if you would like to enter the data manually, you can open the file index.html and enter house information there. 

Enter valid numerical values for the first 7 input boxes, them enter the name of the country, province, and property type. For the last value, enter whether the house in question is being put up For Rent, For Sale, or For Sublease. Next, submit the response.

You will be able to see the predicted house price in terminal after submitting or running your request.py file.
