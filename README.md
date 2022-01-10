# house-price-predict

## Data

The data for this project was found from Kaggle:
- https://www.kaggle.com/msorondo/argentina-venta-de-propiedades
  - 908MB
  - 2 million rows
  - 25 columns
 
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

You should be able to see the predicted house price in terminal after submitting or running your request.py file.
