import requests

url = 'http://127.0.0.1:5000/'

body = {"id": "[3,2,1000,1500,5,-38,-58,'argentina','buenos aires','house','for rent']"}
#{
#    "bedrooms":3,
#    "bathrooms":2, 
#    "surface_total":1000, 
#    "surface_covered":1500, 
#    "rooms":5, 
#    "lat":-38, 
#    "lon":-58, 
#    "l1":"argentina", 
#    "l2":"buenos aires", 
#    "property_type":"house", 
#    "operation_type":"for rent"
#}

response = requests.post(url, data=body)

print(response.json())