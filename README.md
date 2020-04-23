# Vendor-Raw Material-Prediction for restauarnts
This is a project that can help to track the vendor how much raw material is needed to be supplied to a particular restaurant on a weekly basis.

The restaurant.csv is the file from which all the indicidual csv are created. 
The dataset_creation.py file is the code from which the dataset is created for individual dishes in a particular range.
Random numbers, restaurants and weekdays are generated in this particular file.Just check the column number from the restauarnt csv about which dish is being accessed.

The default_restaurant csv for 3 restaurants are the csv that contains the quantity of raw materials like onion,tomato etc. for creation of a particular dish . This varies from restaurant to restauarnt to all restaurants have different csv files for that.

The final_pred.py conatins the final model that is the decision tree for getting the weekly prediction of how many dishes does a particular restaurant sell. So there are 10 dishes for one restaurant and there are 3 restauarnts . So there is a different model created for each dish of every restauarnt. After that the number dishes are multiplied by the quantity of raw materials required for a partiuclar dish. So the raw materials of all dishes are added up in the end to get the final count .

The vendor contains the final count of all the raw materials of the dishes.

The default_vendor_csv is the quantity of how many tomatos ar  make up for 1 kilogram to get specific count.
6 tomatoes=1 kg
6 cheese slices = 1 cheese packed
12 slices =1 bread packet

The quantity of freshly needed things such as vegetables,chicken etc. are calculated on daily baisis wheras all the other things are calculated on weekly basis.

DISHES THAT ARE CONSIDERED ARE:
Pizza,Burger,Non Veg Thali,Veg Thali,Dosa,Sandwich,Pav Bhaji,Misal,idli,kichdi

Raw_materials that are considered are :
Tomato,onion,capsicum,bread,dough,chicken,cheese,corn,rava,sabudana,masala,vegetables,dal,flour,rice,papad,butter

You can connect to me on my LinkedIn too
profile URL:https://www.linkedin.com/in/saurabhzingade/
