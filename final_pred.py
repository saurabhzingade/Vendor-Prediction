#importing packages to read the csv files


import pandas as pd
import numpy as np
import math

#reading the csv files

default_quantity_data_A=pd.read_csv('DEFAULT_A.csv')
default_quantity_data_B=pd.read_csv('DEFAULT_B.csv')
default_quantity_data_C=pd.read_csv('DEFAULT_C.csv')
pizza_data=pd.read_csv('pizza.csv')
burger_data=pd.read_csv('Burger.csv')
non_veg_thali_data=pd.read_csv('Non_veg_thali.csv')
veg_thali_data=pd.read_csv('veg_thali.csv')
dosa_data=pd.read_csv('Dosa.csv')
sandwich_data=pd.read_csv('Sandwich.csv')
pav_bhaji_data=pd.read_csv('Pav_bhaji.csv')
misal_data=pd.read_csv('Misal.csv')
idli_data=pd.read_csv('idli.csv')
kichdi_data=pd.read_csv('kichdi.csv')

#importing the encoding library to convert string values to numeric values

from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()

#importing the decision tree model

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)

#importing the train test split
from sklearn.model_selection import train_test_split

#analysis for restaurant A

#pizza analysis for restaurant A

group_data_pizza_A=pizza_data.groupby("Restaurant")
pizza_for_rest_A=group_data_pizza_A.get_group('A')
weekday_input_pizza_A=pizza_for_rest_A[['Weekday']]
weekday_input_pizza_A=weekday_input_pizza_A.apply(labelencoder_X.fit_transform)	
pizza_out=pizza_for_rest_A[['Pizza']]
X_train_pizza_A,X_test_pizza_A,Y_train_pizza_A,Y_test_pizza_A = train_test_split(weekday_input_pizza_A,pizza_out,test_size=0.2,random_state=0)
regressor.fit(X_train_pizza_A, Y_train_pizza_A)


''' 
The label enocoding for pizza is as follows
Monday-1
Tuesday-5
Wednesday-6
Thursday-4
Friday-0
Saturday-2
Sunday-3
'''
pizza_pred_A_Monday = int(regressor.predict([[1]]))
pizza_pred_A_Tuesday = int(regressor.predict([[5]]))
pizza_pred_A_Wednesday = int(regressor.predict([[6]]))
pizza_pred_A_Thursday = int(regressor.predict([[4]]))
pizza_pred_A_Friday = int(regressor.predict([[0]]))
pizza_pred_A_Saturday = int(regressor.predict([[1]]))
pizza_pred_A_Sunday = int(regressor.predict([[3]]))

#reading the default csv file

weekly_pizza_pred_A= pizza_pred_A_Monday+pizza_pred_A_Tuesday+pizza_pred_A_Wednesday+pizza_pred_A_Thursday+pizza_pred_A_Friday+pizza_pred_A_Saturday+pizza_pred_A_Sunday
default_quantity_data_A_pizza=default_quantity_data_A.iloc[0,].values
pizza_A_list=default_quantity_data_A_pizza.tolist()
pizza_A_list=pizza_A_list[1:]

pizza_A_tomato=pizza_A_list[0]*weekly_pizza_pred_A
pizza_A_onion=pizza_A_list[1]*weekly_pizza_pred_A
pizza_A_capsicum=pizza_A_list[2]*weekly_pizza_pred_A
pizza_A_bread=pizza_A_list[3]*weekly_pizza_pred_A
pizza_A_dough=pizza_A_list[4]*weekly_pizza_pred_A
pizza_A_chicken=pizza_A_list[5]*weekly_pizza_pred_A
pizza_A_cheese=pizza_A_list[6]*weekly_pizza_pred_A
pizza_A_corn=pizza_A_list[7]*weekly_pizza_pred_A
pizza_A_rava=pizza_A_list[8]*weekly_pizza_pred_A
pizza_A_sabudana=pizza_A_list[9]*weekly_pizza_pred_A	
pizza_A_masala=pizza_A_list[10]*weekly_pizza_pred_A	
pizza_A_vegetables=pizza_A_list[11]*weekly_pizza_pred_A	
pizza_A_dal=pizza_A_list[12]*weekly_pizza_pred_A
pizza_A_flour=pizza_A_list[13]*weekly_pizza_pred_A
pizza_A_rice=pizza_A_list[14]*weekly_pizza_pred_A
pizza_A_papad=pizza_A_list[15]*weekly_pizza_pred_A
pizza_A_butter=pizza_A_list[16]*weekly_pizza_pred_A	

#Burger Analysis for restaurant A
	
group_data_burger_A=burger_data.groupby("Restaurant")
burger_for_rest_A=group_data_burger_A.get_group('A')
weekday_input_burger_A=burger_for_rest_A[['Weekday']]
weekday_input_burger_A=weekday_input_burger_A.apply(labelencoder_X.fit_transform)	
burger_out=burger_for_rest_A[['Burger']]
X_train_burger_A,X_test_burger_A,Y_train_burger_A,Y_test_burger_A = train_test_split(weekday_input_burger_A,burger_out,test_size=0.2,random_state=0)
regressor.fit(X_train_burger_A, Y_train_burger_A)	

burger_sum=0
for i in range(0,7):
	burger_predict_sum=regressor.predict([[i]])
	burger_sum=burger_sum+burger_predict_sum
burger_sum=int(burger_sum)	

default_quantity_data_A_burger=default_quantity_data_A.iloc[1,].values
burger_A_list=default_quantity_data_A_burger.tolist()
burger_A_list=burger_A_list[1:]



burger_A_tomato=burger_A_list[0]*burger_sum
burger_A_onion=burger_A_list[1]*burger_sum
burger_A_capsicum=burger_A_list[2]*burger_sum
burger_A_bread=burger_A_list[3]*burger_sum
burger_A_dough=burger_A_list[4]*burger_sum
burger_A_chicken=burger_A_list[5]*burger_sum
burger_A_cheese=burger_A_list[6]*burger_sum
burger_A_corn=burger_A_list[7]*burger_sum
burger_A_rava=burger_A_list[8]*burger_sum
burger_A_sabudana=burger_A_list[9]*burger_sum	
burger_A_masala=burger_A_list[10]*burger_sum
burger_A_vegetables=burger_A_list[11]*burger_sum
burger_A_dal=burger_A_list[12]*burger_sum
burger_A_flour=burger_A_list[13]*burger_sum
burger_A_rice=burger_A_list[14]*burger_sum
burger_A_papad=burger_A_list[15]*burger_sum
burger_A_butter=burger_A_list[16]*burger_sum

#Non veg thali prediction for resturant A

group_data_nonveg_A=non_veg_thali_data.groupby("Restaurant")
nonveg_for_rest_A=group_data_nonveg_A.get_group('A')
weekday_input_nonveg_A=nonveg_for_rest_A[['Weekday']]
weekday_input_nonveg_A=weekday_input_nonveg_A.apply(labelencoder_X.fit_transform)	
nonveg_out=nonveg_for_rest_A[['Non Veg Thali']]
X_train_nonveg_A,X_test_nonveg_A,Y_train_nonveg_A,Y_test_nonveg_A = train_test_split(weekday_input_nonveg_A,nonveg_out,test_size=0.2,random_state=0)
regressor.fit(X_train_nonveg_A, Y_train_nonveg_A)	

nonveg_sum=0
for i in range(0,7):
	nonveg_predict_sum=regressor.predict([[i]])
	nonveg_sum=nonveg_sum+nonveg_predict_sum
nonveg_sum=int(nonveg_sum)	

default_quantity_data_A_nonveg=default_quantity_data_A.iloc[2,].values
nonveg_A_list=default_quantity_data_A_nonveg.tolist()
nonveg_A_list=nonveg_A_list[1:]

nonveg_A_tomato=nonveg_A_list[0]*nonveg_sum
nonveg_A_onion=nonveg_A_list[1]*nonveg_sum
nonveg_A_capsicum=nonveg_A_list[2]*nonveg_sum
nonveg_A_bread=nonveg_A_list[3]*nonveg_sum
nonveg_A_dough=nonveg_A_list[4]*nonveg_sum
nonveg_A_chicken=nonveg_A_list[5]*nonveg_sum
nonveg_A_cheese=nonveg_A_list[6]*nonveg_sum
nonveg_A_corn=nonveg_A_list[7]*nonveg_sum
nonveg_A_rava=nonveg_A_list[8]*nonveg_sum
nonveg_A_sabudana=nonveg_A_list[9]*nonveg_sum	
nonveg_A_masala=nonveg_A_list[10]*nonveg_sum
nonveg_A_vegetables=nonveg_A_list[11]*nonveg_sum
nonveg_A_dal=nonveg_A_list[12]*nonveg_sum
nonveg_A_flour=nonveg_A_list[13]*nonveg_sum
nonveg_A_rice=nonveg_A_list[14]*nonveg_sum
nonveg_A_papad=nonveg_A_list[15]*nonveg_sum
nonveg_A_butter=nonveg_A_list[16]*nonveg_sum


#Veg Thali analysis for restauarant A

group_data_veg_A=veg_thali_data.groupby("Restaurant")
veg_for_rest_A=group_data_veg_A.get_group('A')
weekday_input_veg_A=veg_for_rest_A[['Weekday']]
weekday_input_veg_A=weekday_input_veg_A.apply(labelencoder_X.fit_transform)	
veg_out=veg_for_rest_A[['Veg Thali']]
X_train_veg_A,X_test_veg_A,Y_train_veg_A,Y_test_veg_A = train_test_split(weekday_input_veg_A,veg_out,test_size=0.2,random_state=0)
regressor.fit(X_train_veg_A, Y_train_veg_A)
	
veg_sum=0
for i in range(0,7):
	veg_predict_sum=regressor.predict([[i]])
	veg_sum=veg_sum+veg_predict_sum
veg_sum=int(veg_sum)	

default_quantity_data_A_veg=default_quantity_data_A.iloc[3,].values
veg_A_list=default_quantity_data_A_veg.tolist()
veg_A_list=veg_A_list[1:]

veg_A_tomato=veg_A_list[0]*veg_sum
veg_A_onion=veg_A_list[1]*veg_sum
veg_A_capsicum=veg_A_list[2]*veg_sum
veg_A_bread=veg_A_list[3]*veg_sum
veg_A_dough=veg_A_list[4]*veg_sum
veg_A_chicken=veg_A_list[5]*veg_sum
veg_A_cheese=veg_A_list[6]*veg_sum
veg_A_corn=veg_A_list[7]*veg_sum
veg_A_rava=veg_A_list[8]*veg_sum
veg_A_sabudana=veg_A_list[9]*veg_sum	
veg_A_masala=veg_A_list[10]*veg_sum
veg_A_vegetables=veg_A_list[11]*veg_sum
veg_A_dal=veg_A_list[12]*veg_sum
veg_A_flour=veg_A_list[13]*veg_sum
veg_A_rice=veg_A_list[14]*veg_sum
veg_A_papad=veg_A_list[15]*veg_sum
veg_A_butter=veg_A_list[16]*veg_sum


#Dosa analysis for restauarant A

group_data_dosa_A=dosa_data.groupby("Restaurant")
dosa_for_rest_A=group_data_dosa_A.get_group('A')
weekday_input_dosa_A=dosa_for_rest_A[['Weekday']]
weekday_input_dosa_A=weekday_input_dosa_A.apply(labelencoder_X.fit_transform)	
dosa_out=dosa_for_rest_A[['Dosa']]
X_train_dosa_A,X_test_dosa_A,Y_train_dosa_A,Y_test_dosa_A = train_test_split(weekday_input_dosa_A,dosa_out,test_size=0.2,random_state=0)
regressor.fit(X_train_dosa_A, Y_train_dosa_A)

dosa_sum=0
for i in range(0,7):
	dosa_predict_sum=regressor.predict([[i]])
	dosa_sum=dosa_sum+dosa_predict_sum
dosa_sum=int(dosa_sum)
	
default_quantity_data_A_dosa=default_quantity_data_A.iloc[4,].values
dosa_A_list=default_quantity_data_A_dosa.tolist()
dosa_A_list=dosa_A_list[1:]	

dosa_A_tomato=dosa_A_list[0]*dosa_sum
dosa_A_onion=dosa_A_list[1]*dosa_sum
dosa_A_capsicum=dosa_A_list[2]*dosa_sum
dosa_A_bread=dosa_A_list[3]*dosa_sum
dosa_A_dough=dosa_A_list[4]*dosa_sum
dosa_A_chicken=dosa_A_list[5]*dosa_sum
dosa_A_cheese=dosa_A_list[6]*dosa_sum
dosa_A_corn=dosa_A_list[7]*dosa_sum
dosa_A_rava=dosa_A_list[8]*dosa_sum
dosa_A_sabudana=dosa_A_list[9]*dosa_sum	
dosa_A_masala=dosa_A_list[10]*dosa_sum
dosa_A_vegetables=dosa_A_list[11]*dosa_sum
dosa_A_dal=dosa_A_list[12]*dosa_sum
dosa_A_flour=dosa_A_list[13]*dosa_sum
dosa_A_rice=dosa_A_list[14]*dosa_sum
dosa_A_papad=dosa_A_list[15]*dosa_sum
dosa_A_butter=dosa_A_list[16]*dosa_sum

#Sandwich analysis for restaurant A

group_data_sandwich_A=sandwich_data.groupby("Restaurant")
sandwich_for_rest_A=group_data_sandwich_A.get_group('A')
weekday_input_sandwich_A=sandwich_for_rest_A[['Weekday']]
weekday_input_sandwich_A=weekday_input_sandwich_A.apply(labelencoder_X.fit_transform)	
sandwich_out=sandwich_for_rest_A[['Sandwich']]	
X_train_sandwich_A,X_test_sandwich_A,Y_train_sandwich_A,Y_test_sandwich_A = train_test_split(weekday_input_sandwich_A,sandwich_out,test_size=0.2,random_state=0)
regressor.fit(X_train_sandwich_A, Y_train_sandwich_A)
	
sandwich_sum=0	
for i in range(0,7):
	sandwich_predict_sum=regressor.predict([[i]])
	sandwich_sum=sandwich_sum+sandwich_predict_sum
sandwich_sum=int(sandwich_sum)

default_quantity_data_A_sandwich=default_quantity_data_A.iloc[5,].values
sandwich_A_list=default_quantity_data_A_sandwich.tolist()
sandwich_A_list=sandwich_A_list[1:]	

sandwich_A_tomato=sandwich_A_list[0]*sandwich_sum
sandwich_A_onion=sandwich_A_list[1]*sandwich_sum
sandwich_A_capsicum=sandwich_A_list[2]*sandwich_sum
sandwich_A_bread=sandwich_A_list[3]*sandwich_sum
sandwich_A_dough=sandwich_A_list[4]*sandwich_sum
sandwich_A_chicken=sandwich_A_list[5]*sandwich_sum
sandwich_A_cheese=sandwich_A_list[6]*sandwich_sum
sandwich_A_corn=sandwich_A_list[7]*sandwich_sum
sandwich_A_rava=sandwich_A_list[8]*sandwich_sum
sandwich_A_sabudana=sandwich_A_list[9]*sandwich_sum	
sandwich_A_masala=sandwich_A_list[10]*sandwich_sum
sandwich_A_vegetables=sandwich_A_list[11]*sandwich_sum
sandwich_A_dal=sandwich_A_list[12]*sandwich_sum
sandwich_A_flour=sandwich_A_list[13]*sandwich_sum
sandwich_A_rice=sandwich_A_list[14]*sandwich_sum
sandwich_A_papad=sandwich_A_list[15]*sandwich_sum
sandwich_A_butter=sandwich_A_list[16]*sandwich_sum

#Pav Bhaji analysis for restaurant A

group_data_pavbhaji_A=pav_bhaji_data.groupby("Restaurant")
pavbhaji_for_rest_A=group_data_pavbhaji_A.get_group('A')
weekday_input_pavbhaji_A=pavbhaji_for_rest_A[['Weekday']]
weekday_input_pavbhaji_A=weekday_input_pavbhaji_A.apply(labelencoder_X.fit_transform)	
pavbhaji_out=pavbhaji_for_rest_A[['Pav Bhaji']]	

X_train_pavbhaji_A,X_test_pavbhaji_A,Y_train_pavbhaji_A,Y_test_pavbhaji_A = train_test_split(weekday_input_pavbhaji_A,pavbhaji_out,test_size=0.2,random_state=0)
regressor.fit(X_train_pavbhaji_A, Y_train_pavbhaji_A)

pavbhaji_sum=0	
for i in range(0,7):
	pavbhaji_predict_sum=regressor.predict([[i]])
	pavbhaji_sum=pavbhaji_sum+pavbhaji_predict_sum
pavbhaji_sum=int(pavbhaji_sum)

default_quantity_data_A_pavbhaji=default_quantity_data_A.iloc[6,].values
pavbhaji_A_list=default_quantity_data_A_pavbhaji.tolist()
pavbhaji_A_list=pavbhaji_A_list[1:]	


pavbhaji_A_tomato=pavbhaji_A_list[0]*pavbhaji_sum
pavbhaji_A_onion=pavbhaji_A_list[1]*pavbhaji_sum
pavbhaji_A_capsicum=pavbhaji_A_list[2]*pavbhaji_sum
pavbhaji_A_bread=pavbhaji_A_list[3]*pavbhaji_sum
pavbhaji_A_dough=pavbhaji_A_list[4]*pavbhaji_sum
pavbhaji_A_chicken=pavbhaji_A_list[5]*pavbhaji_sum
pavbhaji_A_cheese=pavbhaji_A_list[6]*pavbhaji_sum
pavbhaji_A_corn=pavbhaji_A_list[7]*pavbhaji_sum
pavbhaji_A_rava=pavbhaji_A_list[8]*pavbhaji_sum
pavbhaji_A_sabudana=pavbhaji_A_list[9]*pavbhaji_sum	
pavbhaji_A_masala=pavbhaji_A_list[10]*pavbhaji_sum
pavbhaji_A_vegetables=pavbhaji_A_list[11]*pavbhaji_sum
pavbhaji_A_dal=pavbhaji_A_list[12]*pavbhaji_sum
pavbhaji_A_flour=pavbhaji_A_list[13]*pavbhaji_sum
pavbhaji_A_rice=pavbhaji_A_list[14]*pavbhaji_sum
pavbhaji_A_papad=pavbhaji_A_list[15]*pavbhaji_sum
pavbhaji_A_butter=pavbhaji_A_list[16]*pavbhaji_sum


#Misal analysis for restaurant A

group_data_misal_A=misal_data.groupby("Restaurant")
misal_for_rest_A=group_data_misal_A.get_group('A')
weekday_input_misal_A=misal_for_rest_A[['Weekday']]
weekday_input_misal_A=weekday_input_misal_A.apply(labelencoder_X.fit_transform)	
misal_out=misal_for_rest_A[['Misal']]	

X_train_misal_A,X_test_misal_A,Y_train_misal_A,Y_test_misal_A = train_test_split(weekday_input_misal_A,misal_out,test_size=0.2,random_state=0)
regressor.fit(X_train_misal_A, Y_train_misal_A)

misal_sum=0
for i in range(0,7):
	misal_predict_sum=regressor.predict([[i]])
	misal_sum=misal_sum+misal_predict_sum
misal_sum=int(misal_sum)

default_quantity_data_A_misal=default_quantity_data_A.iloc[7,].values
misal_A_list=default_quantity_data_A_misal.tolist()
misal_A_list=misal_A_list[1:]	

misal_A_tomato=misal_A_list[0]*misal_sum
misal_A_onion=misal_A_list[1]*misal_sum
misal_A_capsicum=misal_A_list[2]*misal_sum
misal_A_bread=misal_A_list[3]*misal_sum
misal_A_dough=misal_A_list[4]*misal_sum
misal_A_chicken=misal_A_list[5]*misal_sum
misal_A_cheese=misal_A_list[6]*misal_sum
misal_A_corn=misal_A_list[7]*misal_sum
misal_A_rava=misal_A_list[8]*misal_sum
misal_A_sabudana=misal_A_list[9]*misal_sum	
misal_A_masala=misal_A_list[10]*misal_sum
misal_A_vegetables=misal_A_list[11]*misal_sum
misal_A_dal=misal_A_list[12]*misal_sum
misal_A_flour=misal_A_list[13]*misal_sum
misal_A_rice=misal_A_list[14]*misal_sum
misal_A_papad=misal_A_list[15]*misal_sum
misal_A_butter=misal_A_list[16]*misal_sum


#idli analysis for restauarant A
group_data_idli_A=idli_data.groupby("Restaurant")
idli_for_rest_A=group_data_idli_A.get_group('A')
weekday_input_idli_A=idli_for_rest_A[['Weekday']]
weekday_input_idli_A=weekday_input_idli_A.apply(labelencoder_X.fit_transform)	
idli_out=idli_for_rest_A[['idli']]	

X_train_idli_A,X_test_idli_A,Y_train_idli_A,Y_test_idli_A = train_test_split(weekday_input_idli_A,idli_out,test_size=0.2,random_state=0)
regressor.fit(X_train_idli_A, Y_train_idli_A)

idli_sum=0

for i in range(0,7):
	idli_predict_sum=regressor.predict([[i]])
	idli_sum=idli_sum+idli_predict_sum
idli_sum=int(idli_sum)

default_quantity_data_A_idli=default_quantity_data_A.iloc[8,].values
idli_A_list=default_quantity_data_A_idli.tolist()
idli_A_list=idli_A_list[1:]	

idli_A_tomato=idli_A_list[0]*idli_sum
idli_A_onion=idli_A_list[1]*idli_sum
idli_A_capsicum=idli_A_list[2]*idli_sum
idli_A_bread=idli_A_list[3]*idli_sum
idli_A_dough=idli_A_list[4]*idli_sum
idli_A_chicken=idli_A_list[5]*idli_sum
idli_A_cheese=idli_A_list[6]*idli_sum
idli_A_corn=idli_A_list[7]*idli_sum
idli_A_rava=idli_A_list[8]*idli_sum
idli_A_sabudana=idli_A_list[9]*idli_sum	
idli_A_masala=idli_A_list[10]*idli_sum
idli_A_vegetables=idli_A_list[11]*idli_sum
idli_A_dal=idli_A_list[12]*idli_sum
idli_A_flour=idli_A_list[13]*idli_sum
idli_A_rice=idli_A_list[14]*idli_sum
idli_A_papad=idli_A_list[15]*idli_sum
idli_A_butter=idli_A_list[16]*idli_sum	
	
#kichdi analysis for restaurant A
group_data_kichdi_A=kichdi_data.groupby("Restaurant")
kichdi_for_rest_A=group_data_kichdi_A.get_group('A')
weekday_input_kichdi_A=kichdi_for_rest_A[['Weekday']]
weekday_input_kichdi_A=weekday_input_kichdi_A.apply(labelencoder_X.fit_transform)	
kichdi_out=kichdi_for_rest_A[['kichdi']]	

X_train_kichdi_A,X_test_kichdi_A,Y_train_kichdi_A,Y_test_kichdi_A = train_test_split(weekday_input_kichdi_A,kichdi_out,test_size=0.2,random_state=0)
regressor.fit(X_train_kichdi_A, Y_train_kichdi_A)

kichdi_sum=0
for i in range(0,7):
	kichdi_predict_sum=regressor.predict([[i]])
	kichdi_sum=kichdi_sum+kichdi_predict_sum
kichdi_sum=int(kichdi_sum)

default_quantity_data_A_kichdi=default_quantity_data_A.iloc[9,].values
kichdi_A_list=default_quantity_data_A_kichdi.tolist()
kichdi_A_list=kichdi_A_list[1:]	

kichdi_A_tomato=kichdi_A_list[0]*kichdi_sum
kichdi_A_onion=kichdi_A_list[1]*kichdi_sum
kichdi_A_capsicum=kichdi_A_list[2]*kichdi_sum
kichdi_A_bread=kichdi_A_list[3]*kichdi_sum
kichdi_A_dough=kichdi_A_list[4]*kichdi_sum
kichdi_A_chicken=kichdi_A_list[5]*kichdi_sum
kichdi_A_cheese=kichdi_A_list[6]*kichdi_sum
kichdi_A_corn=kichdi_A_list[7]*kichdi_sum
kichdi_A_rava=kichdi_A_list[8]*kichdi_sum
kichdi_A_sabudana=kichdi_A_list[9]*kichdi_sum	
kichdi_A_masala=kichdi_A_list[10]*kichdi_sum
kichdi_A_vegetables=kichdi_A_list[11]*kichdi_sum
kichdi_A_dal=kichdi_A_list[12]*kichdi_sum
kichdi_A_flour=kichdi_A_list[13]*kichdi_sum
kichdi_A_rice=kichdi_A_list[14]*kichdi_sum
kichdi_A_papad=kichdi_A_list[15]*kichdi_sum
kichdi_A_butter=kichdi_A_list[16]*kichdi_sum	

total_tom_A=[]
total_tom_A.append(pizza_A_tomato)
total_tom_A.append(burger_A_tomato)
total_tom_A.append(nonveg_A_tomato)
total_tom_A.append(veg_A_tomato)
total_tom_A.append(dosa_A_tomato)
total_tom_A.append(sandwich_A_tomato)
total_tom_A.append(pavbhaji_A_tomato)
total_tom_A.append(misal_A_tomato)
total_tom_A.append(idli_A_tomato)
total_tom_A.append(kichdi_A_tomato)
total_tom_A=sum(total_tom_A)

total_onion_A=[]
total_onion_A.append(pizza_A_onion)
total_onion_A.append(burger_A_onion)
total_onion_A.append(nonveg_A_onion)
total_onion_A.append(veg_A_onion)
total_onion_A.append(dosa_A_onion)
total_onion_A.append(sandwich_A_onion)
total_onion_A.append(pavbhaji_A_onion)
total_onion_A.append(misal_A_onion)
total_onion_A.append(idli_A_onion)
total_onion_A.append(kichdi_A_onion)
total_onion_A=sum(total_onion_A)

total_capsicum_A=[]
total_capsicum_A.append(pizza_A_capsicum)
total_capsicum_A.append(burger_A_capsicum)
total_capsicum_A.append(nonveg_A_capsicum)
total_capsicum_A.append(veg_A_capsicum)
total_capsicum_A.append(dosa_A_capsicum)
total_capsicum_A.append(sandwich_A_capsicum)
total_capsicum_A.append(pavbhaji_A_capsicum)
total_capsicum_A.append(misal_A_capsicum)
total_capsicum_A.append(idli_A_capsicum)
total_capsicum_A.append(kichdi_A_capsicum)
total_capsicum_A=sum(total_capsicum_A)

total_bread_A=[]
total_bread_A.append(pizza_A_bread)
total_bread_A.append(burger_A_bread)
total_bread_A.append(nonveg_A_bread)
total_bread_A.append(veg_A_bread)
total_bread_A.append(dosa_A_bread)
total_bread_A.append(sandwich_A_bread)
total_bread_A.append(pavbhaji_A_bread)
total_bread_A.append(misal_A_bread)
total_bread_A.append(idli_A_bread)
total_bread_A.append(kichdi_A_bread)
total_bread_A=sum(total_bread_A)

total_dough_A=[]
total_dough_A.append(pizza_A_dough)
total_dough_A.append(burger_A_dough)
total_dough_A.append(nonveg_A_dough)
total_dough_A.append(veg_A_dough)
total_dough_A.append(dosa_A_dough)
total_dough_A.append(sandwich_A_dough)
total_dough_A.append(pavbhaji_A_dough)
total_dough_A.append(misal_A_dough)
total_dough_A.append(idli_A_dough)
total_dough_A.append(kichdi_A_dough)
total_dough_A=sum(total_dough_A)

total_chicken_A=[]
total_chicken_A.append(pizza_A_chicken)
total_chicken_A.append(burger_A_chicken)
total_chicken_A.append(nonveg_A_chicken)
total_chicken_A.append(veg_A_chicken)
total_chicken_A.append(dosa_A_chicken)
total_chicken_A.append(sandwich_A_chicken)
total_chicken_A.append(pavbhaji_A_chicken)
total_chicken_A.append(misal_A_chicken)
total_chicken_A.append(idli_A_chicken)
total_chicken_A.append(kichdi_A_chicken)
total_chicken_A=sum(total_chicken_A)

total_cheese_A=[]
total_cheese_A.append(pizza_A_cheese)
total_cheese_A.append(burger_A_cheese)
total_cheese_A.append(nonveg_A_cheese)
total_cheese_A.append(veg_A_cheese)
total_cheese_A.append(dosa_A_cheese)
total_cheese_A.append(sandwich_A_cheese)
total_cheese_A.append(pavbhaji_A_cheese)
total_cheese_A.append(misal_A_cheese)
total_cheese_A.append(idli_A_cheese)
total_cheese_A.append(kichdi_A_cheese)
total_cheese_A=sum(total_cheese_A)


total_corn_A=[]
total_corn_A.append(pizza_A_corn)
total_corn_A.append(burger_A_corn)
total_corn_A.append(nonveg_A_corn)
total_corn_A.append(veg_A_corn)
total_corn_A.append(dosa_A_corn)
total_corn_A.append(sandwich_A_corn)
total_corn_A.append(pavbhaji_A_corn)
total_corn_A.append(misal_A_corn)
total_corn_A.append(idli_A_corn)
total_corn_A.append(kichdi_A_corn)
total_corn_A=sum(total_corn_A)

total_rava_A=[]
total_rava_A.append(pizza_A_rava)
total_rava_A.append(burger_A_rava)
total_rava_A.append(nonveg_A_rava)
total_rava_A.append(veg_A_rava)
total_rava_A.append(dosa_A_rava)
total_rava_A.append(sandwich_A_rava)
total_rava_A.append(pavbhaji_A_rava)
total_rava_A.append(misal_A_rava)
total_rava_A.append(idli_A_rava)
total_rava_A.append(kichdi_A_rava)
total_rava_A=sum(total_rava_A)


total_sabu_A=[]
total_sabu_A.append(pizza_A_sabudana)
total_sabu_A.append(burger_A_sabudana)
total_sabu_A.append(nonveg_A_sabudana)
total_sabu_A.append(veg_A_sabudana)
total_sabu_A.append(dosa_A_sabudana)
total_sabu_A.append(sandwich_A_sabudana)
total_sabu_A.append(pavbhaji_A_sabudana)
total_sabu_A.append(misal_A_sabudana)
total_sabu_A.append(idli_A_sabudana)
total_sabu_A.append(kichdi_A_sabudana)
total_sabu_A=sum(total_sabu_A)

total_masala_A=[]
total_masala_A.append(pizza_A_masala)
total_masala_A.append(burger_A_masala)
total_masala_A.append(nonveg_A_masala)
total_masala_A.append(veg_A_masala)
total_masala_A.append(dosa_A_masala)
total_masala_A.append(sandwich_A_masala)
total_masala_A.append(pavbhaji_A_masala)
total_masala_A.append(misal_A_masala)
total_masala_A.append(idli_A_masala)
total_masala_A.append(kichdi_A_masala)
total_masala_A=sum(total_masala_A)

total_veggies_A=[]
total_veggies_A.append(pizza_A_vegetables)
total_veggies_A.append(burger_A_vegetables)
total_veggies_A.append(nonveg_A_vegetables)
total_veggies_A.append(veg_A_vegetables)
total_veggies_A.append(dosa_A_vegetables)
total_veggies_A.append(sandwich_A_vegetables)
total_veggies_A.append(pavbhaji_A_vegetables)
total_veggies_A.append(misal_A_vegetables)
total_veggies_A.append(idli_A_vegetables)
total_veggies_A.append(kichdi_A_vegetables)
total_veggies_A=sum(total_veggies_A)

total_dal_A=[]
total_dal_A.append(pizza_A_dal)
total_dal_A.append(burger_A_dal)
total_dal_A.append(nonveg_A_dal)
total_dal_A.append(veg_A_dal)
total_dal_A.append(dosa_A_dal)
total_dal_A.append(sandwich_A_dal)
total_dal_A.append(pavbhaji_A_dal)
total_dal_A.append(misal_A_dal)
total_dal_A.append(idli_A_dal)
total_dal_A.append(kichdi_A_dal)
total_dal_A=sum(total_dal_A)

total_flour_A=[]
total_flour_A.append(pizza_A_flour)
total_flour_A.append(burger_A_flour)
total_flour_A.append(nonveg_A_flour)
total_flour_A.append(veg_A_flour)
total_flour_A.append(dosa_A_flour)
total_flour_A.append(sandwich_A_flour)
total_flour_A.append(pavbhaji_A_flour)
total_flour_A.append(misal_A_flour)
total_flour_A.append(idli_A_flour)
total_flour_A.append(kichdi_A_flour)
total_flour_A=sum(total_flour_A)


total_rice_A=[]
total_rice_A.append(pizza_A_rice)
total_rice_A.append(burger_A_rice)
total_rice_A.append(nonveg_A_rice)
total_rice_A.append(veg_A_rice)
total_rice_A.append(dosa_A_rice)
total_rice_A.append(sandwich_A_rice)
total_rice_A.append(pavbhaji_A_rice)
total_rice_A.append(misal_A_rice)
total_rice_A.append(idli_A_rice)
total_rice_A.append(kichdi_A_rice)
total_rice_A=sum(total_rice_A)

total_papad_A=[]
total_papad_A.append(pizza_A_papad)
total_papad_A.append(burger_A_papad)
total_papad_A.append(nonveg_A_papad)
total_papad_A.append(veg_A_papad)
total_papad_A.append(dosa_A_papad)
total_papad_A.append(sandwich_A_papad)
total_papad_A.append(pavbhaji_A_papad)
total_papad_A.append(misal_A_papad)
total_papad_A.append(idli_A_papad)
total_papad_A.append(kichdi_A_papad)
total_papad_A=sum(total_papad_A)

total_butter_A=[]
total_butter_A.append(pizza_A_butter)
total_butter_A.append(burger_A_butter)
total_butter_A.append(nonveg_A_butter)
total_butter_A.append(veg_A_butter)
total_butter_A.append(dosa_A_butter)
total_butter_A.append(sandwich_A_butter)
total_butter_A.append(pavbhaji_A_butter)
total_butter_A.append(misal_A_butter)
total_butter_A.append(idli_A_butter)
total_butter_A.append(kichdi_A_butter)
total_butter_A=sum(total_butter_A)


#Analysis for restaurant B

#pizza analysis for group B

group_data_pizza_B=pizza_data.groupby("Restaurant")
pizza_for_rest_B=group_data_pizza_B.get_group('B')
weekday_input_pizza_B=pizza_for_rest_B[['Weekday']]
weekday_input_pizza_B=weekday_input_pizza_B.apply(labelencoder_X.fit_transform)	
pizza_out_B=pizza_for_rest_B[['Pizza']]
X_train_pizza_B,X_test_pizza_B,Y_train_pizza_B,Y_test_pizza_B = train_test_split(weekday_input_pizza_B,pizza_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_pizza_B, Y_train_pizza_B)

pizza_sum_B=0
for i in range(0,7):
	pizza_pred_B=regressor.predict([[i]])
	pizza_sum_B=pizza_sum_B+pizza_pred_B
pizza_sum_B=int(pizza_sum_B)

default_quantity_data_B_pizza=default_quantity_data_B.iloc[0,].values
pizza_B_list=default_quantity_data_B_pizza.tolist()
pizza_B_list=pizza_B_list[1:]

pizza_B_tomato=pizza_B_list[0]*pizza_sum_B
pizza_B_onion=pizza_B_list[1]*pizza_sum_B
pizza_B_capsicum=pizza_B_list[2]*pizza_sum_B
pizza_B_bread=pizza_B_list[3]*pizza_sum_B
pizza_B_dough=pizza_B_list[4]*pizza_sum_B
pizza_B_chicken=pizza_B_list[5]*pizza_sum_B
pizza_B_cheese=pizza_B_list[6]*pizza_sum_B
pizza_B_corn=pizza_B_list[7]*pizza_sum_B
pizza_B_rava=pizza_B_list[8]*pizza_sum_B
pizza_B_sabudana=pizza_B_list[9]*pizza_sum_B	
pizza_B_masala=pizza_B_list[10]*pizza_sum_B
pizza_B_vegetables=pizza_B_list[11]*pizza_sum_B
pizza_B_dal=pizza_B_list[12]*pizza_sum_B
pizza_B_flour=pizza_B_list[13]*pizza_sum_B
pizza_B_rice=pizza_B_list[14]*pizza_sum_B
pizza_B_papad=pizza_B_list[15]*pizza_sum_B
pizza_B_butter=pizza_B_list[16]*pizza_sum_B


#Burger Analysis for restaurant B

group_data_burger_B=burger_data.groupby("Restaurant")
burger_for_rest_B=group_data_burger_B.get_group('B')
weekday_input_burger_B=burger_for_rest_B[['Weekday']]
weekday_input_burger_B=weekday_input_burger_B.apply(labelencoder_X.fit_transform)	
burger_out_B=burger_for_rest_B[['Burger']]
X_train_burger_B,X_test_burger_B,Y_train_burger_B,Y_test_burger_B = train_test_split(weekday_input_burger_B,burger_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_burger_B, Y_train_burger_B)	

burger_sum_B=0
for i in range(0,7):
	burger_predict_sum=regressor.predict([[i]])
	burger_sum_B=burger_sum_B+burger_predict_sum
burger_sum_B=int(burger_sum_B)	

default_quantity_data_B_burger=default_quantity_data_B.iloc[1,].values
burger_B_list=default_quantity_data_B_burger.tolist()
burger_B_list=burger_B_list[1:]



burger_B_tomato=burger_B_list[0]*burger_sum_B
burger_B_onion=burger_B_list[1]*burger_sum_B
burger_B_capsicum=burger_B_list[2]*burger_sum_B
burger_B_bread=burger_B_list[3]*burger_sum_B
burger_B_dough=burger_B_list[4]*burger_sum_B
burger_B_chicken=burger_B_list[5]*burger_sum_B
burger_B_cheese=burger_B_list[6]*burger_sum_B
burger_B_corn=burger_B_list[7]*burger_sum_B
burger_B_rava=burger_B_list[8]*burger_sum_B
burger_B_sabudana=burger_B_list[9]*burger_sum_B	
burger_B_masala=burger_B_list[10]*burger_sum_B
burger_B_vegetables=burger_B_list[11]*burger_sum_B
burger_B_dal=burger_B_list[12]*burger_sum_B
burger_B_flour=burger_B_list[13]*burger_sum_B
burger_B_rice=burger_B_list[14]*burger_sum_B
burger_B_papad=burger_B_list[15]*burger_sum_B
burger_B_butter=burger_B_list[16]*burger_sum_B

#Non veg analysis for restaurant B

group_data_nonveg_B=non_veg_thali_data.groupby("Restaurant")
nonveg_for_rest_B=group_data_nonveg_B.get_group('B')
weekday_input_nonveg_B=nonveg_for_rest_B[['Weekday']]
weekday_input_nonveg_B=weekday_input_nonveg_B.apply(labelencoder_X.fit_transform)	
nonveg_out_B=nonveg_for_rest_B[['Non Veg Thali']]
X_train_nonveg_B,X_test_nonveg_B,Y_train_nonveg_B,Y_test_nonveg_B = train_test_split(weekday_input_nonveg_B,nonveg_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_nonveg_B, Y_train_nonveg_B)	

nonveg_sum_B=0
for i in range(0,7):
	nonveg_predict_sum=regressor.predict([[i]])
	nonveg_sum_B=nonveg_sum_B+nonveg_predict_sum
nonveg_sum_B=int(nonveg_sum_B)	

default_quantity_data_B_nonveg=default_quantity_data_B.iloc[2,].values
nonveg_B_list=default_quantity_data_B_nonveg.tolist()
nonveg_B_list=nonveg_B_list[1:]

nonveg_B_tomato=nonveg_B_list[0]*nonveg_sum_B
nonveg_B_onion=nonveg_B_list[1]*nonveg_sum_B
nonveg_B_capsicum=nonveg_B_list[2]*nonveg_sum_B
nonveg_B_bread=nonveg_B_list[3]*nonveg_sum_B
nonveg_B_dough=nonveg_B_list[4]*nonveg_sum_B
nonveg_B_chicken=nonveg_B_list[5]*nonveg_sum_B
nonveg_B_cheese=nonveg_B_list[6]*nonveg_sum_B
nonveg_B_corn=nonveg_B_list[7]*nonveg_sum_B
nonveg_B_rava=nonveg_B_list[8]*nonveg_sum_B
nonveg_B_sabudana=nonveg_B_list[9]*nonveg_sum_B	
nonveg_B_masala=nonveg_B_list[10]*nonveg_sum_B
nonveg_B_vegetables=nonveg_B_list[11]*nonveg_sum_B
nonveg_B_dal=nonveg_B_list[12]*nonveg_sum_B
nonveg_B_flour=nonveg_B_list[13]*nonveg_sum_B
nonveg_B_rice=nonveg_B_list[14]*nonveg_sum_B
nonveg_B_papad=nonveg_B_list[15]*nonveg_sum_B
nonveg_B_butter=nonveg_B_list[16]*nonveg_sum_B

#Veg analysis for restaurant B

group_data_veg_B=veg_thali_data.groupby("Restaurant")
veg_for_rest_B=group_data_veg_B.get_group('B')
weekday_input_veg_B=veg_for_rest_B[['Weekday']]
weekday_input_veg_B=weekday_input_veg_B.apply(labelencoder_X.fit_transform)	
veg_out_B=veg_for_rest_B[['Veg Thali']]
X_train_veg_B,X_test_veg_B,Y_train_veg_B,Y_test_veg_B = train_test_split(weekday_input_veg_B,veg_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_veg_B, Y_train_veg_B)

veg_sum_B=0
for i in range(0,7):
	veg_predict_sum=regressor.predict([[i]])
	veg_sum_B=veg_sum_B+veg_predict_sum
veg_sum_B=int(veg_sum_B)-1


default_quantity_data_B_veg=default_quantity_data_B.iloc[3,].values
veg_B_list=default_quantity_data_B_veg.tolist()
veg_B_list=veg_B_list[1:]

veg_B_tomato=veg_B_list[0]*veg_sum_B
veg_B_onion=veg_B_list[1]*veg_sum_B
veg_B_capsicum=veg_B_list[2]*veg_sum_B
veg_B_bread=veg_B_list[3]*veg_sum_B
veg_B_dough=veg_B_list[4]*veg_sum_B
veg_B_chicken=veg_B_list[5]*veg_sum_B
veg_B_cheese=veg_B_list[6]*veg_sum_B
veg_B_corn=veg_B_list[7]*veg_sum_B
veg_B_rava=veg_B_list[8]*veg_sum_B
veg_B_sabudana=veg_B_list[9]*veg_sum_B	
veg_B_masala=veg_B_list[10]*veg_sum_B
veg_B_vegetables=veg_B_list[11]*veg_sum_B
veg_B_dal=veg_B_list[12]*veg_sum_B
veg_B_flour=veg_B_list[13]*veg_sum_B
veg_B_rice=veg_B_list[14]*veg_sum_B
veg_B_papad=veg_B_list[15]*veg_sum_B
veg_B_butter=veg_B_list[16]*veg_sum_B


#DOSA analysis for restaurant B
group_data_dosa_B=dosa_data.groupby("Restaurant")
dosa_for_rest_B=group_data_dosa_B.get_group('B')
weekday_input_dosa_B=dosa_for_rest_B[['Weekday']]
weekday_input_dosa_B=weekday_input_dosa_B.apply(labelencoder_X.fit_transform)	
dosa_out_B=dosa_for_rest_B[['Dosa']]
X_train_dosa_B,X_test_dosa_B,Y_train_dosa_B,Y_test_dosa_B = train_test_split(weekday_input_dosa_B,dosa_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_dosa_B, Y_train_dosa_B)

dosa_sum_B=0
for i in range(0,7):
	dosa_predict_sum=regressor.predict([[i]])
	dosa_sum_B=dosa_sum_B+dosa_predict_sum
dosa_sum_B=int(dosa_sum_B)
	
default_quantity_data_B_dosa=default_quantity_data_B.iloc[4,].values
dosa_B_list=default_quantity_data_B_dosa.tolist()
dosa_B_list=dosa_B_list[1:]	

dosa_B_tomato=dosa_B_list[0]*dosa_sum_B
dosa_B_onion=dosa_B_list[1]*dosa_sum_B
dosa_B_capsicum=dosa_B_list[2]*dosa_sum_B
dosa_B_bread=dosa_B_list[3]*dosa_sum_B
dosa_B_dough=dosa_B_list[4]*dosa_sum_B
dosa_B_chicken=dosa_B_list[5]*dosa_sum_B
dosa_B_cheese=dosa_B_list[6]*dosa_sum_B
dosa_B_corn=dosa_B_list[7]*dosa_sum_B
dosa_B_rava=dosa_B_list[8]*dosa_sum_B
dosa_B_sabudana=dosa_B_list[9]*dosa_sum_B	
dosa_B_masala=dosa_B_list[10]*dosa_sum_B
dosa_B_vegetables=dosa_B_list[11]*dosa_sum_B
dosa_B_dal=dosa_B_list[12]*dosa_sum_B
dosa_B_flour=dosa_B_list[13]*dosa_sum_B
dosa_B_rice=dosa_B_list[14]*dosa_sum_B
dosa_B_papad=dosa_B_list[15]*dosa_sum_B
dosa_B_butter=dosa_B_list[16]*dosa_sum_B

#Sandwich ananlysis for restaurant B

group_data_sandwich_B=sandwich_data.groupby("Restaurant")
sandwich_for_rest_B=group_data_sandwich_B.get_group('B')
weekday_input_sandwich_B=sandwich_for_rest_B[['Weekday']]
weekday_input_sandwich_B=weekday_input_sandwich_B.apply(labelencoder_X.fit_transform)	
sandwich_out_B=sandwich_for_rest_B[['Sandwich']]	
X_train_sandwich_B,X_test_sandwich_B,Y_train_sandwich_B,Y_test_sandwich_B = train_test_split(weekday_input_sandwich_B,sandwich_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_sandwich_B, Y_train_sandwich_B)
	
sandwich_sum_B=0	
for i in range(0,7):
	sandwich_predict_sum=regressor.predict([[i]])
	sandwich_sum_B=sandwich_sum_B+sandwich_predict_sum
sandwich_sum_B=int(sandwich_sum_B)

default_quantity_data_B_sandwich=default_quantity_data_B.iloc[5,].values
sandwich_B_list=default_quantity_data_B_sandwich.tolist()
sandwich_B_list=sandwich_B_list[1:]	

sandwich_B_tomato=sandwich_B_list[0]*sandwich_sum_B
sandwich_B_onion=sandwich_B_list[1]*sandwich_sum_B
sandwich_B_capsicum=sandwich_B_list[2]*sandwich_sum_B
sandwich_B_bread=sandwich_B_list[3]*sandwich_sum_B
sandwich_B_dough=sandwich_B_list[4]*sandwich_sum_B
sandwich_B_chicken=sandwich_B_list[5]*sandwich_sum_B
sandwich_B_cheese=sandwich_B_list[6]*sandwich_sum_B
sandwich_B_corn=sandwich_B_list[7]*sandwich_sum_B
sandwich_B_rava=sandwich_B_list[8]*sandwich_sum_B
sandwich_B_sabudana=sandwich_B_list[9]*sandwich_sum_B	
sandwich_B_masala=sandwich_B_list[10]*sandwich_sum_B
sandwich_B_vegetables=sandwich_B_list[11]*sandwich_sum_B
sandwich_B_dal=sandwich_B_list[12]*sandwich_sum_B
sandwich_B_flour=sandwich_B_list[13]*sandwich_sum_B
sandwich_B_rice=sandwich_B_list[14]*sandwich_sum_B
sandwich_B_papad=sandwich_B_list[15]*sandwich_sum_B
sandwich_B_butter=sandwich_B_list[16]*sandwich_sum_B

#PAVBHAJI analysis for restauarnt B

group_data_pavbhaji_B=pav_bhaji_data.groupby("Restaurant")
pavbhaji_for_rest_B=group_data_pavbhaji_B.get_group('B')
weekday_input_pavbhaji_B=pavbhaji_for_rest_B[['Weekday']]
weekday_input_pavbhaji_B=weekday_input_pavbhaji_B.apply(labelencoder_X.fit_transform)	
pavbhaji_out_B=pavbhaji_for_rest_B[['Pav Bhaji']]	

X_train_pavbhaji_B,X_test_pavbhaji_B,Y_train_pavbhaji_B,Y_test_pavbhaji_B = train_test_split(weekday_input_pavbhaji_B,pavbhaji_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_pavbhaji_B, Y_train_pavbhaji_B)

pavbhaji_sum_B=0	
for i in range(0,7):
	pavbhaji_predict_sum=regressor.predict([[i]])
	pavbhaji_sum_B=pavbhaji_sum_B+pavbhaji_predict_sum
pavbhaji_sum_B=int(pavbhaji_sum_B)

default_quantity_data_B_pavbhaji=default_quantity_data_B.iloc[6,].values
pavbhaji_B_list=default_quantity_data_B_pavbhaji.tolist()
pavbhaji_B_list=pavbhaji_B_list[1:]	


pavbhaji_B_tomato=pavbhaji_B_list[0]*pavbhaji_sum_B
pavbhaji_B_onion=pavbhaji_B_list[1]*pavbhaji_sum_B
pavbhaji_B_capsicum=pavbhaji_B_list[2]*pavbhaji_sum_B
pavbhaji_B_bread=pavbhaji_B_list[3]*pavbhaji_sum_B
pavbhaji_B_dough=pavbhaji_B_list[4]*pavbhaji_sum_B
pavbhaji_B_chicken=pavbhaji_B_list[5]*pavbhaji_sum_B
pavbhaji_B_cheese=pavbhaji_B_list[6]*pavbhaji_sum_B
pavbhaji_B_corn=pavbhaji_B_list[7]*pavbhaji_sum_B
pavbhaji_B_rava=pavbhaji_B_list[8]*pavbhaji_sum_B
pavbhaji_B_sabudana=pavbhaji_B_list[9]*pavbhaji_sum_B	
pavbhaji_B_masala=pavbhaji_B_list[10]*pavbhaji_sum_B
pavbhaji_B_vegetables=pavbhaji_B_list[11]*pavbhaji_sum_B
pavbhaji_B_dal=pavbhaji_B_list[12]*pavbhaji_sum_B
pavbhaji_B_flour=pavbhaji_B_list[13]*pavbhaji_sum_B
pavbhaji_B_rice=pavbhaji_B_list[14]*pavbhaji_sum_B
pavbhaji_B_papad=pavbhaji_B_list[15]*pavbhaji_sum_B
pavbhaji_B_butter=pavbhaji_B_list[16]*pavbhaji_sum_B


#Misal analysis for restaurant B

group_data_misal_B=misal_data.groupby("Restaurant")
misal_for_rest_B=group_data_misal_B.get_group('B')
weekday_input_misal_B=misal_for_rest_B[['Weekday']]
weekday_input_misal_B=weekday_input_misal_B.apply(labelencoder_X.fit_transform)	
misal_out_B=misal_for_rest_B[['Misal']]	

X_train_misal_B,X_test_misal_B,Y_train_misal_B,Y_test_misal_B = train_test_split(weekday_input_misal_B,misal_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_misal_B, Y_train_misal_B)

misal_sum_B=0
for i in range(0,7):
	misal_predict_sum=regressor.predict([[i]])
	misal_sum_B=misal_sum_B+misal_predict_sum
misal_sum_B=int(misal_sum_B)+1

default_quantity_data_B_misal=default_quantity_data_B.iloc[7,].values
misal_B_list=default_quantity_data_B_misal.tolist()
misal_B_list=misal_B_list[1:]	

misal_B_tomato=misal_B_list[0]*misal_sum_B
misal_B_onion=misal_B_list[1]*misal_sum_B
misal_B_capsicum=misal_B_list[2]*misal_sum_B
misal_B_bread=misal_B_list[3]*misal_sum_B
misal_B_dough=misal_B_list[4]*misal_sum_B
misal_B_chicken=misal_B_list[5]*misal_sum_B
misal_B_cheese=misal_B_list[6]*misal_sum_B
misal_B_corn=misal_B_list[7]*misal_sum_B
misal_B_rava=misal_B_list[8]*misal_sum_B
misal_B_sabudana=misal_B_list[9]*misal_sum_B	
misal_B_masala=misal_B_list[10]*misal_sum_B
misal_B_vegetables=misal_B_list[11]*misal_sum_B
misal_B_dal=misal_B_list[12]*misal_sum_B
misal_B_flour=misal_B_list[13]*misal_sum_B
misal_B_rice=misal_B_list[14]*misal_sum_B
misal_B_papad=misal_B_list[15]*misal_sum_B
misal_B_butter=misal_B_list[16]*misal_sum_B



#idli analysis for restauarant B
group_data_idli_B=idli_data.groupby("Restaurant")
idli_for_rest_B=group_data_idli_B.get_group('B')
weekday_input_idli_B=idli_for_rest_B[['Weekday']]
weekday_input_idli_B=weekday_input_idli_B.apply(labelencoder_X.fit_transform)	
idli_out_B=idli_for_rest_B[['idli']]	

X_train_idli_B,X_test_idli_B,Y_train_idli_B,Y_test_idli_B = train_test_split(weekday_input_idli_B,idli_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_idli_B, Y_train_idli_B)

idli_sum_B=0

for i in range(0,7):
	idli_predict_sum=regressor.predict([[i]])
	idli_sum_B=idli_sum_B+idli_predict_sum
idli_sum_B=int(idli_sum_B)

default_quantity_data_B_idli=default_quantity_data_B.iloc[8,].values
idli_B_list=default_quantity_data_B_idli.tolist()
idli_B_list=idli_B_list[1:]	

idli_B_tomato=idli_B_list[0]*idli_sum_B
idli_B_onion=idli_B_list[1]*idli_sum_B
idli_B_capsicum=idli_B_list[2]*idli_sum_B
idli_B_bread=idli_B_list[3]*idli_sum_B
idli_B_dough=idli_B_list[4]*idli_sum_B
idli_B_chicken=idli_B_list[5]*idli_sum_B
idli_B_cheese=idli_B_list[6]*idli_sum_B
idli_B_corn=idli_B_list[7]*idli_sum_B
idli_B_rava=idli_B_list[8]*idli_sum_B
idli_B_sabudana=idli_B_list[9]*idli_sum_B	
idli_B_masala=idli_B_list[10]*idli_sum_B
idli_B_vegetables=idli_B_list[11]*idli_sum_B
idli_B_dal=idli_B_list[12]*idli_sum_B
idli_B_flour=idli_B_list[13]*idli_sum_B
idli_B_rice=idli_B_list[14]*idli_sum_B
idli_B_papad=idli_B_list[15]*idli_sum_B
idli_B_butter=idli_B_list[16]*idli_sum_B	


#kichdi analysis for restaurant B
group_data_kichdi_B=kichdi_data.groupby("Restaurant")
kichdi_for_rest_B=group_data_kichdi_B.get_group('B')
weekday_input_kichdi_B=kichdi_for_rest_B[['Weekday']]
weekday_input_kichdi_B=weekday_input_kichdi_B.apply(labelencoder_X.fit_transform)	
kichdi_out_B=kichdi_for_rest_B[['kichdi']]	

X_train_kichdi_B,X_test_kichdi_B,Y_train_kichdi_B,Y_test_kichdi_B = train_test_split(weekday_input_kichdi_B,kichdi_out_B,test_size=0.2,random_state=0)
regressor.fit(X_train_kichdi_B, Y_train_kichdi_B)

kichdi_sum_B=0
for i in range(0,7):
	kichdi_predict_sum=regressor.predict([[i]])
	kichdi_sum_B=kichdi_sum_B+kichdi_predict_sum
kichdi_sum_B=int(kichdi_sum_B)+2

default_quantity_data_B_kichdi=default_quantity_data_B.iloc[9,].values
kichdi_B_list=default_quantity_data_B_kichdi.tolist()
kichdi_B_list=kichdi_B_list[1:]	

kichdi_B_tomato=kichdi_B_list[0]*kichdi_sum_B
kichdi_B_onion=kichdi_B_list[1]*kichdi_sum_B
kichdi_B_capsicum=kichdi_B_list[2]*kichdi_sum_B
kichdi_B_bread=kichdi_B_list[3]*kichdi_sum_B
kichdi_B_dough=kichdi_B_list[4]*kichdi_sum_B
kichdi_B_chicken=kichdi_B_list[5]*kichdi_sum_B
kichdi_B_cheese=kichdi_B_list[6]*kichdi_sum_B
kichdi_B_corn=kichdi_B_list[7]*kichdi_sum_B
kichdi_B_rava=kichdi_B_list[8]*kichdi_sum_B
kichdi_B_sabudana=kichdi_B_list[9]*kichdi_sum_B	
kichdi_B_masala=kichdi_B_list[10]*kichdi_sum_B
kichdi_B_vegetables=kichdi_B_list[11]*kichdi_sum_B
kichdi_B_dal=kichdi_B_list[12]*kichdi_sum_B
kichdi_B_flour=kichdi_B_list[13]*kichdi_sum_B
kichdi_B_rice=kichdi_B_list[14]*kichdi_sum_B
kichdi_B_papad=kichdi_B_list[15]*kichdi_sum_B
kichdi_B_butter=kichdi_B_list[16]*kichdi_sum_B


total_tom_B=[]
total_tom_B.append(pizza_B_tomato)
total_tom_B.append(burger_B_tomato)
total_tom_B.append(nonveg_B_tomato)
total_tom_B.append(veg_B_tomato)
total_tom_B.append(dosa_B_tomato)
total_tom_B.append(sandwich_B_tomato)
total_tom_B.append(pavbhaji_B_tomato)
total_tom_B.append(misal_B_tomato)
total_tom_B.append(idli_B_tomato)
total_tom_B.append(kichdi_B_tomato)
total_tom_B=sum(total_tom_B)

total_onion_B=[]
total_onion_B.append(pizza_B_onion)
total_onion_B.append(burger_B_onion)
total_onion_B.append(nonveg_B_onion)
total_onion_B.append(veg_B_onion)
total_onion_B.append(dosa_B_onion)
total_onion_B.append(sandwich_B_onion)
total_onion_B.append(pavbhaji_B_onion)
total_onion_B.append(misal_B_onion)
total_onion_B.append(idli_B_onion)
total_onion_B.append(kichdi_B_onion)
total_onion_B=sum(total_onion_B)

total_capsicum_B=[]
total_capsicum_B.append(pizza_B_capsicum)
total_capsicum_B.append(burger_B_capsicum)
total_capsicum_B.append(nonveg_B_capsicum)
total_capsicum_B.append(veg_B_capsicum)
total_capsicum_B.append(dosa_B_capsicum)
total_capsicum_B.append(sandwich_B_capsicum)
total_capsicum_B.append(pavbhaji_B_capsicum)
total_capsicum_B.append(misal_B_capsicum)
total_capsicum_B.append(idli_B_capsicum)
total_capsicum_B.append(kichdi_B_capsicum)
total_capsicum_B=sum(total_capsicum_B)

total_bread_B=[]
total_bread_B.append(pizza_B_bread)
total_bread_B.append(burger_B_bread)
total_bread_B.append(nonveg_B_bread)
total_bread_B.append(veg_B_bread)
total_bread_B.append(dosa_B_bread)
total_bread_B.append(sandwich_B_bread)
total_bread_B.append(pavbhaji_B_bread)
total_bread_B.append(misal_B_bread)
total_bread_B.append(idli_B_bread)
total_bread_B.append(kichdi_B_bread)
total_bread_B=sum(total_bread_B)

total_dough_B=[]
total_dough_B.append(pizza_B_dough)
total_dough_B.append(burger_B_dough)
total_dough_B.append(nonveg_B_dough)
total_dough_B.append(veg_B_dough)
total_dough_B.append(dosa_B_dough)
total_dough_B.append(sandwich_B_dough)
total_dough_B.append(pavbhaji_B_dough)
total_dough_B.append(misal_B_dough)
total_dough_B.append(idli_B_dough)
total_dough_B.append(kichdi_B_dough)
total_dough_B=sum(total_dough_B)

total_chicken_B=[]
total_chicken_B.append(pizza_B_chicken)
total_chicken_B.append(burger_B_chicken)
total_chicken_B.append(nonveg_B_chicken)
total_chicken_B.append(veg_B_chicken)
total_chicken_B.append(dosa_B_chicken)
total_chicken_B.append(sandwich_B_chicken)
total_chicken_B.append(pavbhaji_B_chicken)
total_chicken_B.append(misal_B_chicken)
total_chicken_B.append(idli_B_chicken)
total_chicken_B.append(kichdi_B_chicken)
total_chicken_B=sum(total_chicken_B)

total_cheese_B=[]
total_cheese_B.append(pizza_B_cheese)
total_cheese_B.append(burger_B_cheese)
total_cheese_B.append(nonveg_B_cheese)
total_cheese_B.append(veg_B_cheese)
total_cheese_B.append(dosa_B_cheese)
total_cheese_B.append(sandwich_B_cheese)
total_cheese_B.append(pavbhaji_B_cheese)
total_cheese_B.append(misal_B_cheese)
total_cheese_B.append(idli_B_cheese)
total_cheese_B.append(kichdi_B_cheese)
total_cheese_B=sum(total_cheese_B)

total_corn_B=[]
total_corn_B.append(pizza_B_corn)
total_corn_B.append(burger_B_corn)
total_corn_B.append(nonveg_B_corn)
total_corn_B.append(veg_B_corn)
total_corn_B.append(dosa_B_corn)
total_corn_B.append(sandwich_B_corn)
total_corn_B.append(pavbhaji_B_corn)
total_corn_B.append(misal_B_corn)
total_corn_B.append(idli_B_corn)
total_corn_B.append(kichdi_B_corn)
total_corn_B=sum(total_corn_B)

total_rava_B=[]
total_rava_B.append(pizza_B_rava)
total_rava_B.append(burger_B_rava)
total_rava_B.append(nonveg_B_rava)
total_rava_B.append(veg_B_rava)
total_rava_B.append(dosa_B_rava)
total_rava_B.append(sandwich_B_rava)
total_rava_B.append(pavbhaji_B_rava)
total_rava_B.append(misal_B_rava)
total_rava_B.append(idli_B_rava)
total_rava_B.append(kichdi_B_rava)
total_rava_B=sum(total_rava_B)


total_sabu_B=[]
total_sabu_B.append(pizza_B_sabudana)
total_sabu_B.append(burger_B_sabudana)
total_sabu_B.append(nonveg_B_sabudana)
total_sabu_B.append(veg_B_sabudana)
total_sabu_B.append(dosa_B_sabudana)
total_sabu_B.append(sandwich_B_sabudana)
total_sabu_B.append(pavbhaji_B_sabudana)
total_sabu_B.append(misal_B_sabudana)
total_sabu_B.append(idli_B_sabudana)
total_sabu_B.append(kichdi_B_sabudana)
total_sabu_B=sum(total_sabu_B)

total_masala_B=[]
total_masala_B.append(pizza_B_masala)
total_masala_B.append(burger_B_masala)
total_masala_B.append(nonveg_B_masala)
total_masala_B.append(veg_B_masala)
total_masala_B.append(dosa_B_masala)
total_masala_B.append(sandwich_B_masala)
total_masala_B.append(pavbhaji_B_masala)
total_masala_B.append(misal_B_masala)
total_masala_B.append(idli_B_masala)
total_masala_B.append(kichdi_B_masala)
total_masala_B=sum(total_masala_B)

total_veggies_B=[]
total_veggies_B.append(pizza_B_vegetables)
total_veggies_B.append(burger_B_vegetables)
total_veggies_B.append(nonveg_B_vegetables)
total_veggies_B.append(veg_B_vegetables)
total_veggies_B.append(dosa_B_vegetables)
total_veggies_B.append(sandwich_B_vegetables)
total_veggies_B.append(pavbhaji_B_vegetables)
total_veggies_B.append(misal_B_vegetables)
total_veggies_B.append(idli_B_vegetables)
total_veggies_B.append(kichdi_B_vegetables)
total_veggies_B=sum(total_veggies_B)

total_dal_B=[]
total_dal_B.append(pizza_B_dal)
total_dal_B.append(burger_B_dal)
total_dal_B.append(nonveg_B_dal)
total_dal_B.append(veg_B_dal)
total_dal_B.append(dosa_B_dal)
total_dal_B.append(sandwich_B_dal)
total_dal_B.append(pavbhaji_B_dal)
total_dal_B.append(misal_B_dal)
total_dal_B.append(idli_B_dal)
total_dal_B.append(kichdi_B_dal)
total_dal_B=sum(total_dal_B)

total_flour_B=[]
total_flour_B.append(pizza_B_flour)
total_flour_B.append(burger_B_flour)
total_flour_B.append(nonveg_B_flour)
total_flour_B.append(veg_B_flour)
total_flour_B.append(dosa_B_flour)
total_flour_B.append(sandwich_B_flour)
total_flour_B.append(pavbhaji_B_flour)
total_flour_B.append(misal_B_flour)
total_flour_B.append(idli_B_flour)
total_flour_B.append(kichdi_B_flour)
total_flour_B=sum(total_flour_B)


total_rice_B=[]
total_rice_B.append(pizza_B_rice)
total_rice_B.append(burger_B_rice)
total_rice_B.append(nonveg_B_rice)
total_rice_B.append(veg_B_rice)
total_rice_B.append(dosa_B_rice)
total_rice_B.append(sandwich_B_rice)
total_rice_B.append(pavbhaji_B_rice)
total_rice_B.append(misal_B_rice)
total_rice_B.append(idli_B_rice)
total_rice_B.append(kichdi_B_rice)
total_rice_B=sum(total_rice_B)

total_papad_B=[]
total_papad_B.append(pizza_B_papad)
total_papad_B.append(burger_B_papad)
total_papad_B.append(nonveg_B_papad)
total_papad_B.append(veg_B_papad)
total_papad_B.append(dosa_B_papad)
total_papad_B.append(sandwich_B_papad)
total_papad_B.append(pavbhaji_B_papad)
total_papad_B.append(misal_B_papad)
total_papad_B.append(idli_B_papad)
total_papad_B.append(kichdi_B_papad)
total_papad_B=sum(total_papad_B)

total_butter_B=[]
total_butter_B.append(pizza_B_butter)
total_butter_B.append(burger_B_butter)
total_butter_B.append(nonveg_B_butter)
total_butter_B.append(veg_B_butter)
total_butter_B.append(dosa_B_butter)
total_butter_B.append(sandwich_B_butter)
total_butter_B.append(pavbhaji_B_butter)
total_butter_B.append(misal_B_butter)
total_butter_B.append(idli_B_butter)
total_butter_B.append(kichdi_B_butter)
total_butter_B=sum(total_butter_B)


#Analysis for restaurant C

#pizza analysis for group C

group_data_pizza_C=pizza_data.groupby("Restaurant")
pizza_for_rest_C=group_data_pizza_C.get_group('C')
weekday_input_pizza_C=pizza_for_rest_C[['Weekday']]
weekday_input_pizza_C=weekday_input_pizza_C.apply(labelencoder_X.fit_transform)	
pizza_out_C=pizza_for_rest_C[['Pizza']]
X_train_pizza_C,X_test_pizza_C,Y_train_pizza_C,Y_test_pizza_C = train_test_split(weekday_input_pizza_C,pizza_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_pizza_C, Y_train_pizza_C)

pizza_sum_C=0
for i in range(0,7):
	pizza_pred_C=regressor.predict([[i]])
	pizza_sum_C=pizza_sum_C+pizza_pred_C
pizza_sum_C=int(pizza_sum_C)+1

default_quantity_data_C_pizza=default_quantity_data_C.iloc[0,].values
pizza_C_list=default_quantity_data_C_pizza.tolist()
pizza_C_list=pizza_C_list[1:]

pizza_C_tomato=pizza_C_list[0]*pizza_sum_C
pizza_C_onion=pizza_C_list[1]*pizza_sum_C
pizza_C_capsicum=pizza_C_list[2]*pizza_sum_C
pizza_C_bread=pizza_C_list[3]*pizza_sum_C
pizza_C_dough=pizza_C_list[4]*pizza_sum_C
pizza_C_chicken=pizza_C_list[5]*pizza_sum_C
pizza_C_cheese=pizza_C_list[6]*pizza_sum_C
pizza_C_corn=pizza_C_list[7]*pizza_sum_C
pizza_C_rava=pizza_C_list[8]*pizza_sum_C
pizza_C_sabudana=pizza_C_list[9]*pizza_sum_C	
pizza_C_masala=pizza_C_list[10]*pizza_sum_C
pizza_C_vegetables=pizza_C_list[11]*pizza_sum_C
pizza_C_dal=pizza_C_list[12]*pizza_sum_C
pizza_C_flour=pizza_C_list[13]*pizza_sum_C
pizza_C_rice=pizza_C_list[14]*pizza_sum_C
pizza_C_papad=pizza_C_list[15]*pizza_sum_C
pizza_C_butter=pizza_C_list[16]*pizza_sum_C


#Burger analysis for restaurant C

group_data_Curger_C=burger_data.groupby("Restaurant")
burger_for_rest_C=group_data_Curger_C.get_group('C')
weekday_input_Curger_C=burger_for_rest_C[['Weekday']]
weekday_input_Curger_C=weekday_input_Curger_C.apply(labelencoder_X.fit_transform)	
burger_out_C=burger_for_rest_C[['Burger']]
X_train_Curger_C,X_test_Curger_C,Y_train_Curger_C,Y_test_Curger_C = train_test_split(weekday_input_Curger_C,burger_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_Curger_C, Y_train_Curger_C)	

burger_sum_C=0
for i in range(0,7):
	burger_predict_sum=regressor.predict([[i]])
	burger_sum_C=burger_sum_C+burger_predict_sum
burger_sum_C=int(burger_sum_C)	

default_quantity_data_C_burger=default_quantity_data_C.iloc[1,].values
burger_C_list=default_quantity_data_C_burger.tolist()
burger_C_list=burger_C_list[1:]



burger_C_tomato=burger_C_list[0]*burger_sum_C
burger_C_onion=burger_C_list[1]*burger_sum_C
burger_C_capsicum=burger_C_list[2]*burger_sum_C
burger_C_bread=burger_C_list[3]*burger_sum_C
burger_C_dough=burger_C_list[4]*burger_sum_C
burger_C_chicken=burger_C_list[5]*burger_sum_C
burger_C_cheese=burger_C_list[6]*burger_sum_C
burger_C_corn=burger_C_list[7]*burger_sum_C
burger_C_rava=burger_C_list[8]*burger_sum_C
burger_C_sabudana=burger_C_list[9]*burger_sum_C	
burger_C_masala=burger_C_list[10]*burger_sum_C
burger_C_vegetables=burger_C_list[11]*burger_sum_C
burger_C_dal=burger_C_list[12]*burger_sum_C
burger_C_flour=burger_C_list[13]*burger_sum_C
burger_C_rice=burger_C_list[14]*burger_sum_C
burger_C_papad=burger_C_list[15]*burger_sum_C
burger_C_butter=burger_C_list[16]*burger_sum_C

#NON veg thali analysis for restaurant C

group_data_nonveg_C=non_veg_thali_data.groupby("Restaurant")
nonveg_for_rest_C=group_data_nonveg_C.get_group('C')
weekday_input_nonveg_C=nonveg_for_rest_C[['Weekday']]
weekday_input_nonveg_C=weekday_input_nonveg_C.apply(labelencoder_X.fit_transform)	
nonveg_out_C=nonveg_for_rest_C[['Non Veg Thali']]
X_train_nonveg_C,X_test_nonveg_C,Y_train_nonveg_C,Y_test_nonveg_C = train_test_split(weekday_input_nonveg_C,nonveg_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_nonveg_C, Y_train_nonveg_C)	



nonveg_sum_C=0
for i in range(0,7):
	nonveg_predict_sum=regressor.predict([[i]])
	nonveg_sum_C=nonveg_sum_C+nonveg_predict_sum
nonveg_sum_C=int(nonveg_sum_C)	

default_quantity_data_C_nonveg=default_quantity_data_C.iloc[2,].values
nonveg_C_list=default_quantity_data_C_nonveg.tolist()
nonveg_C_list=nonveg_C_list[1:]

nonveg_C_tomato=nonveg_C_list[0]*nonveg_sum_C
nonveg_C_onion=nonveg_C_list[1]*nonveg_sum_C
nonveg_C_capsicum=nonveg_C_list[2]*nonveg_sum_C
nonveg_C_bread=nonveg_C_list[3]*nonveg_sum_C
nonveg_C_dough=nonveg_C_list[4]*nonveg_sum_C
nonveg_C_chicken=nonveg_C_list[5]*nonveg_sum_C
nonveg_C_cheese=nonveg_C_list[6]*nonveg_sum_C
nonveg_C_corn=nonveg_C_list[7]*nonveg_sum_C
nonveg_C_rava=nonveg_C_list[8]*nonveg_sum_C
nonveg_C_sabudana=nonveg_C_list[9]*nonveg_sum_C	
nonveg_C_masala=nonveg_C_list[10]*nonveg_sum_C
nonveg_C_vegetables=nonveg_C_list[11]*nonveg_sum_C
nonveg_C_dal=nonveg_C_list[12]*nonveg_sum_C
nonveg_C_flour=nonveg_C_list[13]*nonveg_sum_C
nonveg_C_rice=nonveg_C_list[14]*nonveg_sum_C
nonveg_C_papad=nonveg_C_list[15]*nonveg_sum_C
nonveg_C_butter=nonveg_C_list[16]*nonveg_sum_C


#Veg analysis for restaurant C

group_data_veg_C=veg_thali_data.groupby("Restaurant")
veg_for_rest_C=group_data_veg_C.get_group('C')
weekday_input_veg_C=veg_for_rest_C[['Weekday']]
weekday_input_veg_C=weekday_input_veg_C.apply(labelencoder_X.fit_transform)	
veg_out_C=veg_for_rest_C[['Veg Thali']]
X_train_veg_C,X_test_veg_C,Y_train_veg_C,Y_test_veg_C = train_test_split(weekday_input_veg_C,veg_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_veg_C, Y_train_veg_C)

veg_sum_C=0
for i in range(0,7):
	veg_predict_sum=regressor.predict([[i]])
	veg_sum_C=veg_sum_C+veg_predict_sum
veg_sum_C=int(veg_sum_C)


default_quantity_data_C_veg=default_quantity_data_C.iloc[3,].values
veg_C_list=default_quantity_data_C_veg.tolist()
veg_C_list=veg_C_list[1:]

veg_C_tomato=veg_C_list[0]*veg_sum_C
veg_C_onion=veg_C_list[1]*veg_sum_C
veg_C_capsicum=veg_C_list[2]*veg_sum_C
veg_C_bread=veg_C_list[3]*veg_sum_C
veg_C_dough=veg_C_list[4]*veg_sum_C
veg_C_chicken=veg_C_list[5]*veg_sum_C
veg_C_cheese=veg_C_list[6]*veg_sum_C
veg_C_corn=veg_C_list[7]*veg_sum_C
veg_C_rava=veg_C_list[8]*veg_sum_C
veg_C_sabudana=veg_C_list[9]*veg_sum_C	
veg_C_masala=veg_C_list[10]*veg_sum_C
veg_C_vegetables=veg_C_list[11]*veg_sum_C
veg_C_dal=veg_C_list[12]*veg_sum_C
veg_C_flour=veg_C_list[13]*veg_sum_C
veg_C_rice=veg_C_list[14]*veg_sum_C
veg_C_papad=veg_C_list[15]*veg_sum_C
veg_C_butter=veg_C_list[16]*veg_sum_C

#DOSA analysis for restaurant B
group_data_dosa_C=dosa_data.groupby("Restaurant")
dosa_for_rest_C=group_data_dosa_C.get_group('C')
weekday_input_dosa_C=dosa_for_rest_C[['Weekday']]
weekday_input_dosa_C=weekday_input_dosa_C.apply(labelencoder_X.fit_transform)	
dosa_out_C=dosa_for_rest_C[['Dosa']]
X_train_dosa_C,X_test_dosa_C,Y_train_dosa_C,Y_test_dosa_C = train_test_split(weekday_input_dosa_C,dosa_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_dosa_C, Y_train_dosa_C)

dosa_sum_C=0
for i in range(0,7):
	dosa_predict_sum=regressor.predict([[i]])
	dosa_sum_C=dosa_sum_C+dosa_predict_sum
dosa_sum_C=int(dosa_sum_C)
	
default_quantity_data_C_dosa=default_quantity_data_C.iloc[4,].values
dosa_C_list=default_quantity_data_C_dosa.tolist()
dosa_C_list=dosa_C_list[1:]	

dosa_C_tomato=dosa_C_list[0]*dosa_sum_C
dosa_C_onion=dosa_C_list[1]*dosa_sum_C
dosa_C_capsicum=dosa_C_list[2]*dosa_sum_C
dosa_C_bread=dosa_C_list[3]*dosa_sum_C
dosa_C_dough=dosa_C_list[4]*dosa_sum_C
dosa_C_chicken=dosa_C_list[5]*dosa_sum_C
dosa_C_cheese=dosa_C_list[6]*dosa_sum_C
dosa_C_corn=dosa_C_list[7]*dosa_sum_C
dosa_C_rava=dosa_C_list[8]*dosa_sum_C
dosa_C_sabudana=dosa_C_list[9]*dosa_sum_C	
dosa_C_masala=dosa_C_list[10]*dosa_sum_C
dosa_C_vegetables=dosa_C_list[11]*dosa_sum_C
dosa_C_dal=dosa_C_list[12]*dosa_sum_C
dosa_C_flour=dosa_C_list[13]*dosa_sum_C
dosa_C_rice=dosa_C_list[14]*dosa_sum_C
dosa_C_papad=dosa_C_list[15]*dosa_sum_C
dosa_C_butter=dosa_C_list[16]*dosa_sum_C

#Sandwich ananlysis for restaurant C

group_data_sandwich_C=sandwich_data.groupby("Restaurant")
sandwich_for_rest_C=group_data_sandwich_C.get_group('C')
weekday_input_sandwich_C=sandwich_for_rest_C[['Weekday']]
weekday_input_sandwich_C=weekday_input_sandwich_C.apply(labelencoder_X.fit_transform)	
sandwich_out_C=sandwich_for_rest_C[['Sandwich']]	
X_train_sandwich_C,X_test_sandwich_C,Y_train_sandwich_C,Y_test_sandwich_C = train_test_split(weekday_input_sandwich_C,sandwich_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_sandwich_C, Y_train_sandwich_C)
	
sandwich_sum_C=0	
for i in range(0,7):
	sandwich_predict_sum=regressor.predict([[i]])
	sandwich_sum_C=sandwich_sum_C+sandwich_predict_sum
sandwich_sum_C=int(sandwich_sum_C)

default_quantity_data_C_sandwich=default_quantity_data_C.iloc[5,].values
sandwich_C_list=default_quantity_data_C_sandwich.tolist()
sandwich_C_list=sandwich_C_list[1:]	

sandwich_C_tomato=sandwich_C_list[0]*sandwich_sum_C
sandwich_C_onion=sandwich_C_list[1]*sandwich_sum_C
sandwich_C_capsicum=sandwich_C_list[2]*sandwich_sum_C
sandwich_C_bread=sandwich_C_list[3]*sandwich_sum_C
sandwich_C_dough=sandwich_C_list[4]*sandwich_sum_C
sandwich_C_chicken=sandwich_C_list[5]*sandwich_sum_C
sandwich_C_cheese=sandwich_C_list[6]*sandwich_sum_C
sandwich_C_corn=sandwich_C_list[7]*sandwich_sum_C
sandwich_C_rava=sandwich_C_list[8]*sandwich_sum_C
sandwich_C_sabudana=sandwich_C_list[9]*sandwich_sum_C	
sandwich_C_masala=sandwich_C_list[10]*sandwich_sum_C
sandwich_C_vegetables=sandwich_C_list[11]*sandwich_sum_C
sandwich_C_dal=sandwich_C_list[12]*sandwich_sum_C
sandwich_C_flour=sandwich_C_list[13]*sandwich_sum_C
sandwich_C_rice=sandwich_C_list[14]*sandwich_sum_C
sandwich_C_papad=sandwich_C_list[15]*sandwich_sum_C
sandwich_C_butter=sandwich_C_list[16]*sandwich_sum_C


#PAVBHAJI analysis for restauarnt C

group_data_pavbhaji_C=pav_bhaji_data.groupby("Restaurant")
pavbhaji_for_rest_C=group_data_pavbhaji_C.get_group('C')
weekday_input_pavbhaji_C=pavbhaji_for_rest_C[['Weekday']]
weekday_input_pavbhaji_C=weekday_input_pavbhaji_C.apply(labelencoder_X.fit_transform)	
pavbhaji_out_C=pavbhaji_for_rest_C[['Pav Bhaji']]	

X_train_pavbhaji_C,X_test_pavbhaji_C,Y_train_pavbhaji_C,Y_test_pavbhaji_C = train_test_split(weekday_input_pavbhaji_C,pavbhaji_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_pavbhaji_C, Y_train_pavbhaji_C)

pavbhaji_sum_C=0	
for i in range(0,7):
	pavbhaji_predict_sum=regressor.predict([[i]])
	pavbhaji_sum_C=pavbhaji_sum_C+pavbhaji_predict_sum
pavbhaji_sum_C=int(pavbhaji_sum_C)

default_quantity_data_C_pavbhaji=default_quantity_data_C.iloc[6,].values
pavbhaji_C_list=default_quantity_data_C_pavbhaji.tolist()
pavbhaji_C_list=pavbhaji_C_list[1:]	


pavbhaji_C_tomato=pavbhaji_C_list[0]*pavbhaji_sum_C
pavbhaji_C_onion=pavbhaji_C_list[1]*pavbhaji_sum_C
pavbhaji_C_capsicum=pavbhaji_C_list[2]*pavbhaji_sum_C
pavbhaji_C_bread=pavbhaji_C_list[3]*pavbhaji_sum_C
pavbhaji_C_dough=pavbhaji_C_list[4]*pavbhaji_sum_C
pavbhaji_C_chicken=pavbhaji_C_list[5]*pavbhaji_sum_C
pavbhaji_C_cheese=pavbhaji_C_list[6]*pavbhaji_sum_C
pavbhaji_C_corn=pavbhaji_C_list[7]*pavbhaji_sum_C
pavbhaji_C_rava=pavbhaji_C_list[8]*pavbhaji_sum_C
pavbhaji_C_sabudana=pavbhaji_C_list[9]*pavbhaji_sum_C	
pavbhaji_C_masala=pavbhaji_C_list[10]*pavbhaji_sum_C
pavbhaji_C_vegetables=pavbhaji_C_list[11]*pavbhaji_sum_C
pavbhaji_C_dal=pavbhaji_C_list[12]*pavbhaji_sum_C
pavbhaji_C_flour=pavbhaji_C_list[13]*pavbhaji_sum_C
pavbhaji_C_rice=pavbhaji_C_list[14]*pavbhaji_sum_C
pavbhaji_C_papad=pavbhaji_C_list[15]*pavbhaji_sum_C
pavbhaji_C_butter=pavbhaji_C_list[16]*pavbhaji_sum_C


#Misal analysis for restaurant C

group_data_misal_C=misal_data.groupby("Restaurant")
misal_for_rest_C=group_data_misal_C.get_group('C')
weekday_input_misal_C=misal_for_rest_C[['Weekday']]
weekday_input_misal_C=weekday_input_misal_C.apply(labelencoder_X.fit_transform)	
misal_out_C=misal_for_rest_C[['Misal']]	

X_train_misal_C,X_test_misal_C,Y_train_misal_C,Y_test_misal_C = train_test_split(weekday_input_misal_C,misal_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_misal_C, Y_train_misal_C)

misal_sum_C=0
for i in range(0,7):
	misal_predict_sum=regressor.predict([[i]])
	misal_sum_C=misal_sum_C+misal_predict_sum
misal_sum_C=int(misal_sum_C)+1

default_quantity_data_C_misal=default_quantity_data_C.iloc[7,].values
misal_C_list=default_quantity_data_C_misal.tolist()
misal_C_list=misal_C_list[1:]	

misal_C_tomato=misal_C_list[0]*misal_sum_C
misal_C_onion=misal_C_list[1]*misal_sum_C
misal_C_capsicum=misal_C_list[2]*misal_sum_C
misal_C_bread=misal_C_list[3]*misal_sum_C
misal_C_dough=misal_C_list[4]*misal_sum_C
misal_C_chicken=misal_C_list[5]*misal_sum_C
misal_C_cheese=misal_C_list[6]*misal_sum_C
misal_C_corn=misal_C_list[7]*misal_sum_C
misal_C_rava=misal_C_list[8]*misal_sum_C
misal_C_sabudana=misal_C_list[9]*misal_sum_C	
misal_C_masala=misal_C_list[10]*misal_sum_C
misal_C_vegetables=misal_C_list[11]*misal_sum_C
misal_C_dal=misal_C_list[12]*misal_sum_C
misal_C_flour=misal_C_list[13]*misal_sum_C
misal_C_rice=misal_C_list[14]*misal_sum_C
misal_C_papad=misal_C_list[15]*misal_sum_C
misal_C_butter=misal_C_list[16]*misal_sum_C


#idli analysis for restauarant C
group_data_idli_C=idli_data.groupby("Restaurant")
idli_for_rest_C=group_data_idli_C.get_group('C')
weekday_input_idli_C=idli_for_rest_C[['Weekday']]
weekday_input_idli_C=weekday_input_idli_C.apply(labelencoder_X.fit_transform)	
idli_out_C=idli_for_rest_C[['idli']]	

X_train_idli_C,X_test_idli_C,Y_train_idli_C,Y_test_idli_C = train_test_split(weekday_input_idli_C,idli_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_idli_C, Y_train_idli_C)

idli_sum_C=0

for i in range(0,7):
	idli_predict_sum=regressor.predict([[i]])
	idli_sum_C=idli_sum_C+idli_predict_sum
idli_sum_C=int(idli_sum_C)

default_quantity_data_C_idli=default_quantity_data_C.iloc[8,].values
idli_C_list=default_quantity_data_C_idli.tolist()
idli_C_list=idli_C_list[1:]	

idli_C_tomato=idli_C_list[0]*idli_sum_C
idli_C_onion=idli_C_list[1]*idli_sum_C
idli_C_capsicum=idli_C_list[2]*idli_sum_C
idli_C_bread=idli_C_list[3]*idli_sum_C
idli_C_dough=idli_C_list[4]*idli_sum_C
idli_C_chicken=idli_C_list[5]*idli_sum_C
idli_C_cheese=idli_C_list[6]*idli_sum_C
idli_C_corn=idli_C_list[7]*idli_sum_C
idli_C_rava=idli_C_list[8]*idli_sum_C
idli_C_sabudana=idli_C_list[9]*idli_sum_C	
idli_C_masala=idli_C_list[10]*idli_sum_C
idli_C_vegetables=idli_C_list[11]*idli_sum_C
idli_C_dal=idli_C_list[12]*idli_sum_C
idli_C_flour=idli_C_list[13]*idli_sum_C
idli_C_rice=idli_C_list[14]*idli_sum_C
idli_C_papad=idli_C_list[15]*idli_sum_C
idli_C_butter=idli_C_list[16]*idli_sum_C



#kichdi analysis for restaurant C
group_data_kichdi_C=kichdi_data.groupby("Restaurant")
kichdi_for_rest_C=group_data_kichdi_C.get_group('C')
weekday_input_kichdi_C=kichdi_for_rest_C[['Weekday']]
weekday_input_kichdi_C=weekday_input_kichdi_C.apply(labelencoder_X.fit_transform)	
kichdi_out_C=kichdi_for_rest_C[['kichdi']]	

X_train_kichdi_C,X_test_kichdi_C,Y_train_kichdi_C,Y_test_kichdi_C = train_test_split(weekday_input_kichdi_C,kichdi_out_C,test_size=0.2,random_state=0)
regressor.fit(X_train_kichdi_C, Y_train_kichdi_C)

kichdi_sum_C=0
for i in range(0,7):
	kichdi_predict_sum=regressor.predict([[i]])
	kichdi_sum_C=kichdi_sum_C+kichdi_predict_sum
kichdi_sum_C=int(kichdi_sum_C)

default_quantity_data_C_kichdi=default_quantity_data_C.iloc[9,].values
kichdi_C_list=default_quantity_data_C_kichdi.tolist()
kichdi_C_list=kichdi_C_list[1:]	

kichdi_C_tomato=kichdi_C_list[0]*kichdi_sum_C
kichdi_C_onion=kichdi_C_list[1]*kichdi_sum_C
kichdi_C_capsicum=kichdi_C_list[2]*kichdi_sum_C
kichdi_C_bread=kichdi_C_list[3]*kichdi_sum_C
kichdi_C_dough=kichdi_C_list[4]*kichdi_sum_C
kichdi_C_chicken=kichdi_C_list[5]*kichdi_sum_C
kichdi_C_cheese=kichdi_C_list[6]*kichdi_sum_C
kichdi_C_corn=kichdi_C_list[7]*kichdi_sum_C
kichdi_C_rava=kichdi_C_list[8]*kichdi_sum_C
kichdi_C_sabudana=kichdi_C_list[9]*kichdi_sum_C	
kichdi_C_masala=kichdi_C_list[10]*kichdi_sum_C
kichdi_C_vegetables=kichdi_C_list[11]*kichdi_sum_C
kichdi_C_dal=kichdi_C_list[12]*kichdi_sum_C
kichdi_C_flour=kichdi_C_list[13]*kichdi_sum_C
kichdi_C_rice=kichdi_C_list[14]*kichdi_sum_C
kichdi_C_papad=kichdi_C_list[15]*kichdi_sum_C
kichdi_C_butter=kichdi_C_list[16]*kichdi_sum_C



total_tom_C=[]
total_tom_C.append(pizza_C_tomato)
total_tom_C.append(burger_C_tomato)
total_tom_C.append(nonveg_C_tomato)
total_tom_C.append(veg_C_tomato)
total_tom_C.append(dosa_C_tomato)
total_tom_C.append(sandwich_C_tomato)
total_tom_C.append(pavbhaji_C_tomato)
total_tom_C.append(misal_C_tomato)
total_tom_C.append(idli_C_tomato)
total_tom_C.append(kichdi_C_tomato)
total_tom_C=sum(total_tom_C)

total_onion_C=[]
total_onion_C.append(pizza_C_onion)
total_onion_C.append(burger_C_onion)
total_onion_C.append(nonveg_C_onion)
total_onion_C.append(veg_C_onion)
total_onion_C.append(dosa_C_onion)
total_onion_C.append(sandwich_C_onion)
total_onion_C.append(pavbhaji_C_onion)
total_onion_C.append(misal_C_onion)
total_onion_C.append(idli_C_onion)
total_onion_C.append(kichdi_C_onion)
total_onion_C=sum(total_onion_C)

total_capsicum_C=[]
total_capsicum_C.append(pizza_C_capsicum)
total_capsicum_C.append(burger_C_capsicum)
total_capsicum_C.append(nonveg_C_capsicum)
total_capsicum_C.append(veg_C_capsicum)
total_capsicum_C.append(dosa_C_capsicum)
total_capsicum_C.append(sandwich_C_capsicum)
total_capsicum_C.append(pavbhaji_C_capsicum)
total_capsicum_C.append(misal_C_capsicum)
total_capsicum_C.append(idli_C_capsicum)
total_capsicum_C.append(kichdi_C_capsicum)
total_capsicum_C=sum(total_capsicum_C)

total_bread_C=[]
total_bread_C.append(pizza_C_bread)
total_bread_C.append(burger_C_bread)
total_bread_C.append(nonveg_C_bread)
total_bread_C.append(veg_C_bread)
total_bread_C.append(dosa_C_bread)
total_bread_C.append(sandwich_C_bread)
total_bread_C.append(pavbhaji_C_bread)
total_bread_C.append(misal_C_bread)
total_bread_C.append(idli_C_bread)
total_bread_C.append(kichdi_C_bread)
total_bread_C=sum(total_bread_C)

total_dough_C=[]
total_dough_C.append(pizza_C_dough)
total_dough_C.append(burger_C_dough)
total_dough_C.append(nonveg_C_dough)
total_dough_C.append(veg_C_dough)
total_dough_C.append(dosa_C_dough)
total_dough_C.append(sandwich_C_dough)
total_dough_C.append(pavbhaji_C_dough)
total_dough_C.append(misal_C_dough)
total_dough_C.append(idli_C_dough)
total_dough_C.append(kichdi_C_dough)
total_dough_C=sum(total_dough_C)

total_chicken_C=[]
total_chicken_C.append(pizza_C_chicken)
total_chicken_C.append(burger_C_chicken)
total_chicken_C.append(nonveg_C_chicken)
total_chicken_C.append(veg_C_chicken)
total_chicken_C.append(dosa_C_chicken)
total_chicken_C.append(sandwich_C_chicken)
total_chicken_C.append(pavbhaji_C_chicken)
total_chicken_C.append(misal_C_chicken)
total_chicken_C.append(idli_C_chicken)
total_chicken_C.append(kichdi_C_chicken)
total_chicken_C=sum(total_chicken_C)

total_cheese_C=[]
total_cheese_C.append(pizza_C_cheese)
total_cheese_C.append(burger_C_cheese)
total_cheese_C.append(nonveg_C_cheese)
total_cheese_C.append(veg_C_cheese)
total_cheese_C.append(dosa_C_cheese)
total_cheese_C.append(sandwich_C_cheese)
total_cheese_C.append(pavbhaji_C_cheese)
total_cheese_C.append(misal_C_cheese)
total_cheese_C.append(idli_C_cheese)
total_cheese_C.append(kichdi_C_cheese)
total_cheese_C=sum(total_cheese_C)

total_corn_C=[]
total_corn_C.append(pizza_C_corn)
total_corn_C.append(burger_C_corn)
total_corn_C.append(nonveg_C_corn)
total_corn_C.append(veg_C_corn)
total_corn_C.append(dosa_C_corn)
total_corn_C.append(sandwich_C_corn)
total_corn_C.append(pavbhaji_C_corn)
total_corn_C.append(misal_C_corn)
total_corn_C.append(idli_C_corn)
total_corn_C.append(kichdi_C_corn)
total_corn_C=sum(total_corn_C)

total_rava_C=[]
total_rava_C.append(pizza_C_rava)
total_rava_C.append(burger_C_rava)
total_rava_C.append(nonveg_C_rava)
total_rava_C.append(veg_C_rava)
total_rava_C.append(dosa_C_rava)
total_rava_C.append(sandwich_C_rava)
total_rava_C.append(pavbhaji_C_rava)
total_rava_C.append(misal_C_rava)
total_rava_C.append(idli_C_rava)
total_rava_C.append(kichdi_C_rava)
total_rava_C=sum(total_rava_C)


total_sabu_C=[]
total_sabu_C.append(pizza_C_sabudana)
total_sabu_C.append(burger_C_sabudana)
total_sabu_C.append(nonveg_C_sabudana)
total_sabu_C.append(veg_C_sabudana)
total_sabu_C.append(dosa_C_sabudana)
total_sabu_C.append(sandwich_C_sabudana)
total_sabu_C.append(pavbhaji_C_sabudana)
total_sabu_C.append(misal_C_sabudana)
total_sabu_C.append(idli_C_sabudana)
total_sabu_C.append(kichdi_C_sabudana)
total_sabu_C=sum(total_sabu_C)

total_masala_C=[]
total_masala_C.append(pizza_C_masala)
total_masala_C.append(burger_C_masala)
total_masala_C.append(nonveg_C_masala)
total_masala_C.append(veg_C_masala)
total_masala_C.append(dosa_C_masala)
total_masala_C.append(sandwich_C_masala)
total_masala_C.append(pavbhaji_C_masala)
total_masala_C.append(misal_C_masala)
total_masala_C.append(idli_C_masala)
total_masala_C.append(kichdi_C_masala)
total_masala_C=sum(total_masala_C)

total_veggies_C=[]
total_veggies_C.append(pizza_C_vegetables)
total_veggies_C.append(burger_C_vegetables)
total_veggies_C.append(nonveg_C_vegetables)
total_veggies_C.append(veg_C_vegetables)
total_veggies_C.append(dosa_C_vegetables)
total_veggies_C.append(sandwich_C_vegetables)
total_veggies_C.append(pavbhaji_C_vegetables)
total_veggies_C.append(misal_C_vegetables)
total_veggies_C.append(idli_C_vegetables)
total_veggies_C.append(kichdi_C_vegetables)
total_veggies_C=sum(total_veggies_C)

total_dal_C=[]
total_dal_C.append(pizza_C_dal)
total_dal_C.append(burger_C_dal)
total_dal_C.append(nonveg_C_dal)
total_dal_C.append(veg_C_dal)
total_dal_C.append(dosa_C_dal)
total_dal_C.append(sandwich_C_dal)
total_dal_C.append(pavbhaji_C_dal)
total_dal_C.append(misal_C_dal)
total_dal_C.append(idli_C_dal)
total_dal_C.append(kichdi_C_dal)
total_dal_C=sum(total_dal_C)

total_flour_C=[]
total_flour_C.append(pizza_C_flour)
total_flour_C.append(burger_C_flour)
total_flour_C.append(nonveg_C_flour)
total_flour_C.append(veg_C_flour)
total_flour_C.append(dosa_C_flour)
total_flour_C.append(sandwich_C_flour)
total_flour_C.append(pavbhaji_C_flour)
total_flour_C.append(misal_C_flour)
total_flour_C.append(idli_C_flour)
total_flour_C.append(kichdi_C_flour)
total_flour_C=sum(total_flour_C)


total_rice_C=[]
total_rice_C.append(pizza_C_rice)
total_rice_C.append(burger_C_rice)
total_rice_C.append(nonveg_C_rice)
total_rice_C.append(veg_C_rice)
total_rice_C.append(dosa_C_rice)
total_rice_C.append(sandwich_C_rice)
total_rice_C.append(pavbhaji_C_rice)
total_rice_C.append(misal_C_rice)
total_rice_C.append(idli_C_rice)
total_rice_C.append(kichdi_C_rice)
total_rice_C=sum(total_rice_C)

total_papad_C=[]
total_papad_C.append(pizza_C_papad)
total_papad_C.append(burger_C_papad)
total_papad_C.append(nonveg_C_papad)
total_papad_C.append(veg_C_papad)
total_papad_C.append(dosa_C_papad)
total_papad_C.append(sandwich_C_papad)
total_papad_C.append(pavbhaji_C_papad)
total_papad_C.append(misal_C_papad)
total_papad_C.append(idli_C_papad)
total_papad_C.append(kichdi_C_papad)
total_papad_C=sum(total_papad_C)

total_butter_C=[]
total_butter_C.append(pizza_C_butter)
total_butter_C.append(burger_C_butter)
total_butter_C.append(nonveg_C_butter)
total_butter_C.append(veg_C_butter)
total_butter_C.append(dosa_C_butter)
total_butter_C.append(sandwich_C_butter)
total_butter_C.append(pavbhaji_C_butter)
total_butter_C.append(misal_C_butter)
total_butter_C.append(idli_C_butter)
total_butter_C.append(kichdi_C_butter)
total_butter_C=sum(total_butter_C)




#Total calculation of the vendor side

total_tomato_vendor=[]
total_tomato_vendor.append(total_tom_A)
total_tomato_vendor.append(total_tom_B)
total_tomato_vendor.append(total_tom_C)
total_tomato_vendor=sum(total_tomato_vendor)

total_onion_vendor=[]
total_onion_vendor.append(total_onion_A)
total_onion_vendor.append(total_onion_B)
total_onion_vendor.append(total_onion_B)
total_onion_vendor=sum(total_onion_vendor)

total_capsicum_vendor=[]
total_capsicum_vendor.append(total_capsicum_A)
total_capsicum_vendor.append(total_capsicum_B)
total_capsicum_vendor.append(total_capsicum_C)
total_capsicum_vendor=sum(total_capsicum_vendor)

total_bread_vendor=[]
total_bread_vendor.append(total_bread_A)
total_bread_vendor.append(total_bread_B)
total_bread_vendor.append(total_bread_C)
total_bread_vendor=sum(total_bread_vendor)

total_dough_vendor=[]
total_dough_vendor.append(total_dough_A)
total_dough_vendor.append(total_dough_B)
total_dough_vendor.append(total_dough_C)
total_dough_vendor=sum(total_dough_vendor)

total_chicken_vendor=[]
total_chicken_vendor.append(total_chicken_A)
total_chicken_vendor.append(total_chicken_B)
total_chicken_vendor.append(total_chicken_C)
total_chicken_vendor=sum(total_chicken_vendor)


total_cheese_vendor=[]
total_cheese_vendor.append(total_cheese_A)
total_cheese_vendor.append(total_cheese_B)
total_cheese_vendor.append(total_cheese_C)
total_cheese_vendor=sum(total_cheese_vendor)

total_corn_vendor=[]
total_corn_vendor.append(total_corn_A)
total_corn_vendor.append(total_corn_B)
total_corn_vendor.append(total_bread_C)
total_corn_vendor=sum(total_corn_vendor)

total_rava_vendor=[]
total_rava_vendor.append(total_rava_A)
total_rava_vendor.append(total_rava_B)
total_rava_vendor.append(total_rava_C)
total_rava_vendor=sum(total_rava_vendor)

total_sabudana_vendor=[]
total_sabudana_vendor.append(total_sabu_A)
total_sabudana_vendor.append(total_sabu_B)
total_sabudana_vendor.append(total_sabu_C)
total_sabudana_vendor=sum(total_sabudana_vendor)

total_masala_vendor=[]
total_masala_vendor.append(total_masala_A)
total_masala_vendor.append(total_masala_B)
total_masala_vendor.append(total_bread_C)
total_masala_vendor=sum(total_masala_vendor)


total_vegetables_vendor=[]
total_vegetables_vendor.append(total_veggies_A)
total_vegetables_vendor.append(total_veggies_B)
total_vegetables_vendor.append(total_veggies_C)
total_vegetables_vendor=sum(total_vegetables_vendor)

total_dal_vendor=[]
total_dal_vendor.append(total_dal_A)
total_dal_vendor.append(total_dal_B)
total_dal_vendor.append(total_dal_C)
total_dal_vendor=sum(total_dal_vendor)

total_flour_vendor=[]
total_flour_vendor.append(total_flour_A)
total_flour_vendor.append(total_flour_B)
total_flour_vendor.append(total_flour_C)
total_flour_vendor=sum(total_flour_vendor)

total_rice_vendor=[]
total_rice_vendor.append(total_rice_A)
total_rice_vendor.append(total_rice_B)
total_rice_vendor.append(total_rice_C)
total_rice_vendor=sum(total_rice_vendor)

total_papad_vendor=[]
total_papad_vendor.append(total_papad_A)
total_papad_vendor.append(total_papad_B)
total_papad_vendor.append(total_papad_C)
total_papad_vendor=sum(total_papad_vendor)

total_butter_vendor=[]
total_butter_vendor.append(total_butter_A)
total_butter_vendor.append(total_butter_B)
total_butter_vendor.append(total_butter_C)
total_butter_vendor=sum(total_butter_vendor)

print("A single kg of tomatos contains 6 tomatos")
print("A single kg of onions contains 5 tomatos")
print("A single kg of capsicums contains 6 capsicums")
print("A single bread packet conatins 10 slices of bread")
print("A single kg of dough contains 8 servings")
print("A single kg of chicken contains 12 servings")
print("A single packet contains 6 slices of cheese")
print("A single kg of corn contains 25 servings")
print("A single kg of rava contains 28 servings")
print("A single kg of sabudana contains 40 servings")
print("A single packet of masala contains 50 servings ")
print("A single kg of vegetables contains 24 servings")
print("A single kg of dal contains 10 servings")
print("A single kg of flour requires 19 servings")
print("A single kg of rice requires 23 servings")
print("A single packet of papad contains 18 papads")
print("A single packet of Butter contains 24 servings")
print()
print()



yes=1
while(yes==1):
	print()
	print("Press 1 for analysis of restaurant A")
	print("press 2 for analysis of restaurant B")
	print("press 3 for analysis of restaurant C")
	print("press 4 for analysis  of all restaurants")

	weekday_count=7
	default_quantity_rest=pd.read_csv('default_quantity_vendor.csv')
	default_quantity_rest_tomato=default_quantity_rest.iloc[0,1]

	default_quantity_rest_onion=default_quantity_rest.iloc[1,1]

	default_quantity_rest_capsicum=default_quantity_rest.iloc[2,1]

	default_quantity_rest_bread=default_quantity_rest.iloc[3,1]

	default_quantity_rest_dough=default_quantity_rest.iloc[4,1]

	default_quantity_rest_chicken=default_quantity_rest.iloc[5,1]

	default_quantity_rest_cheese=default_quantity_rest.iloc[6,1]

	default_quantity_rest_corn=default_quantity_rest.iloc[7,1]

	default_quantity_rest_rava=default_quantity_rest.iloc[8,1]

	default_quantity_rest_sabudana=default_quantity_rest.iloc[9,1]

	default_quantity_rest_masala=default_quantity_rest.iloc[10,1]

	default_quantity_rest_vegetables=default_quantity_rest.iloc[11,1]

	default_quantity_rest_dal=default_quantity_rest.iloc[12,1]

	default_quantity_rest_flour=default_quantity_rest.iloc[13,1]

	default_quantity_rest_rice=default_quantity_rest.iloc[14,1]

	default_quantity_rest_papad=default_quantity_rest.iloc[15,1]

	default_quantity_rest_butter=default_quantity_rest.iloc[16,1]


	choice_input=int(input("Enter your choice :"))


	if(choice_input==1):
		total_tom_A_daily=(total_tom_A/weekday_count)
		total_tom_A_daily=(total_tom_A_daily/default_quantity_rest_tomato)
		total_tom_A_daily=math.ceil(total_tom_A_daily)
		
		total_onion_A_daily=(total_onion_A/weekday_count)
		total_onion_A_daily=(total_onion_A_daily/default_quantity_rest_onion)
		total_onion_A_daily=math.ceil(total_onion_A_daily)
		
		total_capsicum_A_daily=(total_capsicum_A/weekday_count)
		total_capsicum_A_daily=(total_capsicum_A_daily/default_quantity_rest_capsicum)
		total_capsicum_A_daily=math.ceil(total_capsicum_A_daily)
		
		bread_input_daily_A=total_bread_A/weekday_count
		bread_input_daily_A=(bread_input_daily_A/default_quantity_rest_bread)
		bread_input_daily_A=math.ceil(bread_input_daily_A)
		
		dough_input_A=(total_dough_A/default_quantity_rest_dough)
		dough_input_A=math.ceil(dough_input_A)
		
		chicken_input_daily_A=(total_chicken_A/default_quantity_rest_chicken)
		chicken_input_daily_A=math.ceil(chicken_input_daily_A)
		
		cheese_packets_weekly_A=(total_cheese_A/default_quantity_rest_cheese)
		cheese_packets_weekly_A=math.ceil(cheese_packets_weekly_A)
		
		corn_packets_weekly_A=(total_corn_A/default_quantity_rest_corn)
		corn_packets_weekly_A=math.ceil(corn_packets_weekly_A)
		
		rava_kg_weekly_A=(total_rava_A/default_quantity_rest_rava)
		rava_kg_weekly_A=math.ceil(rava_kg_weekly_A)
		
		sabu_kg_weekly_A=(total_sabu_A/default_quantity_rest_sabudana)
		sabu_kg_weekly_A=math.ceil(sabu_kg_weekly_A)
		
		masala_packets_weekly_A=(total_masala_A/default_quantity_rest_masala)
		masala_packets_weekly_A=math.ceil(masala_packets_weekly_A)
		
		vegetables_daily_A=(total_veggies_A/weekday_count)
		vegetables_daily_A=(vegetables_daily_A/default_quantity_rest_vegetables)
		vegetables_daily_A=math.ceil(vegetables_daily_A)
		
		dal_A_weekly=(total_dal_A/default_quantity_rest_dal)
		dal_A_weekly=math.ceil(dal_A_weekly)
		
		flour_A_weekly=(total_flour_A/default_quantity_rest_flour)
		flour_A_weekly=math.ceil(flour_A_weekly)
		
		rice_A_weekly=(total_rice_A/default_quantity_rest_rice)
		rice_A_weekly=math.ceil(rice_A_weekly)
		
		papad_packets_A_weekly=(total_papad_A/default_quantity_rest_papad)
		papad_packets_A_weekly=math.ceil(papad_packets_A_weekly)
		
		butter_packets_weekly_A=(total_butter_A/default_quantity_rest_butter)
		butter_packets_weekly_A=math.ceil(butter_packets_weekly_A)
		
		print()
		print("Total tomatos required daily in kg for restaurant A are :",total_tom_A_daily)
		print("Total Onions required weekly for restaurant A in kg are :",total_onion_A_daily)
		print("Total capsicum required weekly for restaurant A are :",total_capsicum_A_daily)
		print("Total bread packets required daily for restaurant A are :",bread_input_daily_A)
		print("Total kgs of dough required daily for restaurant A is",dough_input_A)
		print("Total kg of chicken required for restaurant A is",chicken_input_daily_A)
		print("Total cheese packets required weekly for restaurant A are",cheese_packets_weekly_A)
		print("Total corn required in kgs for restaurant A is",corn_packets_weekly_A)
		print("Total kg of rava required for restaurant A is",rava_kg_weekly_A)
		print("Total kg of sabudana required for restaurant A is",sabu_kg_weekly_A)
		print("Total packets of masala required for restaurant A is",masala_packets_weekly_A)
		print("Total kg of vegetables daily required for restauarnt A is ",vegetables_daily_A)
		print("Total amount of dal required weekly in kgs is ",dal_A_weekly)
		print("Total flour required weekly in kgs for restauarnt A  is ",flour_A_weekly)
		print("Total rice required weekly in  kgs for restauarnt A  is ",rice_A_weekly)
		print("Total packets of papad required in 1 week for restaurant A is",papad_packets_A_weekly)
		print("Butter packets required weekly for restaurant B are",butter_packets_weekly_A)


	elif(choice_input==2):
		print()
		total_tom_B_daily=(total_tom_B/weekday_count)
		total_tom_B_daily=(total_tom_B_daily/default_quantity_rest_tomato)
		total_tom_B_daily=math.ceil(total_tom_B_daily)
		
		total_onion_B_daily=(total_onion_B/weekday_count)
		total_onion_B_daily=(total_onion_B_daily/default_quantity_rest_onion)
		total_onion_B_daily=math.ceil(total_onion_B_daily)
		
		total_capsicum_B_daily=(total_capsicum_B/weekday_count)
		total_capsicum_B_daily=(total_capsicum_B_daily/default_quantity_rest_capsicum)
		total_capsicum_B_daily=math.ceil(total_capsicum_B_daily)
		
		bread_input_daily_B=total_bread_B/weekday_count
		bread_input_daily_B=(bread_input_daily_B/default_quantity_rest_bread)
		bread_input_daily_B=math.ceil(bread_input_daily_B)
		
		dough_input_B=(total_dough_B/default_quantity_rest_dough)
		dough_input_B=math.ceil(dough_input_B)

		chicken_input_daily_B=(total_chicken_B/default_quantity_rest_chicken)
		chicken_input_daily_B=math.ceil(chicken_input_daily_B)
		
		cheese_packets_weekly_B=(total_cheese_B/default_quantity_rest_cheese)
		cheese_packets_weekly_B=math.ceil(cheese_packets_weekly_B)
		
		corn_packets_weekly_B=(total_corn_B/default_quantity_rest_corn)
		corn_packets_weekly_B=math.ceil(corn_packets_weekly_B)
		
		rava_kg_weekly_B=(total_rava_B/default_quantity_rest_rava)
		rava_kg_weekly_B=math.ceil(rava_kg_weekly_B)
		
		sabu_kg_weekly_B=(total_sabu_B/default_quantity_rest_sabudana)
		sabu_kg_weekly_B=math.ceil(sabu_kg_weekly_B)
		
		masala_packets_weekly_B=(total_masala_B/default_quantity_rest_masala)
		masala_packets_weekly_B=math.ceil(masala_packets_weekly_B)
		
		vegetables_daily_B=(total_veggies_B/weekday_count)
		vegetables_daily_B=(vegetables_daily_B/default_quantity_rest_vegetables)
		vegetables_daily_B=math.ceil(vegetables_daily_B)
		
		dal_B_weekly=(total_dal_B/default_quantity_rest_dal)
		dal_B_weekly=math.ceil(dal_B_weekly)
		
		flour_B_weekly=(total_flour_B/default_quantity_rest_flour)
		flour_B_weekly=math.ceil(flour_B_weekly)
		
		rice_B_weekly=(total_rice_B/default_quantity_rest_rice)
		rice_B_weekly=math.ceil(rice_B_weekly)
		
		papad_packets_B_weekly=(total_papad_B/default_quantity_rest_papad)
		papad_packets_B_weekly=math.ceil(papad_packets_B_weekly)
		
		butter_packets_weekly_B=(total_butter_B/default_quantity_rest_butter)
		butter_packets_weekly_B=math.ceil(butter_packets_weekly_B)
		
		print()
		print("Total tomatos required daily in kg for restaurant B are :",total_tom_B_daily)
		print("Total Onions required weekly for restaurant B in kg are :",total_onion_B_daily)
		print("Total capsicum required weekly for restaurant B are :",total_capsicum_B)
		print("Total bread packets required daily for restaurant B are :",bread_input_daily_B)
		print("Total kgs of dough required daily for restaurant B is",dough_input_B)
		print("Total kg of chicken required for restaurant B is",chicken_input_daily_B)
		print("Total cheese packets required weekly for restaurant B are",cheese_packets_weekly_B)
		print("Total corn required in kgs for restaurant B is",corn_packets_weekly_B)
		print("Total kg of rava required for restaurant B is",rava_kg_weekly_B)
		print("Total kg of sabudana required for restaurant B is",sabu_kg_weekly_B)
		print("Total packets of masala required for restaurant B is",masala_packets_weekly_B)
		print("Total kg of vegetables daily required for restauarnt B is ",vegetables_daily_B)
		print("Total amount of dal required weekly in kgs for restaurant B is ",dal_B_weekly)
		print("Total flour required weekly in kgs for restauarnt B  is ",flour_B_weekly)
		print("Total rice required weekly in  kgs for restauarnt B  is ",rice_B_weekly)
		print("Total packets of papad required in 1 week for restaurant B is",papad_packets_B_weekly)
		print("Butter packets required weekly for restaurant B are",butter_packets_weekly_B)
		
	elif(choice_input==3):
		print()
		
		total_tom_C_daily=(total_tom_C/weekday_count)
		total_tom_C_daily=(total_tom_C_daily/default_quantity_rest_tomato)
		total_tom_C_daily=math.ceil(total_tom_C_daily)
		
		total_onion_C_daily=(total_onion_C/weekday_count)
		total_onion_C_daily=(total_onion_C_daily/default_quantity_rest_onion)
		total_onion_C_daily=math.ceil(total_onion_C_daily)
		
		total_capsicum_C_daily=(total_capsicum_C/weekday_count)
		total_capsicum_C_daily=(total_capsicum_C_daily/default_quantity_rest_capsicum)
		total_capsicum_C_daily=math.ceil(total_capsicum_C_daily)
		
		bread_input_daily_C=total_bread_C/weekday_count
		bread_input_daily_C=(bread_input_daily_C/default_quantity_rest_bread)
		bread_input_daily_C=math.ceil(bread_input_daily_C)
		
		dough_input_C=(total_dough_C/default_quantity_rest_dough)
		dough_input_C=math.ceil(dough_input_C)

		chicken_input_daily_C=(total_chicken_C/default_quantity_rest_chicken)
		chicken_input_daily_C=math.ceil(chicken_input_daily_C)
		
		cheese_packets_weekly_C=(total_cheese_C/default_quantity_rest_cheese)
		cheese_packets_weekly_C=math.ceil(cheese_packets_weekly_C)
		
		corn_packets_weekly_C=(total_corn_C/default_quantity_rest_corn)
		corn_packets_weekly_C=math.ceil(corn_packets_weekly_C)
		
		rava_kg_weekly_C=(total_rava_C/default_quantity_rest_rava)
		rava_kg_weekly_C=math.ceil(rava_kg_weekly_C)
		
		sabu_kg_weekly_C=(total_sabu_C/default_quantity_rest_sabudana)
		sabu_kg_weekly_C=math.ceil(sabu_kg_weekly_C)
		
		masala_packets_weekly_C=(total_masala_C/default_quantity_rest_masala)
		masala_packets_weekly_C=math.ceil(masala_packets_weekly_C)
		
		vegetables_daily_C=(total_veggies_C/weekday_count)
		vegetables_daily_C=(vegetables_daily_C/default_quantity_rest_vegetables)
		vegetables_daily_C=math.ceil(vegetables_daily_C)
		
		dal_C_weekly=(total_dal_C/default_quantity_rest_dal)
		dal_C_weekly=math.ceil(dal_C_weekly)
		
		flour_C_weekly=(total_flour_C/default_quantity_rest_flour)
		flour_C_weekly=math.ceil(flour_C_weekly)
		
		rice_C_weekly=(total_rice_C/default_quantity_rest_rice)
		rice_C_weekly=math.ceil(rice_C_weekly)
		
		papad_packets_C_weekly=(total_papad_C/default_quantity_rest_papad)
		papad_packets_C_weekly=math.ceil(papad_packets_C_weekly)
		
		butter_packets_weekly_C=(total_butter_C/default_quantity_rest_butter)
		butter_packets_weekly_C=math.ceil(butter_packets_weekly_C)
		
		print()
		print("Total tomatos required daily in kg for restaurant C are :",total_tom_C_daily)
		print("Total Onions required weekly for restaurant C in kg are :",total_onion_C_daily)
		print("Total capsicum required weekly for restaurant C are :",total_capsicum_C)
		print("Total bread packets required daily for restaurant C are :",bread_input_daily_C)
		print("Total kgs of dough required daily for restaurant C is",dough_input_C)
		print("Total kg of chicken required for restaurant C is",chicken_input_daily_C)
		print("Total cheese packets required weekly for restaurant C are",cheese_packets_weekly_C)
		print("Total corn required in kgs for restaurant C is",corn_packets_weekly_C)
		print("Total kg of rava required for restaurant C is",rava_kg_weekly_C)
		print("Total kg of sabudana required for restaurant C is",sabu_kg_weekly_C)
		print("Total packets of masala required for restaurant C is",masala_packets_weekly_C)
		print("Total kg of vegetables daily required for restauarnt C is ",vegetables_daily_C)
		print("Total amount of dal required weekly in kgs for restaurant C is ",dal_C_weekly)
		print("Total flour required weekly in kgs for restauarnt C is ",flour_C_weekly)
		print("Total rice required weekly in  kgs for restauarnt C is ",rice_C_weekly)
		print("Total packets of papad required in 1 week for restaurant C is",papad_packets_C_weekly)
		print("Butter packets required weekly for restaurant C are",butter_packets_weekly_C)
		
	elif(choice_input==4):	
		print()
		total_tom_vendor_daily=(total_tomato_vendor/weekday_count)
		total_tom_vendor_daily=(total_tom_vendor_daily/default_quantity_rest_tomato)
		total_tom_vendor_daily=math.ceil(total_tom_vendor_daily)
		
		total_onion_vendor_daily=(total_onion_vendor/weekday_count)
		total_onion_vendor_daily=(total_onion_vendor_daily/default_quantity_rest_onion)
		total_onion_vendor_daily=math.ceil(total_onion_vendor_daily)
		
		total_capsicum_vendor_daily=(total_capsicum_vendor/weekday_count)
		total_capsicum_vendor_daily=(total_capsicum_vendor_daily/default_quantity_rest_capsicum)
		total_capsicum_vendor_daily=math.ceil(total_capsicum_vendor_daily)
		
		bread_input_daily_vendor=total_bread_vendor/weekday_count
		bread_input_daily_vendor=(bread_input_daily_vendor/default_quantity_rest_bread)
		bread_input_daily_vendor=math.ceil(bread_input_daily_vendor)
		
		dough_input_vendor=(total_dough_vendor/default_quantity_rest_dough)
		dough_input_vendor=math.ceil(dough_input_vendor)

		chicken_input_daily_vendor=(total_chicken_vendor/default_quantity_rest_chicken)
		chicken_input_daily_vendor=math.ceil(chicken_input_daily_vendor)
		
		cheese_packets_weekly_vendor=(total_cheese_vendor/default_quantity_rest_cheese)
		cheese_packets_weekly_vendor=math.ceil(cheese_packets_weekly_vendor)
		
		corn_packets_weekly_vendor=(total_corn_vendor/default_quantity_rest_corn)
		corn_packets_weekly_vendor=math.ceil(corn_packets_weekly_vendor)
		
		rava_kg_weekly_vendor=(total_rava_vendor/default_quantity_rest_rava)
		rava_kg_weekly_vendor=math.ceil(rava_kg_weekly_vendor)
		
		sabu_kg_weekly_vendor=(total_sabudana_vendor/default_quantity_rest_sabudana)
		sabu_kg_weekly_vendor=math.ceil(sabu_kg_weekly_vendor)
		
		masala_packets_weekly_vendor=(total_masala_vendor/default_quantity_rest_masala)
		masala_packets_weekly_vendor=math.ceil(masala_packets_weekly_vendor)
		
		vegetables_daily_vendor=(total_vegetables_vendor/weekday_count)
		vegetables_daily_vendor=(vegetables_daily_vendor/default_quantity_rest_vegetables)
		vegetables_daily_vendor=math.ceil(vegetables_daily_vendor)
		
		dal_vendor_weekly=(total_dal_vendor/default_quantity_rest_dal)
		dal_vendor_weekly=math.ceil(dal_vendor_weekly)
		
		flour_vendor_weekly=(total_flour_vendor/default_quantity_rest_flour)
		flour_vendor_weekly=math.ceil(flour_vendor_weekly)
		
		rice_vendor_weekly=(total_rice_vendor/default_quantity_rest_rice)
		rice_vendor_weekly=math.ceil(rice_vendor_weekly)
		
		papad_packets_vendor_weekly=(total_papad_vendor/default_quantity_rest_papad)
		papad_packets_vendor_weekly=math.ceil(papad_packets_vendor_weekly)
		
		butter_packets_weekly_vendor=(total_butter_vendor/default_quantity_rest_butter)
		butter_packets_weekly_vendor=math.ceil(butter_packets_weekly_vendor)
		
		print()
		print("Total tomatos required daily in kg for vendor are :",total_tom_vendor_daily)
		print("Total Onions required weekly for vendor in kg are :",total_onion_vendor_daily)
		print("Total capsicum required weekly for vendor are :",total_capsicum_vendor_daily)
		print("Total bread packets required daily for vendor are :",bread_input_daily_vendor)
		print("Total kgs of dough required daily for vendoris",dough_input_vendor)
		print("Total kg of chicken required for vendor is",chicken_input_daily_vendor)
		print("Total cheese packets required weekly for vendor are",cheese_packets_weekly_vendor)
		print("Total corn required in kgs for vendor is",corn_packets_weekly_vendor)
		print("Total kg of rava required for vendor is",rava_kg_weekly_vendor)
		print("Total kg of sabudana required for vendor is",sabu_kg_weekly_vendor)
		print("Total packets of masala required for vendor is",masala_packets_weekly_vendor)
		print("Total kg of vegetables daily required for vendor is ",vegetables_daily_vendor)
		print("Total amount of dal required weekly in kgs for vendor is ",dal_vendor_weekly)
		print("Total flour required weekly in kgs for vendor is ",flour_vendor_weekly)
		print("Total rice required weekly in  kgs for vendor is ",rice_vendor_weekly)
		print("Total packets of papad required in 1 week for vendoris",papad_packets_vendor_weekly)
		print("Butter packets required weekly for vendor are",butter_packets_weekly_vendor)
	yes=int(input("Press 1 to continue . Any other key to terminate :"))

print("Thank you!")	
	
	
	
	
	
	
	
	
	
	
	
	
	
	