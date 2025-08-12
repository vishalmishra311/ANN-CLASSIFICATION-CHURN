import streamlit as st
import tensorflow as tf
import  pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler ,  LabelEncoder, OneHotEncoder
import pickle
## streamlit app
st.title('Customer Churn PRediction')
## load the trained 

model = tf.keras.models.load_model("../model.h5")

## load the encoders and scaler
with open("../label_encoder_gender.pkl", "rb") as file:
    lable_encoder_gender = pickle.load(file)
with open("../onehot_encoder_geo.pkl", "rb") as file:
    lable_encoder_geo = pickle.load(file)
with open ("../scaler.pkl",'rb') as file :
    scaler = pickle.load(file)
    
## user input 
geography = st.selectbox('Geography',lable_encoder_geo.categories_[0])
gender = st.selectbox('Gender', lable_encoder_gender.classes_)
age = st.slider("Age",18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary =  st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox ('Is Active Member',[0,1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lable_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# one hot encode geography
geo_encoded = lable_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=lable_encoder_geo.get_feature_names_out(['Geography']))
# combine one -hot encoded columns with input data 
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
# scale the input data 
input_data_scaled = scaler.transform(input_data)

# prediction churn 
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]


st.write(f'churn Probabil:{prediction_prob:.2f}')
if prediction_prob>0.5:
    st.write("The customer is likely to churn")
else:
     st.write("The customer is not likely to churn")

