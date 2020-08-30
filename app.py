import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
from PIL import Image
image = Image.open('cover.jpg')

df = pd.read_csv("EDA_dp.csv")

# st.write(df.head())

st.image(image, use_column_width=True) 

st.title('Data Scientist Salary Predictor')

st.markdown('<style>h1{color: brown;}</style>', unsafe_allow_html=True)

st.info("This Web Application is created and maintained by *_DHEERAJ_ _KUMAR_ _K_*")

st.subheader('Company Details \n Data Scrapped from GlassDoor ')

Rating = st.slider('Glassdoor Rating of the Company',min_value=1.0, max_value=5.0, step=0.1)

# st.subheader('Number of Competitors')

No_of_Competitors = st.slider('Number of Competitors',min_value=0.0, max_value=4.0, step=1.0)

# st.subheader('Company Age')

Age = st.slider('Age of the Company', step=1.0, min_value=0.0,max_value=330.0)

## description length
# st.subheader("Choose Approximate length of job description")

desc_length = st.slider('Choose Approximate length of job description',min_value=110.0, max_value=15121.0, step=10.0)

# st.subheader("Choose Revenue of the company")
Revenue_1,Revenue_Less_than_10_million,Revenue_Unknown_Non_Applicable, Revenue_1_to_5_billion,Revenue_10_to_50_billion, Revenue_10_to_50_million,Revenue_100_to_500_billion,Revenue_100_to_500_million, Revenue_5_to_10_billion,Revenue_50_to_100_billion,Revenue_50_to_100_million,Revenue_500_million_to_1_billion,Revenue_500_billion = 0,0,0,0,0,0,0,0,0,0,0,0,0

list_of_revenue = list(df['Revenue'].unique())
for i in range(len(list_of_revenue)):
    if list_of_revenue[i] == "-1":
        list_of_revenue[i] = "₹1 to ₹5 billion (INR)"
        break

revenue = st.selectbox('Revenue', list_of_revenue,index=0)

# st.write(list(df['Revenue'].unique()))

if revenue == "₹1 to ₹5 million (INR)":
    Revenue_1 = 1
elif revenue =="Less than ₹10 million (INR)":
    Revenue_Less_than_10_million = 1
elif revenue == "Unknown / Non-Applicable":
    Revenue_Unknown_Non_Applicable = 1 #dont use and operator
elif revenue == '₹1 to ₹5 billion (INR)':
    Revenue_1_to_5_billion = 1
elif revenue == "₹10 to ₹50 billion (INR)":#5
    Revenue_10_to_50_billion = 1
elif revenue == "₹10 to ₹50 million (INR)":
    Revenue_10_to_50_million = 1
elif revenue == "₹100 to ₹500 billion (INR)":
    Revenue_100_to_500_billion = 1
elif revenue == "₹10 to ₹500 million (INR)":
    Revenue_100_to_500_million = 1
elif revenue == "₹5 to ₹10 billion (INR)":
    Revenue_5_to_10_billion = 1
elif revenue == "₹50 to ₹100 billion (INR)":
    Revenue_50_to_100_billion = 1
elif revenue == "₹50 to ₹100 million (INR)":
    Revenue_50_to_100_million = 1
elif revenue == "₹500 million to ₹1 billion (INR)":
    Revenue_500_million_to_1_billion = 1
elif revenue == "₹500+ billion (INR)":#13
    Revenue_500_billion = 1

# st.write(Revenue_1,
#        Revenue_Less_than_10_million,
#        Revenue_Unknown_Non_Applicable, Revenue_1_to_5_billion,
#        Revenue_10_to_50_billion, Revenue_10_to_50_million,
#        Revenue_100_to_500_billion,
#        Revenue_100_to_500_million, Revenue_5_to_10_billion,
#        Revenue_50_to_100_billion,
#        Revenue_50_to_100_million,
#        Revenue_500_million_to_1_billion,
#        Revenue_500_billion)

# st.subheader("Choose size of the company")

Size_1,Size_1_to_50_employees, Size_10000_employees,Size_1001_to_5000_employees,Size_201_to_500_employees,Size_5001_to_10000_employees, Size_501_to_1000_employees,Size_51_to_200_employees, Size_Unknown =0,0,0,0,0,0,0,0,0

list_of_size = list(df['Size'].unique())
for i in range(len(list_of_size)):
    if list_of_size[i] == "-1":
        list_of_size[i] = "N/A"
        break

# st.write(list_of_size)

size = st.selectbox('Company Size',list_of_size,index=0)

# st.write(list(df['Size'].unique()))

if  size == "N/A":#1
    Size_1 = 1
elif size =="1 to 50 employees":
    Size_1_to_50_employees = 1
elif size == "10000+ employees":
    Size_10000_employees = 1 #dont use and operator
elif size == '1001 to 5000 employees':
    Size_1001_to_5000_employees = 1
elif size == "201 to 500 employees":#5
    Size_201_to_500_employees = 1
elif size == "5001 to 10000 employees":
    Size_5001_to_10000_employees = 1
elif size == "501 to 1000 employees":
    Size_501_to_1000_employees = 1
elif size == "51 to 200 employees":
    Size_51_to_200_employees = 1
elif size == "Unknown":#9
    Size_Unknown = 1

# st.write(Size_1,
#   Size_1_to_50_employees, Size_10000_employees,
#            Size_1001_to_5000_employees,Size_201_to_500_employees,
#        Size_5001_to_10000_employees, Size_501_to_1000_employees,
#        Size_51_to_200_employees, Size_Unknown)

# st.subheader("Job Sector")

# st.write(list(df['Sector'].unique()))

Sector_1,Sector_Accounting_Legal, Sector_Aerospace_Defence,Sector_Agriculture_Forestry,Sector_Arts_Entertainment_Recreation, Sector_Biotech_Pharmaceuticals, Sector_Business_Services,Sector_Consumer_Services, Sector_Education, Sector_Finance,Sector_Government, Sector_Healthcare,Sector_Information_Technology, Sector_Insurance,Sector_Manufacturing, Sector_Media, Sector_Non_Profits,Sector_Oil_Gas_Utilities, Sector_Real_Estate,Sector_Restaurants_Food_Service, Sector_Retail,Sector_Telecommunications, Sector_Transportation_Logistics,Sector_Travel_Tourism = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

list_of_sector = list(df['Sector'].unique())
for i in range(len(list_of_sector)):
    if list_of_sector[i] == "-1":
        list_of_sector[i] = "N/A"
        break

sector = st.selectbox('Job Sector', list_of_sector,index=0)
# st.write(list(df['Sector'].unique()))

if sector == "Biotech & Pharmaceuticals":#1
    Sector_Biotech_Pharmaceuticals = 1
elif sector =="Business Services":
    Sector_Business_Services = 1
elif sector == "Aerospace & Defence":
    Sector_Aerospace_Defence = 1 #dont use and operator
elif sector == 'Information Technology':
    Sector_Information_Technology = 1
elif sector == "Insurance":
    Sector_Insurance = 1
elif sector == 'Manufacturing':
    Sector_Manufacturing = 1
elif sector == "Agriculture & Forestry":
    Sector_Agriculture_Forestry = 1
elif sector == "Retail":
    Sector_Retail = 1
elif sector == 'Travel & Tourism':
    Sector_Travel_Tourism = 1
elif sector == "N/A": #10th
    Sector_1 = 1
elif sector == "Finance": #10th
    Sector_Finance = 1
elif sector == "Oil, Gas, Energy & Utilities":
    Sector_Oil_Gas_Utilities = 1
elif sector == "Healthcare":
    Sector_Healthcare = 1
elif sector == "Government":
    Sector_Government = 1
elif sector == "Education":
    Sector_Education = 1
elif sector == "Restaurants, Pubs, Bars & Food Service":
    Sector_Restaurants_Food_Service= 1
elif sector == "Accounting & Legal":
    Sector_Accounting_Legal = 1
elif sector == "Media":
    Sector_Media = 1
elif sector == "Telecommunications":
    Sector_Telecommunications = 1
elif sector == "Transportation & Logistics":
    Sector_Transportation_Logistics = 1
elif sector == "Real Estate":
    Sector_Real_Estate = 1
elif sector == "Arts, Entertainment & Recreation":
    Sector_Arts_Entertainment_Recreation = 1
elif sector == "Consumer Services":
    Sector_Consumer_Services = 1
elif sector == "Non-Profits":
    Sector_Non_Profits = 1

# st.write(Sector_1,Sector_Accounting_Legal, Sector_Aerospace_Defence,Sector_Agriculture_Forestry,
# Sector_Arts_Entertainment_Recreation, Sector_Biotech_Pharmaceuticals, Sector_Business_Services,
# Sector_Consumer_Services, Sector_Education, Sector_Finance,Sector_Government, Sector_Healthcare,
# Sector_Information_Technology, Sector_Insurance,
# Sector_Manufacturing, Sector_Media, Sector_Non_Profits,Sector_Oil_Gas_Utilities, Sector_Real_Estate,Sector_Restaurants_Food_Service, Sector_Retail,Sector_Telecommunications, Sector_Transportation_Logistics,Sector_Travel_Tourism)

# st.write(sector)


st.subheader('Language skills')

python_yn = st.checkbox('Python')
if python_yn:
    python_yn = 1
else:
    python_yn = 0

r = st.checkbox('R')
if r:
    r = 1
else:
    r = 0

sas = st.checkbox('SaS')
if sas:
    sas = 1
else:
    sas = 0

aws = st.checkbox('AWS')
if aws:
    aws = 1
else:
    aws = 0

spark = st.checkbox('Spark')
if spark:
    spark = 1
else:
    spark = 0

hadoop = st.checkbox('Hadoop')
if hadoop:
    hadoop = 1
else:
    hadoop = 0

tensorFlow = st.checkbox('TensorFlow')
if tensorFlow:
    tensorFlow = 1
else:
    tensorFlow = 0

sql = st.checkbox('SQL')
if sql:
    sql = 1
else:
    sql = 0

st.subheader("Web Deployment Skill")

flask = st.checkbox('Flask')
if flask:
    flask = 1
else:
    flask = 0

st.subheader("Statistical Skill")

stats = st.checkbox('Statistics')
if stats:
    stats = 1
else:
    stats = 0

st.subheader("Data Visualization Skills")

tableau = st.checkbox('Tableau')
if tableau:
    tableau = 1
else:
    tableau = 0

powerBI = st.checkbox('PowerBI')
if powerBI:
    powerBI = 1
else:
    powerBI = 0

st.subheader("Other Skills")

excel = st.checkbox('Excel')
if excel:
    excel = 1
else:
    excel = 0

st.subheader("Location Selection")

Job_State_CA,Job_State_MA,Job_State_CT,Job_State_NJ,Job_State_NY,Job_State_WA =0,0,0,0,0,0

state = st.selectbox(
     'States',  ('California', 'Massachusetts', 'Connecticut','Massachusetts','New Jersey','New York','Washington'),index=0)

if state == 'California':
    Job_State_CA = 1
elif state =='Massachusetts':
    Job_State_MA = 1
elif state == 'Connecticut':
    Job_State_CT = 1
elif state == 'New Jersey':
    Job_State_NJ = 1
elif state == 'New York':
    Job_State_NY = 1
elif state == 'Washington':
    Job_State_WA = 1

# st.write(Job_State_CA,Job_State_MA,Job_State_CT,Job_State_NJ,Job_State_NY,Job_State_WA)

Seniority_junior,Seniority_na,Seniority_senior =0,0,0

same_state = st.radio("Would you like to work in the in Same State Headquarters Office ?",('Yes','No'), index=0)

if same_state == 'Yes':
    same_state = 1
elif same_state == 'No':
    same_state = 0

## Job Seniority

st.subheader("Experience Level")

Seniority_junior,Seniority_na,Seniority_senior =0,0,0

seniority = st.radio("Choose Seniority Level",('NA','Junior Level','Senior Level'), index=0)

if seniority == 'Junior Level':
    Seniority_junior = 1
elif seniority == 'NA':
    Seniority_na = 1
elif seniority == 'Senior Level':
    Seniority_senior = 1

# st.info('**NA - Not Applicable')

# st.write(Seniority_junior,Seniority_na,Seniority_senior)

features = [Rating,No_of_Competitors,same_state,Age,r,python_yn,
aws,excel,spark,tableau,sql,tensorFlow,powerBI,
sas,flask,hadoop,stats,desc_length,Size_1,Size_1_to_50_employees, Size_10000_employees,Size_1001_to_5000_employees,Size_201_to_500_employees,
Size_5001_to_10000_employees, Size_501_to_1000_employees,Size_51_to_200_employees, Size_Unknown,Sector_1,Sector_Accounting_Legal,
Sector_Aerospace_Defence,Sector_Agriculture_Forestry,Sector_Arts_Entertainment_Recreation, Sector_Biotech_Pharmaceuticals, Sector_Business_Services,
Sector_Consumer_Services, Sector_Education, Sector_Finance,Sector_Government, Sector_Healthcare,
Sector_Information_Technology, Sector_Insurance,Sector_Manufacturing, Sector_Media, Sector_Non_Profits,Sector_Oil_Gas_Utilities,
Sector_Real_Estate,Sector_Restaurants_Food_Service, Sector_Retail,Sector_Telecommunications, Sector_Transportation_Logistics,Sector_Travel_Tourism,
Revenue_1,
       Revenue_Less_than_10_million,
       Revenue_Unknown_Non_Applicable, Revenue_1_to_5_billion,
       Revenue_10_to_50_billion, Revenue_10_to_50_million,
       Revenue_100_to_500_billion,
       Revenue_100_to_500_million, Revenue_5_to_10_billion,
       Revenue_50_to_100_billion,
       Revenue_50_to_100_million,
       Revenue_500_million_to_1_billion,
       Revenue_500_billion,Job_State_CA,Job_State_MA,Job_State_CT,Job_State_NJ,Job_State_NY,Job_State_WA,
       Seniority_junior,Seniority_na,Seniority_senior]

# st.write(len(features))
float_feature_list = []
for i in features:
    float_feature_list.append(float(i)) 
# st.write(features)
final_features = np.array(float_feature_list).reshape(1, -1)

# model_building
file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

#predict_button call
if st.button('Predict The Salary'):
    prediction = model.predict(final_features)
    st.success(f'Your predicted salary is US$ {round(prediction[0],3)*1000} ')
