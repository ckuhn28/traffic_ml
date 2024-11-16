# Traffic Volume Prediction Machine Learning
# IME 565 Midterm, Fall 2024
# Chandler Kuhn

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

# Set up the title and description
# NOTE: Had to do some internet 'deep diving' to make a rainbow header (and then add spaces that would be recognized to space out the colors) <-*Extra credit worthy*
st.header(""":rainbow[**Traffic Volume Prediction**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]""")

st.write("Utilize this Machine Learning application to predict traffic volume.")
st.image('traffic_image.gif')

# Load the pre-trained model
model_pickle = open('traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the traffic datasets
traffic_df = pd.read_csv('Traffic_Volume_mod.csv') # updated dataset created by the .ipynb file with the proper date related columns
traffic_sample = pd.read_csv('traffic_data_user.csv')

# Create a sidebar for inputs
st.sidebar.image('traffic_sidebar.jpg', caption="Traffic Volume Predictor")
st.sidebar.write('### Input Features')
st.sidebar.write('Upload data file or manually input Traffic Volume Factors:')

# Option to upload a CSV file
with st.sidebar.expander('Option 1: Upload CSV File', expanded=False):
    uploaded_file = st.file_uploader("Upload a CSV file containing traffic details:", type=["csv"])
    st.write('Sample CSV Format for Upload:')  
    st.write(traffic_sample.head(5))
    st.warning('Ensure your data has the same column names and data types as above.')  
# Option to fill out a form
with st.sidebar.expander('Option 2: Fill out Form', expanded=False):
    st.write('Enter the traffic details manually using the form below:')
    with st.form(key='traffic_form'):
        holiday = st.selectbox('Is it a holiday?', options = [None, 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Christmas Day', 'New Years Day', 'Washingtons Birthday', 'Memorial Day', 'Independence Day', 'State Fair', 'Labor Day', 'Martin Luther King Jr Day'], help="Select the holiday")
        temp = st.number_input('Temperature (K)', min_value=traffic_df['temp'].min(), max_value=traffic_df['temp'].max(), value=281.21, step=0.01)
        rain = st.number_input('Rain in Hour (mm)', min_value=traffic_df['rain_1h'].min(), max_value=traffic_df['rain_1h'].max(), value=0.33, step=0.01)
        snow = st.number_input('Snow in Hour (mm)', min_value=traffic_df['snow_1h'].min(), max_value=traffic_df['snow_1h'].max(), value=0.0, step=0.01)
        clouds = st.number_input('Clouds (%)', min_value=traffic_df['clouds_all'].min(), max_value=traffic_df['clouds_all'].max(), value=49, step=1)
        weather = st.selectbox('Current Weather Type', options = ['Clouds', 'Clear', 'Rain', 'Mist', 'Snow', 'Drizzle', 'Haze', 'Fog', 'Thunderstorm', 'Smoke', 'Squall'], help="Select the weather type")
        month = st.selectbox('Choose Month', options = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], help="Select the month of the year")
        weekday = st.selectbox('Choose Day of the Week', options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], help="Select the day of the week")
        hour = st.selectbox('Choose Hour', options=[str(i) for i in range(0, 24)], help="Select the hour of the day")
        submit_button = st.form_submit_button(label='Submit')
# alpha = None
if submit_button:
    st.success('Form Data submitted successfully.', icon=':material/verified:')
elif uploaded_file is not None:
    st.success('CSV file successfully uploaded.', icon=':material/verified:')
else:
    # if alpha is None:
    st.info('Please choose a data input method to proceed.', icon=':material/data_info_alert:')

# If a CSV is uploaded, read and use it for inputs instead of sliders
if uploaded_file is None: # and alpha is not None:
    alpha = st.slider("Select the confidence level:", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Select the confidence level for the prediction intervals")
    alpha_val = alpha
    encode_df = traffic_df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume'])
    encode_df['hour'] = encode_df['hour'].astype(str)

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain, snow, clouds, weather, month, weekday, str(hour)]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Get the prediction with its intervals
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha_val)
    pred_value = prediction[0]
    lower_limit = intervals[:, 0][0][0]
    upper_limit = intervals[:, 1][0][0]

    # Ensure limits are non-zero
    lower_limit = max(0, lower_limit) # ensure lower limit is non-negative
    upper_limit = max(0, upper_limit) # ensure upper limit is non-negative

    st.write("## Predicting Traffic Volume...")
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value:.0f}")
    st.write(f"Using a **{(1-alpha_val)*100}% Prediction Interval**: [{lower_limit:.0f}, {upper_limit:.0f}]")
elif uploaded_file is not None: # and alpha is not None:
    input_df = pd.read_csv(uploaded_file)
    alpha = st.slider("Select the confidence level:", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Select the confidence level for the prediction intervals")
    alpha_val = alpha

    for index, row in input_df.iterrows():
        holiday = row['holiday']
        temp = row['temp']
        rain = row['rain_1h']
        snow = row['snow_1h']
        clouds = row['clouds_all']
        weather = row['weather_main']
        month = row['month']
        weekday = row['weekday']
        hour = row['hour']
        
        # Encode the inputs for model prediction
        encode_df = traffic_df.copy()
        encode_df = encode_df.drop(columns=['traffic_volume'])
        encode_df['hour'] = encode_df['hour'].astype(str)

        # Combine the list of user data as a row to default_df
        encode_df.loc[len(encode_df)] = [holiday, temp, rain, snow, clouds, weather, month, weekday, str(hour)]

        # Create dummies for encode_df
        encode_dummy_df = pd.get_dummies(encode_df)

        # Extract encoded user data
        user_encoded_df = encode_dummy_df.tail(1)

        # Get the prediction with its intervals
        prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha_val)
        pred_value = prediction[0]
        lower_limit = intervals[:, 0][0][0]
        upper_limit = intervals[:, 1][0][0]

        # Ensure limits are non-zero
        lower_limit = max(0, lower_limit) # ensure lower limit is non-negative
        upper_limit = max(0, upper_limit) # ensure upper limit is non-negative

        input_df.at[index, 'Predicted Volume'] = f"{pred_value:.0f}"
        input_df.at[index, 'Lower Volume Limit'] = f"{lower_limit:.0f}"
        input_df.at[index, 'Upper Volume Limit'] = f"{upper_limit:.0f}"
  
    st.write(f"## Prediction Results with a {(1-alpha_val)*100}% Confidence Interval")
    st.write(input_df)

# Additional tabs for DT model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")

# password_guess = st.text_input("What is the Password?")
# if password_guess != st.secrets["ckuhnadmissions"]:
#  st.stop()