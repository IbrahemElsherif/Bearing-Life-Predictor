import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# Title and description
st.title("Bearing Failure Prediction App")
st.write("This app predicts bearing failure based on various sensor measurements.")

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Create input features section
st.header("Input Features")

# Create columns for input features
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sensor Measurements")
    radial_force = st.number_input("Radial Force", value=5.16)
    rotation_speed = st.number_input("Rotation Speed", value=45.0)
    torque = st.number_input("Torque", value=45.3)
    horizontal_acc = st.number_input("Horizontal Acceleration", value=0.0001)
    vertical_acc = st.number_input("Vertical Acceleration", value=0.0000411)
    temperature = st.number_input("Temperature", value=23.6)

with col2:
    st.subheader("Metadata")
    bearing_id = st.selectbox("Bearing ID", options=[1, 2, 3, 4, 5, 6, 7, 8])
    timestamp = st.date_input("Date", value=datetime.now())
    
    # Convert timestamp to datetime
    timestamp_datetime = datetime.combine(timestamp, datetime.min.time())
    st.write(f"Selected datetime: {timestamp_datetime}")

# Create prediction button
if st.button("Predict Bearing Condition"):
    if model is not None:
        # Create input dataframe
        input_data = pd.DataFrame({
            'BearingID': [bearing_id],
            'Timestamp': [timestamp_datetime],
            'RadialForce': [radial_force],
            'RotationSpeed': [rotation_speed],
            'Torque': [torque],
            'HorizontalAcc': [horizontal_acc],
            'VerticalAcc': [vertical_acc],
            'Temperature': [temperature]
        })
        
        st.subheader("Input Data Preview")
        st.dataframe(input_data)
        
        try:
            # Get the first transformer (DataTransformer)
            data_transformer = None
            for transformer in model.get('_transformers', {}).values():
                if hasattr(transformer, 'transformer_and_mapper_list'):
                    data_transformer = transformer
                    break
            
            if data_transformer:
                # Apply transformations
                transformed_data = data_transformer.transform(input_data)
                
                # Get the actual model (usually stored as part of the AutoML pipeline)
                # Note: This needs to be adapted based on your specific model structure
                prediction_model = data_transformer._get_model_from_pipeline()
                
                # Make prediction
                prediction = prediction_model.predict(transformed_data)
                
                st.subheader("Prediction Results")
                st.write(f"Predicted Value: {prediction[0]:.4f}")
                
                # You might want to add interpretation based on your specific use case
                if prediction[0] > 0.7:  # Example threshold
                    st.error("High risk of bearing failure detected!")
                elif prediction[0] > 0.4:
                    st.warning("Moderate risk of bearing failure.")
                else:
                    st.success("Low risk of bearing failure.")
            else:
                st.error("Could not find the data transformer in the model.")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Model structure may be different than expected. Please check the model architecture.")
    else:
        st.error("Model could not be loaded. Please check the model.pkl file exists.")

# Add information about the model
st.sidebar.header("Model Information")
if model is not None:
    st.sidebar.write("Model type: Azure AutoML")
    st.sidebar.write("Task type: Regression")
    
    if '_transformers' in model and 'datatransformer' in model['_transformers']:
        transformer = model['_transformers']['datatransformer']
        raw_features = getattr(transformer, '_raw_feature_names', [])
        st.sidebar.write(f"Raw features: {raw_features}")
        
        mapper = getattr(transformer, 'mapper', None)
        if mapper:
            transformed_names = getattr(mapper, 'transformed_names_', [])
            st.sidebar.write(f"Transformed features: {transformed_names}")
else:
    st.sidebar.write("Model not loaded")

# Add some documentation
st.sidebar.header("About the App")
st.sidebar.write("""
This application uses an Azure AutoML model to predict bearing failure based on sensor data.
The model includes various preprocessing steps such as:
- StringCast for categorical variables
- CountVectorizer for text features
- SimpleImputer for handling missing values
- DateTimeTransformer for timestamp features

The prediction represents the likelihood or severity of bearing failure.
""")

# Add requirements
st.sidebar.header("Requirements")
st.sidebar.code("""
streamlit
pandas
numpy
pickle
datetime
""")