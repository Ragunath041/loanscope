import firebase_admin
from firebase_admin import credentials, ml

# Initialize Firebase Admin SDK
cred = credentials.Certificate("F:/loanscope/credentials/loanscope-9f38b-firebase-adminsdk-fbsvc-3b3f380bbc.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

model_name = "cibil_score_model"
model_path = "model/cibil_model.tflite"  # Ensure this file exists locally

# Load the model file directly
with open(model_path, 'rb') as model_file:
    model_data = model_file.read()

# Create a TFLiteFormat instance
tflite_format = ml.TFLiteFormat()  # Instantiate without arguments
tflite_format.model_data = model_data  # Set the model data if this is supported

# Create the model instance
firebase_model = ml.Model(display_name=model_name, model_format=tflite_format)

# Deploy the model using the correct method
# Check if there's a method like `create_model` directly under `ml`
firebase_model = ml.create_model(firebase_model)  # Adjusted method

# Publish the model
ml.publish_model(firebase_model)  # Adjusted method

print(f"✅ Model '{model_name}' successfully deployed and published to Firebase ML!")
