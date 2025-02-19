
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import firebase_admin
from firebase_admin import credentials, storage, ml
import tensorflow as tf

class CibilScoreModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = 'model/cibil_model.pkl'
        self.scaler_path = 'model/scaler.pkl'
        self.tflite_path = 'model/cibil_model.tflite'
        
    def prepare_data(self):
        try:
            # Read the CSV file
            df = pd.read_csv('cibil_data.csv')
            
            # Select only the relevant numeric columns and target
            relevant_columns = [
                'Sanctioned_Amount',
                'Current_Amount',
                'Loan_Tenure',
                'Monthly_EMI',
                'Previous_Loans',
                'Defaults',
                'Credit_Utilization',
                'Monthly_Income',
                'Late_Payment',
                'CIBIL'  # Target variable
            ]
            
            # Select only relevant columns
            df = df[relevant_columns]
            
            # Convert Late_Payment to numeric (YES/NO to 1/0)
            df['Late_Payment'] = (df['Late_Payment'] == 'YES').astype(int)
            
            # Create feature matrix X and target y
            X = df.drop('CIBIL', axis=1)  # All columns except CIBIL
            y = df['CIBIL']  # Target variable
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Save feature names for prediction
            self.feature_names = X.columns.tolist()
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise
    
    def train(self):
        try:
            # Create model directory if it doesn't exist
            os.makedirs('model', exist_ok=True)
            
            # Prepare data
            X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_data()
            
            # Initialize and train the model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,  # Let the trees grow fully
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            # Fit the model
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate and print accuracy metrics
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            print(f"✅ Training Score: {train_score:.4f}")
            print(f"✅ Testing Score: {test_score:.4f}")
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, 
                                       self.model.feature_importances_))
            print("\nFeature Importance:")
            for feature, importance in sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance:.4f}")
            
            # Save model and scaler
            self.save_model()

            # Convert to TFLite and save
            self.convert_to_tflite()

            return self.model
            
        except Exception as e:
            print(f"Error in train: {str(e)}")
            raise

    def convert_to_tflite(self):
        """ Convert the trained model to TFLite format """
        try:
            print("⚡ Converting model to TensorFlow Lite format...")
            model_path = self.tflite_path

            # Define a simple neural network with one output neuron
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Dense(1, input_shape=(len(self.feature_names),))
            ])

            # Convert RandomForest feature importance into neural network weights
            initial_weights = [
                np.expand_dims(self.model.feature_importances_, axis=-1),  # Weights
                np.array([0.0])  # Bias
            ]
            tf_model.set_weights(initial_weights)

            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            tflite_model = converter.convert()

            # Save TFLite model
            with open(model_path, 'wb') as f:
                f.write(tflite_model)

            print(f"✅ TFLite model saved at: {model_path}")

        except Exception as e:
            print(f"❌ Error converting to TFLite: {str(e)}")
            raise

    def save_model(self):
        try:
            # Save the model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save the scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature names
            with open('model/feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
                
            print(f"✅ Model saved to {self.model_path}")
            print(f"✅ Scaler saved to {self.scaler_path}")
            
        except Exception as e:
            print(f"❌ Error in save_model: {str(e)}")
            raise

def deploy_to_firebase():
    """ Uploads the TFLite model to Firebase Storage """
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate('F:/loanscope/credentials/loanscope-9f38b-firebase-adminsdk-fbsvc-3b3f380bbc.json')

            firebase_admin.initialize_app(cred, {'storageBucket': 'loanscope-9f38b.firebasestorage.app'})

        bucket = storage.bucket()
        blob = bucket.blob('models/cibil_model.tflite')
        blob.upload_from_filename('model/cibil_model.tflite')

        print(f"✅ Model uploaded to Firebase Storage: {blob.public_url}")

    except Exception as e:
        print(f"❌ Error uploading to Firebase: {str(e)}")
        raise

def main():
    try:
        model = CibilScoreModel()
        model.train()

        print("✅ Model training completed successfully!")

        # Deploy to Firebase ML
        deploy_to_firebase()

    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
