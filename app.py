from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
import asyncio
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from bulk_url_processor import URLFeatureExtractor  
from deployment_pipeline import MLModelPipeline
from sklearn.ensemble import RandomForestClassifier
from pre_processing_pipeline import PreprocessingPipeline

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'pamsy'  # Required for flash messages
preprocessor = PreprocessingPipeline()
# Load the trained model
try:
    model = joblib.load('models/xgboost_model.joblib')
except Exception as e:
    print(f"Error loading model or pipeline: {e}")
    model = None

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')

# Define the URL detection route
@app.route('/url_detection')
def url_detection():
    return render_template('url_detection.html')

# Define the route to check the URL
@app.route('/check_url', methods=['POST'])
async def check_url():
    app.logger.info("POST request received at /check_url")
    if request.method == 'POST':
        url = request.form.get('url')
        app.logger.info(f"URL received: {url}")
        if not url:
            flash("Please enter a valid URL.", "error")
            app.logger.warning("No URL provided.")
            return redirect(url_for('url_detection'))
        
        try:
            app.logger.info("Initializing feature extractor.")
            extractor = URLFeatureExtractor(url)
            
            app.logger.info("Extracting features.")
            features = await extractor.extract_all_features()
            app.logger.info(f"Features extracted: {features}")
            
            if not features:
                flash("Failed to extract features from the URL.", "error")
                app.logger.error("Feature extraction returned no features.")
                return redirect(url_for('url_detection'))
            
            features_df = pd.DataFrame([features])
            app.logger.info(f"Features DataFrame: {features_df}")
            
            if model is None:
                flash("Model is not loaded. Please check the server configuration.", "error")
                app.logger.error("Model is not loaded.")
                return redirect(url_for('url_detection'))
            
            prediction = model.predict(features_df)[0]
            app.logger.info(f"Model prediction: {prediction}")
            
            result = 'Phishing URL' if prediction == 1 else 'Legitimate URL'
            app.logger.info(f"Result: {result}")
            
            return render_template('url_detection.html', url=url, result=result)
        
        except Exception as e:
            app.logger.error(f"Exception occurred: {e}")
            flash("An error occurred while processing the URL.", "error")
            return redirect(url_for('url_detection'))

    return redirect(url_for('url_detection'))

    print(result)
    
# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

