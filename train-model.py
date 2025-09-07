# train_model.py - CORRECTED VERSION
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
from statsmodels.tsa.arima.model import ARIMA
import joblib
import logging
from datetime import datetime
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InventoryForecaster:
    def __init__(self, project_id, service_account_path="service-account.json"):
        self.project_id = project_id
        # Use your service account file for authentication
        self.client = bigquery.Client.from_service_account_json(service_account_path, project=project_id)
        self.storage_client = storage.Client.from_service_account_json(service_account_path, project=project_id)
        
    def extract_data(self):
        """Get data from BigQuery - same as your notebook"""
        logger.info("Getting data from BigQuery...")
        
        # These are the same queries from your notebook
        places_query = "SELECT * FROM LovingLoyaltyDB.dim_places"
        o_query = "SELECT * FROM LovingLoyaltyDB.fct_orders"
        oi_query = "SELECT * FROM LovingLoyaltyDB.fct_orders_items"
        
        places_df = self.client.query(places_query).to_dataframe()
        o_df = self.client.query(o_query).to_dataframe()
        oi_df = self.client.query(oi_query).to_dataframe()
        
        return places_df, o_df, oi_df
    
    def preprocess_data(self, places_df, o_df, oi_df):
        """Clean the data - same as your notebook"""
        logger.info("Cleaning the data...")
        
        # Same cleaning steps from your notebook
        places_df = places_df[['id', 'title', 'latitude', 'longitude']]
        o_df = o_df[['id', 'created', 'place_id', 'status']]
        oi_df = oi_df[['id', 'created', 'title', 'item_id', 'order_id', 'status']]
        
        # Filter fulfilled orders only
        oi_filtered_df = oi_df[oi_df['status'] == 'Fulfilled']
        
        # Convert time to proper format
        o_df['created'] = pd.to_datetime(o_df['created'], unit='s')
        oi_df['created'] = pd.to_datetime(oi_df['created'], unit='s')
        
        # Join the data together
        orders_with_items = pd.merge(
            o_df, oi_df, left_on='id', right_on='order_id', suffixes=('_order', '_item')
        )
        enriched_data = pd.merge(
            orders_with_items, places_df, left_on='place_id', right_on='id', suffixes=('', '_place')
        )
        
        # Group by week (same as your notebook)
        enriched_data['week'] = enriched_data['created_order'].dt.to_period('W')
        aggregated_data = (
            enriched_data.groupby(['place_id', 'item_id', 'week'])
            .size()
            .reset_index(name='total_orders')
        )
        aggregated_data['week_starting'] = aggregated_data['week'].apply(lambda x: x.start_time)
        
        return aggregated_data
    
    def train_models(self, aggregated_data, min_observations=20):
        """Train ARIMA models for each place-item combination"""
        logger.info("Training ARIMA models...")
        
        # Same pivoting as your notebook
        pivoted_data = aggregated_data.pivot_table(
            index='week_starting', columns=['place_id', 'item_id'], 
            values='total_orders', fill_value=0
        )
        
        models = {}
        model_info = {}
        
        # Train a model for each place-item combination
        for (place_id, item_id) in pivoted_data.columns:
            ts = pivoted_data[(place_id, item_id)]
            
            # Skip if not enough data
            if len(ts[ts > 0]) < min_observations:
                continue
                
            try:
                # Train ARIMA model (same as your notebook)
                model = ARIMA(ts, order=(2, 1, 2))
                model_fit = model.fit()
                
                # Save the trained model
                models[f"{place_id}_{item_id}"] = model_fit
                model_info[f"{place_id}_{item_id}"] = {
                    'place_id': place_id,
                    'item_id': item_id,
                    'aic': model_fit.aic,
                    'training_data_points': len(ts),
                    'last_updated': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Trained model for place {place_id}, item {item_id}")
                
            except Exception as e:
                logger.warning(f"❌ Failed to train model for place {place_id}, item {item_id}: {e}")
                continue
        
        return models, model_info
    
    def save_models(self, models, model_info, bucket_name):
        """Save models to Google Cloud Storage - CORRECTED VERSION"""
        logger.info(f"Saving {len(models)} models to Cloud Storage...")
        
        bucket = self.storage_client.bucket(bucket_name)
        
        # Save each model
        for model_key, model in models.items():
            model_blob = bucket.blob(f"inventory_models/{model_key}.pkl")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Save model to the temporary file
                joblib.dump(model, temp_filename)
                # Upload the file to cloud storage
                model_blob.upload_from_filename(temp_filename)
                logger.info(f"✅ Saved model {model_key}")
            finally:
                # Always clean up temp file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
        
        # Save model information
        metadata_blob = bucket.blob("inventory_models/metadata.pkl")
        
        # Create a temporary file for metadata
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Save metadata to the temporary file
            joblib.dump(model_info, temp_filename)
            # Upload the file to cloud storage
            metadata_blob.upload_from_filename(temp_filename)
            logger.info("✅ Saved model metadata")
        finally:
            # Always clean up temp file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        
        logger.info("✅ All models saved successfully!")

# This runs when you execute the file
if __name__ == "__main__":
    # CHANGE THESE TO YOUR VALUES
    PROJECT_ID = "nse-gcp-ema-tt-0eee5-sbx-1"  # Replace with your project ID
    BUCKET_NAME = f"{PROJECT_ID}-ml-models"  # We'll create this bucket
    
    print("🚀 Starting model training...")
    forecaster = InventoryForecaster(PROJECT_ID)
    
    # Step 1: Get data
    places_df, o_df, oi_df = forecaster.extract_data()
    print(f"📊 Got data: {len(places_df)} places, {len(o_df)} orders, {len(oi_df)} order items")
    
    # Step 2: Clean data
    aggregated_data = forecaster.preprocess_data(places_df, o_df, oi_df)
    print(f"🧹 Cleaned data: {len(aggregated_data)} records")
    
    # Step 3: Train models
    models, metadata = forecaster.train_models(aggregated_data)
    print(f"🤖 Trained {len(models)} models")
    
    # Step 4: Save models
    forecaster.save_models(models, metadata, BUCKET_NAME)
    print("✅ Done! Your models are ready for deployment.")