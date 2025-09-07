# train_model.py - Recommendation Model Training
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
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

class RecommendationTrainer:
    def __init__(self, project_id, service_account_path="service-account.json"):
        self.project_id = project_id
        # Use your service account file for authentication
        self.client = bigquery.Client.from_service_account_json(service_account_path, project=project_id)
        self.storage_client = storage.Client.from_service_account_json(service_account_path, project=project_id)
        
    def extract_data(self):
        """Get data from BigQuery - same pattern as your notebook"""
        logger.info("Getting data from BigQuery...")
        
        # Load catalog data
        catalog_query = f"""
        SELECT
            id AS item_id,
            title,
            price,
            currencyCode,
            availableTime
        FROM `{self.project_id}.LovingLoyaltyDB.vertex_ai_catalog_items_py`
        WHERE price > 0
        """
        
        # Load user events data
        events_query = f"""
        SELECT
            visitorId AS user_id,
            item_id,
            quantity,
            originalPrice,
            age_group,
            area,
            gender,
            country,
            orders_count
        FROM `{self.project_id}.LovingLoyaltyDB.vertex_ai_user_events_py`
        WHERE originalPrice > 0 AND quantity > 0
        """
        
        df_catalog = self.client.query(catalog_query).to_dataframe()
        df_events = self.client.query(events_query).to_dataframe()
        
        return df_catalog, df_events
    
    def get_time_of_day(self, timestamp):
        """Categorize timestamp into time periods"""
        try:
            hour = pd.to_datetime(timestamp, utc=True).hour
            if 5 <= hour < 12:
                return "Morning"
            elif 12 <= hour < 18:
                return "Afternoon"  
            elif 18 <= hour < 24:
                return "Evening"
            else:
                return "Night"
        except Exception:
            return "Unknown"
    
    def preprocess_data(self, df_catalog, df_events):
        """Clean and prepare the data - same as your notebook"""
        logger.info("Cleaning the data...")
        
        # Add time of day feature to catalog
        df_catalog['time_of_day'] = df_catalog['availableTime'].apply(self.get_time_of_day)
        
        # Fill missing values in events
        df_events = df_events.fillna({
            "age_group": "Unknown", 
            "area": "Unknown", 
            "gender": "Unknown", 
            "country": "Unknown",
            "orders_count": 0
        })
        
        # Calculate total spent per transaction
        df_events['total_spent'] = (
            pd.to_numeric(df_events['originalPrice'], errors='coerce').fillna(0) * 
            df_events['quantity'].fillna(0)
        )
        
        # Encode categorical features and store encoders
        label_encoders = {}
        for col in ['age_group', 'area', 'gender', 'country']:
            df_events[col] = df_events[col].astype(str).fillna('Unknown')
            le = LabelEncoder()
            df_events[col] = le.fit_transform(df_events[col])
            label_encoders[col] = le
        
        # Aggregate user-level features
        user_features = df_events.groupby('user_id').agg({
            'age_group': 'first',
            'area': 'first',
            'gender': 'first',
            'country': 'first',
            'orders_count': 'first',
            'total_spent': 'sum'
        }).reset_index()
        
        # Create user-item interaction matrix
        user_item_matrix = (
            df_events.groupby(['user_id', 'item_id'])
            .agg(total_purchases=('quantity', 'sum'))
            .reset_index()
        )
        
        # Pivot to create user-item matrix
        user_item_pivot = user_item_matrix.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='total_purchases', 
            fill_value=0
        )
        
        # Merge user features with item interactions
        features_full = user_features.merge(user_item_pivot, on='user_id', how='left').fillna(0)
        
        # Calculate item popularity and food category mapping
        item_popularity = (
            df_events.groupby('item_id')
            .agg({
                'quantity': 'sum',
                'user_id': 'nunique'
            })
            .rename(columns={'quantity': 'total_purchases', 'user_id': 'unique_users'})
            .reset_index()
        )
        
        # Merge with catalog to get food item details
        item_details = df_catalog.merge(item_popularity, on='item_id', how='left').fillna(0)
        
        return features_full, df_events, item_details, label_encoders
    
    def train_models(self, features_full, df_events, item_details, label_encoders, n_clusters=8):
        """Train clustering model and build recommendations"""
        logger.info("Training K-means clustering model...")
        
        # Prepare features for clustering (exclude user_id)
        X = features_full.drop(columns=['user_id'])
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
        
        # Train K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X_scaled)
        
        # Predict clusters for all users
        features_full['cluster'] = kmeans.predict(X_scaled)
        
        # Create user-cluster mapping
        user_clusters = dict(zip(features_full['user_id'].astype(str), features_full['cluster']))
        
        # Calculate top food items per cluster
        df_events['user_id'] = df_events['user_id'].astype(str)
        features_full['user_id'] = features_full['user_id'].astype(str)
        
        df_events_with_clusters = df_events.merge(
            features_full[['user_id', 'cluster']], 
            on='user_id', 
            how='left'
        )
        
        # Aggregate item popularity by cluster
        cluster_item_popularity = (
            df_events_with_clusters.groupby(['cluster', 'item_id'])
            .agg({
                'quantity': 'sum',
                'user_id': 'nunique'
            })
            .rename(columns={'quantity': 'total_purchases', 'user_id': 'unique_users'})
            .reset_index()
        )
        
        # Calculate popularity score (combination of purchases and unique users)
        cluster_item_popularity['popularity_score'] = (
            cluster_item_popularity['total_purchases'] * 0.7 + 
            cluster_item_popularity['unique_users'] * 0.3
        )
        
        cluster_item_popularity = cluster_item_popularity.sort_values(
            ['cluster', 'popularity_score'], 
            ascending=[True, False]
        )
        
        # Merge with item details to get food information
        cluster_recommendations = cluster_item_popularity.merge(
            item_details[['item_id', 'title', 'price', 'time_of_day']], 
            on='item_id', 
            how='left'
        )
        
        models = {
            'kmeans': kmeans,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'user_clusters': user_clusters,
            'cluster_recommendations': cluster_recommendations,
            'item_details': item_details
        }
        
        model_info = {
            'model_version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'training_date': datetime.now().isoformat(),
            'n_clusters': n_clusters,
            'random_state': 42,
            'total_users': len(user_clusters),
            'total_items': len(item_details),
            'model_path': 'recommendation_models/'
        }
        
        logger.info(f"✅ Trained clustering model with {n_clusters} clusters")
        logger.info(f"📊 {len(user_clusters)} users, {len(item_details)} items")
        
        return models, model_info
    
    def save_models(self, models, model_info, bucket_name):
        """Save models to Google Cloud Storage - same pattern as inventory forecaster"""
        logger.info(f"Saving recommendation model to Cloud Storage...")
        
        bucket = self.storage_client.bucket(bucket_name)
        
        # Save each model component
        model_components = [
            'kmeans_model',
            'scaler', 
            'label_encoders',
            'user_clusters',
            'cluster_recommendations',
            'item_details'
        ]
        
        for component in model_components:
            model_blob = bucket.blob(f"recommendation_models/{component}.pkl")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Save model to the temporary file
                if component == 'kmeans_model':
                    joblib.dump(models['kmeans'], temp_filename)
                else:
                    joblib.dump(models[component], temp_filename)
                    
                # Upload the file to cloud storage
                model_blob.upload_from_filename(temp_filename)
                logger.info(f"✅ Saved {component}")
            finally:
                # Always clean up temp file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
        
        # Save model information
        metadata_blob = bucket.blob("recommendation_models/metadata.pkl")
        
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
    BUCKET_NAME = f"{PROJECT_ID}_ml_models"  # Same pattern as forecasting service
    
    print("🚀 Starting recommendation model training...")
    trainer = RecommendationTrainer(PROJECT_ID)
    
    # Step 1: Get data
    df_catalog, df_events = trainer.extract_data()
    print(f"📊 Got data: {len(df_catalog)} catalog items, {len(df_events)} user events")
    
    # Step 2: Clean data
    features_full, df_events_clean, item_details, label_encoders = trainer.preprocess_data(df_catalog, df_events)
    print(f"🧹 Cleaned data: {len(features_full)} users, {len(item_details)} items")
    
    # Step 3: Train models
    models, metadata = trainer.train_models(features_full, df_events_clean, item_details, label_encoders)
    print(f"🤖 Trained recommendation model with {metadata['n_clusters']} clusters")
    
    # Step 4: Save models
    trainer.save_models(models, metadata, BUCKET_NAME)
    print("✅ Done! Your recommendation model is ready for deployment.")