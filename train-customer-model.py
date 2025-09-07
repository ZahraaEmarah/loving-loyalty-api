"""
Model Training Script for Loving Loyalty Recommendation System
Trains clustering model and saves artifacts to Google Cloud Storage
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from google.cloud import bigquery, storage
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, project_id: str, bucket_name: str, dataset_id: str = "LovingLoyaltyDB"):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.dataset_id = dataset_id
        self.bq_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        
        # Model configuration
        self.n_clusters = 8
        self.random_state = 42
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_time_of_day(self, timestamp: str) -> str:
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
    
    def load_data_from_bigquery(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load catalog and user events data from BigQuery"""
        logger.info("Loading data from BigQuery...")
        
        # Load catalog data
        catalog_query = f"""
        SELECT
            id AS item_id,
            title,
            price,
            currencyCode,
            availableTime
        FROM `{self.project_id}.{self.dataset_id}.vertex_ai_catalog_items_py`
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
        FROM `{self.project_id}.{self.dataset_id}.vertex_ai_user_events_py`
        WHERE originalPrice > 0 AND quantity > 0
        """
        
        try:
            df_catalog = self.bq_client.query(catalog_query).to_dataframe()
            df_events = self.bq_client.query(events_query).to_dataframe()
            
            # Add time of day feature to catalog
            df_catalog['time_of_day'] = df_catalog['availableTime'].apply(self.get_time_of_day)
            
            logger.info(f"Loaded {len(df_catalog)} catalog items and {len(df_events)} user events")
            return df_catalog, df_events
            
        except Exception as e:
            logger.error(f"Error loading data from BigQuery: {e}")
            raise
    
    def preprocess_data(self, df_catalog: pd.DataFrame, df_events: pd.DataFrame) -> Dict:
        """Preprocess data for training"""
        logger.info("Preprocessing data...")
        
        try:
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
            
            # Prepare features for clustering (exclude user_id)
            X = features_full.drop(columns=['user_id'])
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
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
            
            logger.info(f"Preprocessed data: {len(features_full)} users, {len(item_details)} items")
            
            return {
                'features_matrix': X.values,
                'features_full': features_full,
                'df_events': df_events,
                'df_catalog': df_catalog,
                'item_details': item_details,
                'label_encoders': label_encoders,
                'user_item_matrix': user_item_matrix
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train_clustering_model(self, X: np.ndarray) -> Tuple[KMeans, StandardScaler]:
        """Train K-means clustering model"""
        logger.info(f"Training K-means model with {self.n_clusters} clusters...")
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train K-means
            kmeans = KMeans(
                n_clusters=self.n_clusters, 
                random_state=self.random_state, 
                n_init=10,
                max_iter=300
            )
            kmeans.fit(X_scaled)
            
            logger.info(f"Model training completed. Inertia: {kmeans.inertia_:.2f}")
            return kmeans, scaler
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def build_recommendation_data(self, preprocessed_data: Dict, kmeans: KMeans, scaler: StandardScaler) -> Dict:
        """Build recommendation lookup tables"""
        logger.info("Building recommendation data...")
        
        try:
            features_full = preprocessed_data['features_full']
            df_events = preprocessed_data['df_events']
            item_details = preprocessed_data['item_details']
            
            # Predict clusters for all users
            X = features_full.drop(columns=['user_id'])
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
            X_scaled = scaler.transform(X.values)
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
            
            logger.info(f"Built recommendations for {len(user_clusters)} users across {self.n_clusters} clusters")
            
            return {
                'user_clusters': user_clusters,
                'cluster_recommendations': cluster_recommendations,
                'item_details': item_details
            }
            
        except Exception as e:
            logger.error(f"Error building recommendation data: {e}")
            raise
    
    def save_model_to_gcs(self, model_artifacts: Dict) -> str:
        """Save model artifacts to Google Cloud Storage"""
        logger.info(f"Saving model artifacts to GCS bucket: {self.bucket_name}")
        
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            model_path = f"models/loving_loyalty_recommendations/{self.model_version}/"
            
            # Save individual components
            artifacts_to_save = {
                'kmeans_model.pkl': model_artifacts['kmeans'],
                'scaler.pkl': model_artifacts['scaler'],
                'label_encoders.pkl': model_artifacts['label_encoders'],
                'user_clusters.pkl': model_artifacts['user_clusters'],
                'cluster_recommendations.pkl': model_artifacts['cluster_recommendations'],
                'item_details.pkl': model_artifacts['item_details']
            }
            
            # Save each artifact
            for filename, artifact in artifacts_to_save.items():
                blob_name = model_path + filename
                blob = bucket.blob(blob_name)
                
                # Serialize and upload
                serialized_data = pickle.dumps(artifact)
                blob.upload_from_string(serialized_data)
                logger.info(f"Saved {filename} to gs://{self.bucket_name}/{blob_name}")
            
            # Save metadata
            metadata = {
                'model_version': self.model_version,
                'training_date': datetime.now().isoformat(),
                'n_clusters': self.n_clusters,
                'random_state': self.random_state,
                'total_users': len(model_artifacts['user_clusters']),
                'total_items': len(model_artifacts['item_details']),
                'model_path': model_path
            }
            
            metadata_blob = bucket.blob(model_path + 'metadata.json')
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
            
            # Update latest model pointer
            latest_blob = bucket.blob('models/loving_loyalty_recommendations/latest.json')
            latest_blob.upload_from_string(json.dumps({
                'latest_version': self.model_version,
                'model_path': model_path,
                'updated_at': datetime.now().isoformat()
            }, indent=2))
            
            logger.info(f"Model training and saving completed successfully!")
            logger.info(f"Model version: {self.model_version}")
            logger.info(f"Model path: gs://{self.bucket_name}/{model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model to GCS: {e}")
            raise
    
    def train_and_save(self) -> str:
        """Main training pipeline"""
        logger.info("Starting model training pipeline...")
        
        try:
            # Load data
            df_catalog, df_events = self.load_data_from_bigquery()
            
            if len(df_events) == 0:
                raise ValueError("No user events data found")
            
            # Preprocess data
            preprocessed_data = self.preprocess_data(df_catalog, df_events)
            
            # Train model
            kmeans, scaler = self.train_clustering_model(preprocessed_data['features_matrix'])
            
            # Build recommendation data
            rec_data = self.build_recommendation_data(preprocessed_data, kmeans, scaler)
            
            # Prepare model artifacts
            model_artifacts = {
                'kmeans': kmeans,
                'scaler': scaler,
                'label_encoders': preprocessed_data['label_encoders'],
                'user_clusters': rec_data['user_clusters'],
                'cluster_recommendations': rec_data['cluster_recommendations'],
                'item_details': rec_data['item_details']
            }
            
            # Save to GCS
            model_path = self.save_model_to_gcs(model_artifacts)
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise

def main():
    """Main function"""
    # Configuration - adjust these values
    PROJECT_ID = "nse-gcp-ema-tt-0eee5-sbx-1"
    BUCKET_NAME = f"{PROJECT_ID}_ml_models"  # Following the same pattern as forecasting service
    DATASET_ID = "LovingLoyaltyDB"
    
    # Initialize trainer
    trainer = ModelTrainer(
        project_id=PROJECT_ID,
        bucket_name=BUCKET_NAME,
        dataset_id=DATASET_ID
    )
    
    try:
        # Train and save model
        model_path = trainer.train_and_save()
        print(f"✅ Model training completed successfully!")
        print(f"📁 Model saved to: gs://{BUCKET_NAME}/{model_path}")
        print(f"🏷️  Model version: {trainer.model_version}")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        raise

if __name__ == "__main__":
    main()