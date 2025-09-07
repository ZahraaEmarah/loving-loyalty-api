# from flask import Flask, jsonify, request
# from google.cloud import bigquery
# from google.auth.exceptions import DefaultCredentialsError
# import inventory_management

# app = Flask(__name__)

# # BigQuery client (uses service account JSON file)
# try:
#     # Try to use service account file if available
#     client = bigquery.Client.from_service_account_json("service-account.json")
# except (FileNotFoundError, DefaultCredentialsError):
#     # Fallback to default credentials (e.g., GOOGLE_APPLICATION_CREDENTIALS env variable)
#     client = bigquery.Client()

# @app.route("/forecast-inventory-stock", methods=["GET"])
# def forecast():
#     place_id = request.args.get("place_id", type=int)
#     item_id = request.args.get("item_id", type=int)
#     if place_id is None or item_id is None:
#         return jsonify({"error": "Missing place_id or item_id"}), 400

#     forecast_result = inventory_management.get_forecast_for_place_item(place_id, item_id)
#     if forecast_result is None:
#         return jsonify({"error": "No data available for given place_id and item_id"}), 404

#     # Return forecast as JSON
#     return jsonify({
#         "place_id": place_id,
#         "item_id": item_id,
#         "forecast": forecast_result.tolist()
#     })

# if __name__ == "__main__":
#     app.run(port=5000, debug=True)

# app.py - CORRECTED VERSION
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from google.cloud import storage
from google.cloud import bigquery
import logging
from datetime import datetime, timedelta
import os
import tempfile

_model_cache = {
    'kmeans': None,
    'scaler': None,
    'label_encoders': None,
    'user_clusters': None,
    'cluster_recommendations': None,
    'item_details': None,
    'metadata': None,
    'loaded_at': None
}

# Create the web application
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingService:
    def __init__(self):
        # CHANGE THIS TO YOUR PROJECT ID
        self.project_id = ""  # Replace with your project ID
        self.bucket_name = f"{self.project_id}_ml_models"
        
        # Check if we're running locally or in the cloud
        service_account_path = "service-account.json"
        if os.path.exists(service_account_path):
            # Running locally - use service account file
            self.storage_client = storage.Client.from_service_account_json(service_account_path)
            self.bigquery_client = bigquery.Client.from_service_account_json(service_account_path)
        else:
            # Running in cloud - use environment authentication
            self.storage_client = storage.Client()
            self.bigquery_client = bigquery.Client()
        
        self.models = {}
        self.metadata = {}
        self.load_models()
    
    def get_current_inventory_level(self, place_id, item_id, weeks_back=4):
        """Get the current/recent inventory level from BigQuery"""
        try:
            # Query to get recent order data for this place-item combination
            query = f"""
            WITH recent_orders AS (
                SELECT 
                    o.place_id,
                    oi.item_id,
                    o.created as order_date,
                    COUNT(*) as daily_orders
                FROM LovingLoyaltyDB.fct_orders o
                JOIN LovingLoyaltyDB.fct_orders_items oi ON o.id = oi.order_id
                WHERE o.place_id = {place_id}
                  AND oi.item_id = {item_id}
                  AND oi.status = 'Fulfilled'
                  AND o.created >= UNIX_TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL {weeks_back} WEEK))
                GROUP BY o.place_id, oi.item_id, o.created
            ),
            weekly_summary AS (
                SELECT 
                    place_id,
                    item_id,
                    EXTRACT(WEEK FROM TIMESTAMP_SECONDS(order_date)) as week_number,
                    EXTRACT(YEAR FROM TIMESTAMP_SECONDS(order_date)) as year,
                    SUM(daily_orders) as weekly_orders
                FROM recent_orders
                GROUP BY place_id, item_id, week_number, year
                ORDER BY year DESC, week_number DESC
                LIMIT 1
            )
            SELECT 
                place_id,
                item_id,
                weekly_orders as current_weekly_orders,
                CONCAT(year, '-W', LPAD(CAST(week_number AS STRING), 2, '0')) as latest_week
            FROM weekly_summary
            """
            
            result = self.bigquery_client.query(query).to_dataframe()
            
            if len(result) > 0:
                return {
                    "current_weekly_orders": int(result.iloc[0]['current_weekly_orders']),
                    "latest_week": result.iloc[0]['latest_week'],
                    "data_available": True
                }
            else:
                return {
                    "current_weekly_orders": 0,
                    "latest_week": "No recent data",
                    "data_available": False
                }
                
        except Exception as e:
            logger.error(f"Error getting current inventory for place {place_id}, item {item_id}: {e}")
            return {
                "current_weekly_orders": 0,
                "latest_week": "Error fetching data",
                "data_available": False,
                "error": str(e)
            }
    
    def load_models(self):
        """Load the trained models from Google Cloud Storage - CORRECTED VERSION"""
        logger.info("Loading models from Cloud Storage...")
        
        bucket = self.storage_client.bucket(self.bucket_name)
        
        # Load model information first
        try:
            metadata_blob = bucket.blob("inventory_models/metadata.pkl")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Download from cloud storage to temp file
                metadata_blob.download_to_filename(temp_filename)
                # Load from temp file
                self.metadata = joblib.load(temp_filename)
                logger.info(f"📋 Loaded information for {len(self.metadata)} models")
            finally:
                # Always clean up temp file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                    
        except Exception as e:
            logger.error(f"❌ Failed to load model information: {e}")
            return
        
        # Load each model
        for model_key in self.metadata.keys():
            try:
                model_blob = bucket.blob(f"inventory_models/{model_key}.pkl")
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                    temp_filename = temp_file.name
                
                try:
                    # Download from cloud storage to temp file
                    model_blob.download_to_filename(temp_filename)
                    # Load from temp file
                    self.models[model_key] = joblib.load(temp_filename)
                finally:
                    # Always clean up temp file
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)
                        
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_key}: {e}")
        
        logger.info(f"✅ Successfully loaded {len(self.models)} models")
    
    def forecast(self, place_id, item_id, steps=12):
        """Generate forecast for specific place and item WITH current inventory data"""
        model_key = f"{place_id}_{item_id}"
        
        # Get current inventory level
        current_data = self.get_current_inventory_level(place_id, item_id)
        
        if model_key not in self.models:
            return {
                "error": f"No model found for place {place_id}, item {item_id}",
                "current_data": current_data
            }
        
        try:
            model = self.models[model_key]
            forecast = model.forecast(steps=steps)
            
            # Create future dates (weekly)
            last_date = datetime.now()
            future_dates = [
                (last_date + timedelta(weeks=i+1)).strftime("%Y-%m-%d")
                for i in range(steps)
            ]
            
            # Calculate summary statistics
            forecast_values = [max(0, round(float(pred), 2)) for pred in forecast]
            avg_predicted = round(sum(forecast_values) / len(forecast_values), 2)
            
            return {
                "place_id": place_id,
                "item_id": item_id,
                "current_data": current_data,
                "forecast": {
                    "predictions": forecast_values,
                    "dates": future_dates,
                    "average_predicted_weekly": avg_predicted,
                    "total_predicted_period": round(sum(forecast_values), 2)
                },
                "comparison": {
                    "current_vs_avg_predicted": {
                        "current": current_data.get("current_weekly_orders", 0),
                        "predicted_avg": avg_predicted,
                        "difference": round(avg_predicted - current_data.get("current_weekly_orders", 0), 2),
                        "percentage_change": round(
                            ((avg_predicted - current_data.get("current_weekly_orders", 0)) / 
                             max(current_data.get("current_weekly_orders", 1), 1)) * 100, 2
                        ) if current_data.get("current_weekly_orders", 0) > 0 else "N/A"
                    }
                },
                "model_metadata": self.metadata.get(model_key, {})
            }
            
        except Exception as e:
            return {
                "error": f"Forecasting failed: {str(e)}",
                "current_data": current_data
            }
    

class RecommendationService:
    def __init__(self):
        # CHANGE THIS TO YOUR PROJECT ID
        self.project_id = ""  # Replace with your project ID
        self.bucket_name = f"{self.project_id}_ml_models_recommendations"
        self.dataset_id = "LovingLoyaltyDB"
        
        # Check if we're running locally or in the cloud
        service_account_path = "service-account.json"
        if os.path.exists(service_account_path):
            # Running locally - use service account file
            self.storage_client = storage.Client.from_service_account_json(service_account_path)
            self.bigquery_client = bigquery.Client.from_service_account_json(service_account_path)
        else:
            # Running in cloud - use environment authentication
            self.storage_client = storage.Client()
            self.bigquery_client = bigquery.Client()
        
        # Model components
        self.models = {}
        self.metadata = {}
        self.user_clusters = {}
        self.cluster_recommendations = None
        self.item_details = None
        
        # Load models on initialization
        self.load_models()
    
    def get_current_user_activity(self, customer_id, days_back=30):
        """Get the current/recent user activity from BigQuery"""
        try:
            # Query to get recent user activity
            query = f"""
            WITH recent_activity AS (
                SELECT 
                    visitorId as user_id,
                    item_id,
                    quantity,
                    originalPrice,
                    eventTime,
                    age_group,
                    area,
                    gender,
                    country,
                    orders_count
                FROM `{self.project_id}.{self.dataset_id}.vertex_ai_user_events_py`
                WHERE visitorId = '{customer_id}'
                  AND TIMESTAMP(eventTime) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
                ORDER BY eventTime DESC
            ),
            activity_summary AS (
                SELECT 
                    user_id,
                    COUNT(DISTINCT item_id) as unique_items_purchased,
                    SUM(quantity) as total_quantity,
                    SUM(quantity * originalPrice) as total_spent,
                    AVG(originalPrice) as avg_item_price,
                    MAX(eventTime) as last_purchase_date,
                    ANY_VALUE(age_group) as age_group,
                    ANY_VALUE(area) as area,
                    ANY_VALUE(gender) as gender,
                    ANY_VALUE(country) as country,
                    ANY_VALUE(orders_count) as total_orders
                FROM recent_activity
                GROUP BY user_id
            )
            SELECT 
                user_id,
                unique_items_purchased,
                total_quantity,
                total_spent,
                avg_item_price,
                last_purchase_date,
                age_group,
                area,
                gender,
                country,
                total_orders
            FROM activity_summary
            """
            
            result = self.bigquery_client.query(query).to_dataframe()
            
            if len(result) > 0:
                row = result.iloc[0]
                return {
                    "user_id": customer_id,
                    "unique_items_purchased": int(row['unique_items_purchased']),
                    "total_quantity": int(row['total_quantity']),
                    "total_spent": float(row['total_spent']),
                    "avg_item_price": float(row['avg_item_price']),
                    "last_purchase_date": str(row['last_purchase_date']),
                    "age_group": str(row['age_group']) if pd.notna(row['age_group']) else "Unknown",
                    "area": str(row['area']) if pd.notna(row['area']) else "Unknown",
                    "gender": str(row['gender']) if pd.notna(row['gender']) else "Unknown",
                    "country": str(row['country']) if pd.notna(row['country']) else "Unknown",
                    "total_orders": int(row['total_orders']) if pd.notna(row['total_orders']) else 0,
                    "data_available": True
                }
            else:
                return {
                    "user_id": customer_id,
                    "data_available": False,
                    "message": "No recent activity found for this customer"
                }
                
        except Exception as e:
            logger.error(f"Error getting current activity for customer {customer_id}: {e}")
            return {
                "user_id": customer_id,
                "data_available": False,
                "error": str(e)
            }
    
    def load_models(self):
        """Load the trained recommendation models from Google Cloud Storage"""
        logger.info("Loading recommendation models from Cloud Storage...")
        
        bucket = self.storage_client.bucket(self.bucket_name)
        
        # Load model metadata first
        try:
            metadata_blob = bucket.blob("recommendation_models/metadata.pkl")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Download from cloud storage to temp file
                metadata_blob.download_to_filename(temp_filename)
                # Load from temp file
                self.metadata = joblib.load(temp_filename)
                logger.info(f"📋 Loaded model metadata: {self.metadata}")
            finally:
                # Always clean up temp file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                    
        except Exception as e:
            logger.error(f"❌ Failed to load model metadata: {e}")
            return
        
        # Load each model component
        model_components = [
            'kmeans_model',
            'scaler',
            'label_encoders',
            'user_clusters',
            'cluster_recommendations',
            'item_details'
        ]
        
        for component in model_components:
            try:
                component_blob = bucket.blob(f"recommendation_models/{component}.pkl")
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                    temp_filename = temp_file.name
                
                try:
                    # Download from cloud storage to temp file
                    component_blob.download_to_filename(temp_filename)
                    # Load from temp file
                    loaded_component = joblib.load(temp_filename)
                    
                    # Store in appropriate attribute
                    if component == 'kmeans_model':
                        self.models['kmeans'] = loaded_component
                    elif component == 'scaler':
                        self.models['scaler'] = loaded_component
                    elif component == 'label_encoders':
                        self.models['label_encoders'] = loaded_component
                    elif component == 'user_clusters':
                        self.user_clusters = loaded_component
                    elif component == 'cluster_recommendations':
                        self.cluster_recommendations = loaded_component
                    elif component == 'item_details':
                        self.item_details = loaded_component
                        
                finally:
                    # Always clean up temp file
                    if os.path.exists(temp_filename):
                        os.unlink(temp_filename)
                        
            except Exception as e:
                logger.error(f"❌ Failed to load model component {component}: {e}")
        
        logger.info(f"✅ Successfully loaded recommendation model components")
        logger.info(f"📊 Total users in model: {len(self.user_clusters) if self.user_clusters else 0}")
        logger.info(f"📦 Total items in catalog: {len(self.item_details) if self.item_details is not None else 0}")
    
    def predict(self, customer_id, top_n=3):
        """Generate top N food recommendations for specific customer WITH current activity data"""
        customer_id = str(customer_id)
        
        # Get current user activity
        current_data = self.get_current_user_activity(customer_id)
        
        if customer_id not in self.user_clusters:
            return {
                "error": f"No recommendations available for customer {customer_id}. Customer not found in training data.",
                "current_data": current_data,
                "available_customers": len(self.user_clusters) if self.user_clusters else 0
            }
        
        try:
            # Get user's cluster
            user_cluster = self.user_clusters[customer_id]
            
            # Get top items for this cluster
            if self.cluster_recommendations is None:
                raise ValueError("Cluster recommendations not loaded")
            
            cluster_items = self.cluster_recommendations[
                self.cluster_recommendations['cluster'] == user_cluster
            ].head(top_n)
            
            # Merge with item details if available
            if self.item_details is not None:
                cluster_items = cluster_items.merge(
                    self.item_details[['item_id', 'title', 'price', 'time_of_day']], 
                    on='item_id', 
                    how='left'
                )
            
            # Format predictions
            predictions = []
            for _, row in cluster_items.iterrows():
                prediction = {
                    "item_id": str(row['item_id']),
                    "title": row['title'] if 'title' in row and pd.notna(row['title']) else f"Food Item {row['item_id']}",
                    "price": float(row['price']) if 'price' in row and pd.notna(row['price']) else 0.0,
                    "time_of_day": row['time_of_day'] if 'time_of_day' in row and pd.notna(row['time_of_day']) else "Any",
                    "popularity_score": float(row['popularity_score']) if 'popularity_score' in row else 0.0,
                    "total_purchases": int(row['total_purchases']) if 'total_purchases' in row else 0,
                    "unique_users": int(row['unique_users']) if 'unique_users' in row else 0,
                    "cluster_rank": len(predictions) + 1
                }
                predictions.append(prediction)
            
            # Calculate summary statistics
            avg_price = sum(p['price'] for p in predictions) / len(predictions) if predictions else 0
            total_popularity = sum(p['popularity_score'] for p in predictions)
            
            return {
                "customer_id": customer_id,
                "cluster_id": int(user_cluster),
                "current_data": current_data,
                "recommendations": {
                    "predictions": predictions,
                    "total_recommendations": len(predictions),
                    "avg_predicted_price": round(avg_price, 2),
                    "total_popularity_score": round(total_popularity, 2)
                },
                "insights": {
                    "user_segment": f"Cluster {user_cluster}",
                    "recommendation_basis": "Collaborative filtering based on similar users",
                    "confidence": "High" if len(predictions) >= top_n else "Medium"
                },
                "model_metadata": {
                    "model_version": self.metadata.get("model_version", "unknown"),
                    "training_date": self.metadata.get("training_date", "unknown"),
                    "total_clusters": self.metadata.get("n_clusters", "unknown")
                }
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "customer_id": customer_id,
                "current_data": current_data
            }

# Initialize the service
recommendation_service = RecommendationService()

# Initialize service
forecasting_service = ForecastingService()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": len(forecasting_service.models),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/forecast', methods=['POST'])
def predict_demand():
    try:
        data = request.json
        place_id = data.get('place_id')
        item_id = data.get('item_id')
        steps = data.get('steps', 12)
        
        if not place_id or not item_id:
            return jsonify({"error": "You must provide both place_id and item_id"}), 400
        
        result = forecasting_service.forecast(place_id, item_id, steps)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict/<customer_id>', methods=['GET'])
def predict_for_customer(customer_id: str):
    """Get top 3 food predictions for a customer (main endpoint)"""
    try:
        # Get top_n parameter (default to 3, max 10)
        top_n = min(request.args.get('top_n', 3, type=int), 10)
        
        # Generate predictions
        result = recommendation_service.predict(customer_id, top_n)
        
        if "error" in result:
            return jsonify(result), 404 if "not found" in result["error"].lower() else 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'customer_id': customer_id,
            'message': str(e)
        }), 500


# Start the web service
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)