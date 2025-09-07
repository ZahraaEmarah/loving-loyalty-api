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

# Create the web application
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingService:
    def __init__(self):
        # CHANGE THIS TO YOUR PROJECT ID
        self.project_id = "YOUR-PROJECT-ID"  # Replace with your project ID
        self.bucket_name = f"{self.project_id}-ml-models"
        
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
    
    def bulk_forecast(self, place_ids, item_ids, steps=12):
        """Generate forecasts for multiple place-item combinations"""
        results = []
        
        for place_id in place_ids:
            for item_id in item_ids:
                forecast_result = self.forecast(place_id, item_id, steps)
                results.append(forecast_result)
        
        return results

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

@app.route('/bulk_forecast', methods=['POST'])
def bulk_predict_demand():
    try:
        data = request.json
        place_ids = data.get('place_ids', [])
        item_ids = data.get('item_ids', [])
        steps = data.get('steps', 12)
        
        if not place_ids or not item_ids:
            return jsonify({"error": "place_ids and item_ids are required"}), 400
        
        results = forecasting_service.bulk_forecast(place_ids, item_ids, steps)
        return jsonify({"forecasts": results})
        
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """Show all available models"""
    model_info = []
    for model_key, metadata in forecasting_service.metadata.items():
        model_info.append({
            "model_key": model_key,
            "place_id": metadata.get('place_id'),
            "item_id": metadata.get('item_id'),
            "aic": metadata.get('aic'),
            "training_data_points": metadata.get('training_data_points'),
            "last_updated": metadata.get('last_updated')
        })
    
    return jsonify({
        "total_models": len(model_info),
        "models": model_info
    })

# Start the web service
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)