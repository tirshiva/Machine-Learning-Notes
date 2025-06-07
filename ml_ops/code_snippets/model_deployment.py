"""
Model Deployment with FastAPI and Docker

This module demonstrates how to deploy a machine learning model using FastAPI
and Docker. It includes API endpoints, input validation, and model serving.

Author: ML Notes
Date: 2024
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
from typing import List, Optional
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for serving machine learning model predictions",
    version="1.0.0"
)

# Define input/output schemas
class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)
    timestamp: Optional[datetime] = None

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    timestamp: datetime

# Load model (in a real application, this would be loaded from a file)
try:
    model = joblib.load('model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.get("/")
async def root():
    """
    Root endpoint for health check.
    """
    return {"status": "healthy", "message": "ML Model API is running"}

@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown",
        "classes": model.classes_.tolist() if hasattr(model, 'classes_') else "Unknown"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make predictions using the loaded model.
    
    Args:
        input_data (PredictionInput): Input features for prediction
        
    Returns:
        PredictionOutput: Prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features).max()
        
        # Get timestamp
        timestamp = input_data.timestamp or datetime.now()
        
        # Log prediction
        logger.info(f"Prediction made: {prediction} with probability {probability:.4f}")
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            timestamp=timestamp
        )
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 