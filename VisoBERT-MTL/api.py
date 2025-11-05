"""
FastAPI Backend for Dual-Task ABSA Model
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os

# Handle imports from different directories
try:
    from model_service import get_model_service
except ImportError:
    # Try relative import if running as module
    from .model_service import get_model_service


# Initialize FastAPI app
app = FastAPI(
    title="Dual-Task ABSA API",
    description="API for Vietnamese Aspect-Based Sentiment Analysis (Aspect Detection + Sentiment Classification)",
    version="1.0.0"
)

# CORS middleware để frontend có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, nên chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization - model service will be loaded on first request
model_service = None

def get_model_service_instance():
    """Lazy load model service"""
    global model_service
    if model_service is None:
        print("Initializing model service...")
        model_service = get_model_service()
    return model_service


# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Input text to analyze", min_length=1)
    min_aspect_confidence: float = Field(0.5, description="Minimum confidence for aspect to be considered present (0.0-1.0)", ge=0.0, le=1.0)
    filter_absent: bool = Field(True, description="Only return aspects that are detected (present=True)")
    min_sentiment_confidence: float = Field(0.5, description="Minimum confidence for sentiment prediction (0.0-1.0)", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(3, description="Only return top K aspects with highest aspect confidence (recommended: 3-5, None = return all)", ge=1, le=11)


class PredictionResponse(BaseModel):
    text: str
    predictions: Dict[str, Dict[str, Any]]


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1)
    min_aspect_confidence: float = Field(0.5, description="Minimum confidence for aspect detection")
    filter_absent: bool = Field(True, description="Only return detected aspects")
    min_sentiment_confidence: float = Field(0.5, description="Minimum confidence for sentiment")
    top_k: Optional[int] = Field(3, description="Only return top K aspects", ge=1, le=11)


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Dual-Task ABSA API",
        "version": "1.0.0",
        "description": "Aspect Detection + Sentiment Classification",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Predict aspect detection and sentiment for single text",
            "/predict/batch": "Predict for multiple texts",
            "/docs": "API documentation",
            "/model/info": "Get model information"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status"""
    try:
        service = get_model_service_instance()
        return {
            "status": "healthy",
            "model_loaded": service is not None,
            "device": str(service.device) if service else "unknown"
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "device": f"error: {str(e)}"
        }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict aspect detection and sentiment classification for all aspects from a single text
    
    **Dual-Task Approach (Two-Stage):**
    1. **Step 1: Aspect Detection** - Uses aspect_detection_head (sigmoid) to detect which aspects are present
    2. **Step 2: Sentiment Classification** - Uses sentiment_classification_head (softmax) ONLY for detected aspects
    
    **Important:** Only aspects detected in Step 1 will have sentiment predictions in Step 2.
    This is different from multi-label approach which predicts sentiment for all aspects.
    
    Filtering options:
    - min_aspect_confidence: Minimum confidence for aspect to be considered "present"
    - filter_absent: Only return aspects that are detected (present=True)
    - min_sentiment_confidence: Minimum confidence for sentiment prediction
    - top_k: Only return top K aspects with highest aspect confidence
    
    Example:
    ```json
    {
        "text": "Pin trâu camera xấu",
        "min_aspect_confidence": 0.5,
        "filter_absent": true,
        "min_sentiment_confidence": 0.5,
        "top_k": 3
    }
    ```
    """
    try:
        service = get_model_service_instance()
        result = service.predict(
            request.text,
            min_aspect_confidence=request.min_aspect_confidence,
            filter_absent=request.filter_absent,
            min_sentiment_confidence=request.min_sentiment_confidence,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict aspect detection and sentiment for multiple texts at once
    
    Example:
    ```json
    {
        "texts": [
            "Pin trâu camera xấu",
            "Màn hình đẹp giá rẻ"
        ],
        "min_aspect_confidence": 0.5,
        "filter_absent": true,
        "top_k": 3
    }
    ```
    """
    try:
        if len(request.texts) > 100:  # Limit batch size
            raise HTTPException(
                status_code=400, 
                detail="Batch size too large. Maximum 100 texts per request."
            )
        
        service = get_model_service_instance()
        results = service.predict_batch(
            request.texts,
            min_aspect_confidence=request.min_aspect_confidence,
            filter_absent=request.filter_absent,
            min_sentiment_confidence=request.min_sentiment_confidence,
            top_k=request.top_k
        )
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/aspects", tags=["Information"])
async def get_aspects():
    """Get list of all aspects"""
    try:
        service = get_model_service_instance()
        return {
            "aspects": service.aspect_names,
            "count": len(service.aspect_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting aspects: {str(e)}")


@app.get("/sentiments", tags=["Information"])
async def get_sentiments():
    """Get list of all sentiments"""
    try:
        service = get_model_service_instance()
        return {
            "sentiments": service.sentiment_names,
            "count": len(service.sentiment_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sentiments: {str(e)}")


@app.get("/model/info", tags=["Information"])
async def get_model_info():
    """Get information about the loaded model"""
    try:
        service = get_model_service_instance()
        
        # Get checkpoint path
        checkpoint_path = os.path.join(service.model_dir, 'best_model.pt')
        checkpoint_exists = os.path.exists(checkpoint_path)
        
        info = {
            "model_name": service.config['model']['name'],
            "model_type": "dual-task (Aspect Detection + Sentiment Classification)",
            "model_dir": service.model_dir,
            "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_exists else None,
            "checkpoint_exists": checkpoint_exists,
            "device": str(service.device),
            "num_aspects": len(service.aspect_names),
            "aspects": service.aspect_names,
            "num_sentiments": len(service.sentiment_names),
            "sentiments": service.sentiment_names
        }
        
        # Add file info if checkpoint exists
        if checkpoint_exists:
            import datetime
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            file_mtime = os.path.getmtime(checkpoint_path)
            file_date = datetime.datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            info["checkpoint_size_mb"] = round(file_size, 2)
            info["checkpoint_last_modified"] = file_date
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


if __name__ == "__main__":
    # Run API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

