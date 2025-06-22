"""
FastAPI App for PayPal Fraud Detection
============================================

Real-time fraud detection API with ML inference pipeline.
Provides endpoints for single transaction prediction, batch processing,
and model management.
"""

from datetime import datetime
from typing import Dict, Optional
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.inference import FraudDetectionInference
from app.models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse, HealthResponse,
    StatsResponse, ThresholdUpdateRequest
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="PayPal Fraud Detection API",
    description="Real-time fraud detection system for PayPal transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline instance
inference_pipeline: Optional[FraudDetectionInference] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the ML inference pipeline on startup."""
    global inference_pipeline
    
    try:
        logger.info("Initializing fraud detection inference pipeline...")
        inference_pipeline = FraudDetectionInference()
        
        # Perform health check
        health = inference_pipeline.health_check()
        if not health['overall_healthy']:
            logger.error("Inference pipeline health check failed", health=health)
            raise RuntimeError("Inference pipeline is not healthy")
        
        logger.info("Fraud detection API started successfully")
        
    except Exception as e:
        logger.error("Failed to initialize inference pipeline", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down fraud detection API...")

def get_inference_pipeline() -> FraudDetectionInference:
    """Dependency to get the inference pipeline."""
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline not initialized")
    return inference_pipeline

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PayPal Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(pipeline: FraudDetectionInference = Depends(get_inference_pipeline)):
    """Health check endpoint."""
    try:
        health_status = pipeline.health_check()
        
        return HealthResponse(
            status="healthy" if health_status['overall_healthy'] else "unhealthy",
            timestamp=datetime.now().isoformat(),
            details=health_status
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Health check failed")

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    request: PredictionRequest,
    pipeline: FraudDetectionInference = Depends(get_inference_pipeline)
):
    """
    Predict fraud probability for a single transaction.
    
    This endpoint analyzes a transaction and user data to determine
    the likelihood of fraud and recommend an appropriate action.
    """
    try:
        logger.info("Processing fraud prediction request", 
                   user_id=request.user_data.id,
                   transaction_id=request.transaction_data.id)
        
        # Convert Pydantic models to dictionaries
        user_data = request.user_data.dict()
        transaction_data = request.transaction_data.dict()
        
        # Make prediction
        result = await pipeline.predict_async(user_data, transaction_data, request.model_name)
        
        # Create response
        response = PredictionResponse(
            fraud_probability=result['fraud_probability'],
            is_fraud_prediction=result['is_fraud_prediction'],
            confidence_level=result['confidence_level'],
            recommended_action=result['recommended_action'],
            model_used=result['model_used'],
            timestamp=result['timestamp'],
            risk_factors=result.get('risk_factors', []),
            feature_contributions=result.get('feature_contributions', {})
        )
        
        logger.info("Fraud prediction completed",
                   user_id=request.user_data.id,
                   fraud_probability=result['fraud_probability'],
                   action=result['recommended_action'])
        
        return response
        
    except Exception as e:
        logger.error("Fraud prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    request: BatchPredictionRequest,
    pipeline: FraudDetectionInference = Depends(get_inference_pipeline)
):
    """
    Predict fraud probability for multiple transactions.
    
    Processes a batch of transactions for efficient bulk fraud detection.
    """
    try:
        if len(request.users_data) != len(request.transactions_data):
            raise HTTPException(
                status_code=400, 
                detail="Users and transactions data must have the same length"
            )
        
        logger.info("Processing batch fraud prediction",
                   batch_size=len(request.users_data))
        
        # Convert to dictionaries
        users_data = [user.dict() for user in request.users_data]
        transactions_data = [txn.dict() for txn in request.transactions_data]
        
        # Make batch prediction
        results = pipeline.predict_batch(users_data, transactions_data, request.model_name)
        
        # Create response
        predictions = []
        for result in results:
            prediction = PredictionResponse(
                fraud_probability=result['fraud_probability'],
                is_fraud_prediction=result['is_fraud_prediction'],
                confidence_level=result['confidence_level'],
                recommended_action=result['recommended_action'],
                model_used=result['model_used'],
                timestamp=result['timestamp'],
                risk_factors=result.get('risk_factors', []),
                feature_contributions=result.get('feature_contributions', {})
            )
            predictions.append(prediction)
        
        response = BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("Batch fraud prediction completed", batch_size=len(predictions))
        
        return response
        
    except Exception as e:
        logger.error("Batch fraud prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_prediction_stats(pipeline: FraudDetectionInference = Depends(get_inference_pipeline)):
    """Get prediction statistics and performance metrics."""
    try:
        stats = pipeline.get_prediction_stats()
        
        return StatsResponse(
            total_predictions=stats['total_predictions'],
            fraud_predictions=stats['fraud_predictions'],
            lock_actions=stats['lock_actions'],
            alert_actions=stats['alert_actions'],
            approve_actions=stats['approve_actions'],
            fraud_rate=stats.get('fraud_rate', 0.0),
            lock_rate=stats.get('lock_rate', 0.0),
            alert_rate=stats.get('alert_rate', 0.0),
            approve_rate=stats.get('approve_rate', 0.0),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

@app.post("/config/thresholds")
async def update_thresholds(
    request: ThresholdUpdateRequest,
    pipeline: FraudDetectionInference = Depends(get_inference_pipeline)
):
    """Update decision thresholds for fraud actions."""
    try:
        # Validate thresholds
        thresholds = request.thresholds
        
        if not (0 <= thresholds.get('ALERT_AGENT', 0.3) <= thresholds.get('LOCK_USER', 0.8) <= 1):
            raise HTTPException(
                status_code=400,
                detail="Invalid thresholds: ALERT_AGENT threshold must be <= LOCK_USER threshold"
            )
        
        # Update thresholds
        pipeline.update_decision_thresholds(thresholds)
        
        logger.info("Decision thresholds updated", thresholds=thresholds)
        
        return {
            "message": "Thresholds updated successfully",
            "new_thresholds": thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to update thresholds", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update thresholds: {str(e)}")

@app.get("/models")
async def list_available_models(pipeline: FraudDetectionInference = Depends(get_inference_pipeline)):
    """List available ML models."""
    try:
        models = list(pipeline.models.keys())
        
        return {
            "available_models": models,
            "best_model": pipeline.best_model_name,
            "model_count": len(models),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list models")

@app.post("/predict/explain")
async def explain_prediction(
    request: PredictionRequest,
    pipeline: FraudDetectionInference = Depends(get_inference_pipeline)
):
    """
    Get detailed explanation for a fraud prediction.
    
    Provides feature contributions and risk factor analysis
    for model interpretability.
    """
    try:
        logger.info("Processing prediction explanation request")
        
        # Convert to dictionaries
        user_data = request.user_data.dict()
        transaction_data = request.transaction_data.dict()
        
        # Make prediction with explanations
        result = await pipeline.predict_async(user_data, transaction_data, request.model_name)
        
        # Enhanced explanation response
        explanation = {
            "prediction": {
                "fraud_probability": result['fraud_probability'],
                "recommended_action": result['recommended_action'],
                "confidence_level": result['confidence_level']
            },
            "feature_contributions": result.get('feature_contributions', {}),
            "risk_factors": result.get('risk_factors', []),
            "decision_logic": {
                "thresholds": pipeline.decision_thresholds,
                "model_used": result['model_used']
            },
            "timestamp": result['timestamp']
        }
        
        return explanation
        
    except Exception as e:
        logger.error("Prediction explanation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error("Internal server error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )