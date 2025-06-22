"""
Pydantic Data Models for PayPal Fraud Detection API
================================================

This module defines the data models used for API requests and responses
in the PayPal fraud detection system.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class FraudAction(str, Enum):    
    """Actions to take when fraud is detected."""
    LOCK_USER = "LOCK_USER"
    ALERT_AGENT = "ALERT_AGENT"
    APPROVE = "APPROVE"


class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TransactionType(str, Enum):
    """Transaction types."""
    CARD_PAYMENT = "CARD_PAYMENT"
    TOPUP = "TOPUP"
    ATM = "ATM"
    BANK_TRANSFER = "BANK_TRANSFER"
    P2P = "P2P"


class KYCStatus(str, Enum):
    """KYC status options."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    NONE = "NONE"


class UserData(BaseModel):
    """User profile information for fraud detection."""
    id: str = Field(..., description="Unique user identifier")
    has_email: int = Field(1, description="Whether user has email (0 or 1)")
    phone_country: Optional[str] = Field("GB", description="Phone country code")
    terms_version: Optional[str] = Field(None, description="Terms version accepted")
    country: str = Field(..., description="User's country")
    birth_year: int = Field(..., description="User's birth year")
    kyc: KYCStatus = Field(KYCStatus.PASSED, description="KYC verification status")
    failed_sign_in_attempts: int = Field(0, description="Number of failed sign-in attempts")
    
    @validator('birth_year')
    def validate_birth_year(cls, v):
        current_year = datetime.now().year
        if v < 1900 or v > current_year:
            raise ValueError('Invalid birth year')
        return v


class TransactionData(BaseModel):
    """Transaction information for fraud detection."""
    id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    amount_usd: float = Field(..., description="Transaction amount in USD", gt=0)
    currency: str = Field("USD", description="Transaction currency")
    state: Optional[str] = Field("COMPLETED", description="Transaction state")
    merchant_category: Optional[str] = Field(None, description="Merchant category")
    merchant_country: Optional[str] = Field(None, description="Merchant country")
    entry_method: Optional[str] = Field("chip", description="Entry method")
    type: TransactionType = Field(TransactionType.CARD_PAYMENT, description="Transaction type")
    source: Optional[str] = Field("GAIA", description="Transaction source")
    is_crypto: bool = Field(False, description="Whether transaction involves cryptocurrency")
    
    @validator('amount_usd')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class PredictionRequest(BaseModel):
    """Request for fraud prediction."""
    user_data: UserData = Field(..., description="User profile information")
    transaction_data: TransactionData = Field(..., description="Transaction details")
    model_name: Optional[str] = Field(None, description="Specific model to use (optional)")


class PredictionResponse(BaseModel):
    """Response from fraud prediction."""
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)", ge=0, le=1)
    is_fraud_prediction: bool = Field(..., description="Binary fraud prediction")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level of prediction")
    recommended_action: FraudAction = Field(..., description="Recommended action to take")
    model_used: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    risk_factors: List[str] = Field([], description="Identified risk factors")
    feature_contributions: Dict[str, float] = Field({}, description="Feature contributions to prediction")


class BatchPredictionRequest(BaseModel):
    """Request for batch fraud prediction."""
    users_data: List[UserData] = Field(..., description="List of user profiles")
    transactions_data: List[TransactionData] = Field(..., description="List of transactions")
    model_name: Optional[str] = Field(None, description="Specific model to use (optional)")
    
    @validator('transactions_data')
    def validate_batch_size(cls, v, values):
        if 'users_data' in values and len(v) != len(values['users_data']):
            raise ValueError('Users and transactions data must have the same length')
        return v


class BatchPredictionResponse(BaseModel):
    """Response from batch fraud prediction."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of predictions processed")
    timestamp: str = Field(..., description="Batch processing timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Health check timestamp")
    details: Dict[str, bool] = Field(..., description="Detailed health information")


class StatsResponse(BaseModel):
    """Prediction statistics response."""
    total_predictions: int = Field(..., description="Total number of predictions made")
    fraud_predictions: int = Field(..., description="Number of fraud predictions")
    lock_actions: int = Field(..., description="Number of LOCK_USER actions")
    alert_actions: int = Field(..., description="Number of ALERT_AGENT actions")
    approve_actions: int = Field(..., description="Number of APPROVE actions")
    fraud_rate: float = Field(..., description="Overall fraud rate")
    lock_rate: float = Field(..., description="Lock action rate")
    alert_rate: float = Field(..., description="Alert action rate")
    approve_rate: float = Field(..., description="Approve action rate")
    timestamp: str = Field(..., description="Statistics timestamp")


class ThresholdUpdateRequest(BaseModel):
    """Request to update decision thresholds."""
    thresholds: Dict[str, float] = Field(
        ..., 
        description="New threshold values",
        example={
            "LOCK_USER": 0.8,
            "ALERT_AGENT": 0.3
        }
    )
    
    @validator('thresholds')
    def validate_thresholds(cls, v):
        if 'LOCK_USER' in v and 'ALERT_AGENT' in v:
            if v['ALERT_AGENT'] > v['LOCK_USER']:
                raise ValueError('ALERT_AGENT threshold must be <= LOCK_USER threshold')
        
        for action, threshold in v.items():
            if not 0 <= threshold <= 1:
                raise ValueError(f'Threshold for {action} must be between 0 and 1')
        
        return v


# Legacy models for backward compatibility
class TransactionRequest(BaseModel):
    """Legacy transaction request model."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(..., description="Transaction currency")
    merchant_category: str = Field(..., description="Merchant category code")
    country: str = Field(..., description="Transaction country")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserProfile(BaseModel):
    """Legacy user profile model."""
    user_id: str
    account_creation_date: datetime
    country: str
    kyc_status: bool
    account_age_days: int
    total_transactions: int
    total_amount: float
    failed_transactions: int
    

class FraudPrediction(BaseModel):
    """Legacy fraud prediction response."""
    transaction_id: str
    user_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    action: FraudAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_factors: Dict[str, Any]
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_version: str
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    threshold: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AlertEvent(BaseModel):
    """Alert event for monitoring."""
    transaction_id: str
    user_id: str
    alert_type: str
    severity: str
    message: str
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)