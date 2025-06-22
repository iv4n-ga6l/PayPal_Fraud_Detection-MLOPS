# 🛡️ PayPal Fraud Detection 

A **real-time machine learning system** for detecting fraudulent transactions. MLOps solution with FastAPI, feature engineering, and action recommendations.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models 
python train_model.py --quick-train

# 3. Start API server
python start_server.py

# 4. Launch demo
streamlit run demo.py
```

## 📊 Key Features

- **Real-time predictions** 
- **Intelligent actions**: APPROVE, ALERT_AGENT, or LOCK_USER  
- **50+ engineered features** from user profiles and transaction patterns
- **Multiple ML models**: Random Forest, Gradient Boosting, Logistic Regression
- **Interactive demo** with fraud scenarios and risk visualization

## System Components

```
Data → Feature Engineering → ML Models → API → Demo Interface
```

### Core Files
- `train_model.py` - Train fraud detection models
- `app/main.py` - FastAPI server for predictions  
- `demo.py` - Streamlit demo app
- `pipelines/` - Feature engineering and ML pipelines

## 🔗 API Usage

**Predict single transaction:**
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "user_data": {
    "id": "user_123", "country": "GB", "kyc": "PASSED", 
    "birth_year": 1990, "failed_sign_in_attempts": 0
  },
  "transaction_data": {
    "id": "txn_123", "amount_usd": 250, "currency": "USD",
    "type": "CARD_PAYMENT", "merchant_category": "retail"
  }
}'
```

**Response:**
```json
{
  "fraud_probability": 0.15,
  "recommended_action": "APPROVE",
  "confidence_level": "HIGH",
  "model_used": "random_forest"
}
```

## 📈 Performance

- **AUC:** 0.994 (Random Forest)
- **F1 Score:** 0.87 
- **Fraud Detection Rate:** 92%+
- **False Positive Rate:** <8%

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Access at:
# API: http://localhost:8000/docs
# Demo: http://localhost:8501
```

## 🛠️ Development

### Project Structure
```
├── app/                   # FastAPI app
├── pipelines/             # ML pipelines  
├── data/                  # Dataset files
├── models/                # Trained models
├── notebooks/             # Analysis notebooks
├── demo.py                # Streamlit demo
└── train_model.py         # Training script
```

## 📚 Documentation

- **API Docs:** `http://localhost:8000/docs`
- **Notebook:** `notebooks/PayPal_Fraud_Detection.ipynb`
- **Configuration:** `config.ini`

---

**🚀 Ready to detect fraud!** Start with `python train_model.py --quick-train`
