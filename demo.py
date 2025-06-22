"""
PayPal Fraud Detection - Streamlit Demo Application
================================================

Interactive demo application that showcases the real-time fraud detection system.
Provides a user-friendly interface to test transactions and visualize results.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="PayPal Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    
    .risk-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    
    .action-lock {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
    }
    
    .action-alert {
        background-color: #fd7e14;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
    }
    
    .action-approve {
        background-color: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_api_connection() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_api_stats() -> Optional[Dict]:
    """Get API statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def make_prediction(user_data: Dict, transaction_data: Dict) -> Optional[Dict]:
    """Make a fraud prediction via API."""
    try:
        payload = {
            "user_data": user_data,
            "transaction_data": transaction_data
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def create_sample_data() -> List[Dict]:
    """Create sample transaction scenarios for testing."""
    samples = [
        {
            "name": "Low Risk - Regular Purchase",
            "user_data": {
                "id": "user_001",
                "has_email": 1,
                "phone_country": "GB",
                "country": "GB",
                "birth_year": 1985,
                "kyc": "PASSED",
                "failed_sign_in_attempts": 0
            },
            "transaction_data": {
                "id": "txn_001",
                "user_id": "user_001",
                "amount_usd": 75,
                "currency": "GBP",
                "state": "COMPLETED",
                "merchant_category": "restaurant",
                "merchant_country": "GBR",
                "entry_method": "chip",
                "type": "CARD_PAYMENT",
                "source": "GAIA",
                "is_crypto": False
            }
        },
        {
            "name": "Medium Risk - High Amount",
            "user_data": {
                "id": "user_002",
                "has_email": 1,
                "phone_country": "US",
                "country": "US",
                "birth_year": 1990,
                "kyc": "PASSED",
                "failed_sign_in_attempts": 0
            },
            "transaction_data": {
                "id": "txn_002",
                "user_id": "user_002",
                "amount_usd": 1500,
                "currency": "USD",
                "state": "COMPLETED",
                "merchant_category": "store",
                "merchant_country": "USA",
                "entry_method": "cont",
                "type": "CARD_PAYMENT",
                "source": "GAIA",
                "is_crypto": False
            }
        },
        {
            "name": "High Risk - Failed KYC + ATM",
            "user_data": {
                "id": "user_003",
                "has_email": 0,
                "phone_country": "XX",
                "country": "XX",
                "birth_year": 1970,
                "kyc": "FAILED",
                "failed_sign_in_attempts": 3
            },
            "transaction_data": {
                "id": "txn_003",
                "user_id": "user_003",
                "amount_usd": 500,
                "currency": "USD",
                "state": "COMPLETED",
                "merchant_category": "atm",
                "merchant_country": "XXX",
                "entry_method": "manu",
                "type": "ATM",
                "source": "INTERNAL",
                "is_crypto": False
            }
        }
    ]
    
    return samples


def render_prediction_result(result: Dict):
    """Render prediction results with nice formatting."""
    fraud_prob = result['fraud_probability']
    action = result['recommended_action']
    confidence = result['confidence_level']
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Fraud Probability",
            f"{fraud_prob:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Prediction",
            "FRAUD" if result['is_fraud_prediction'] else "LEGITIMATE",
            delta=None
        )
    
    with col3:
        confidence_color = {
            'HIGH': '🟢',
            'MEDIUM': '🟡',
            'LOW': '🔴'
        }
        st.metric(
            "Confidence",
            f"{confidence_color.get(confidence, '⚪')} {confidence}",
            delta=None
        )
    
    with col4:
        action_color = {
            'APPROVE': '✅',
            'ALERT_AGENT': '⚠️',
            'LOCK_USER': '🚫'
        }
        st.metric(
            "Action",
            f"{action_color.get(action, '❓')} {action}",
            delta=None
        )
    
    # Risk visualization
    st.subheader("Risk Assessment")
    
    # Create gauge chart for fraud probability
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fraud_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Risk (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Risk factors
        st.subheader("Risk Factors")
        risk_factors = result.get('risk_factors', [])
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"🔸 {factor.replace('_', ' ').title()}")
        else:
            st.markdown("✅ No significant risk factors detected")
        
        # Feature contributions (if available)
        # feature_contributions = result.get('feature_contributions', {})
        # if feature_contributions:
        #     st.subheader("Top Contributing Features")
            
        #     # Sort by contribution
        #     sorted_features = sorted(
        #         feature_contributions.items(),
        #         key=lambda x: abs(x[1]),
        #         reverse=True
        #     )[:5]
            
        #     for feature, contribution in sorted_features:
        #         st.markdown(f"• {feature}: {contribution:.3f}")


def render_transaction_form():
    """Render the transaction input form."""
    st.header("🔍 Test Fraud Detection")
    
    # Sample scenarios
    st.subheader("Quick Test Scenarios")
    samples = create_sample_data()
    
    cols = st.columns(len(samples))
    
    for i, (col, sample) in enumerate(zip(cols, samples)):
        with col:
            if st.button(sample['name'], key=f"sample_{i}"):
                st.session_state['sample_data'] = sample
    
    st.divider()
    
    # Manual input form
    st.subheader("Manual Input")
    
    # Check if sample data was selected
    sample_data = st.session_state.get('sample_data', {})
    
    with st.form("prediction_form"):
        # User Information
        st.markdown("### 👤 User Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input(
                "User ID",
                value=sample_data.get('user_data', {}).get('id', 'user_123')
            )
            
            country = st.selectbox(
                "Country",
                options=['GB', 'US', 'FR', 'DE', 'ES', 'IT', 'XX'],
                index=0 if not sample_data else ['GB', 'US', 'FR', 'DE', 'ES', 'IT', 'XX'].index(
                    sample_data.get('user_data', {}).get('country', 'GB')
                )
            )
            
            birth_year = st.number_input(
                "Birth Year",
                min_value=1940,
                max_value=2005,
                value=sample_data.get('user_data', {}).get('birth_year', 1990)
            )
        
        with col2:
            has_email = st.selectbox(
                "Has Email",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                index=0 if not sample_data else sample_data.get('user_data', {}).get('has_email', 1)
            )
            
            kyc_status = st.selectbox(
                "KYC Status",
                options=['PASSED', 'FAILED', 'PENDING', 'NONE'],
                index=0 if not sample_data else ['PASSED', 'FAILED', 'PENDING', 'NONE'].index(
                    sample_data.get('user_data', {}).get('kyc', 'PASSED')
                )
            )
            
            failed_attempts = st.number_input(
                "Failed Sign-in Attempts",
                min_value=0,
                max_value=10,
                value=sample_data.get('user_data', {}).get('failed_sign_in_attempts', 0)
            )
        
        # Transaction Information
        st.markdown("### 💳 Transaction Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_id = st.text_input(
                "Transaction ID",
                value=sample_data.get('transaction_data', {}).get('id', 'txn_123')
            )
            
            amount = st.number_input(
                "Amount (USD)",
                min_value=0.01,
                max_value=10000.0,
                value=float(sample_data.get('transaction_data', {}).get('amount_usd', 100))
            )
            
            currency = st.selectbox(
                "Currency",
                options=['USD', 'GBP', 'EUR', 'CAD', 'AUD'],
                index=0 if not sample_data else ['USD', 'GBP', 'EUR', 'CAD', 'AUD'].index(
                    sample_data.get('transaction_data', {}).get('currency', 'USD')
                ) if sample_data.get('transaction_data', {}).get('currency', 'USD') in ['USD', 'GBP', 'EUR', 'CAD', 'AUD'] else 0
            )
            
            merchant_category = st.selectbox(
                "Merchant Category",
                options=['restaurant', 'store', 'atm', 'supermarket', 'gas_station', 'bar', 'cafe'],
                index=0 if not sample_data else (
                    ['restaurant', 'store', 'atm', 'supermarket', 'gas_station', 'bar', 'cafe'].index(
                        sample_data.get('transaction_data', {}).get('merchant_category', 'restaurant')
                    ) if sample_data.get('transaction_data', {}).get('merchant_category') in 
                    ['restaurant', 'store', 'atm', 'supermarket', 'gas_station', 'bar', 'cafe'] else 0
                )
            )
        
        with col2:
            transaction_type = st.selectbox(
                "Transaction Type",
                options=['CARD_PAYMENT', 'TOPUP', 'ATM', 'BANK_TRANSFER', 'P2P'],
                index=0 if not sample_data else ['CARD_PAYMENT', 'TOPUP', 'ATM', 'BANK_TRANSFER', 'P2P'].index(
                    sample_data.get('transaction_data', {}).get('type', 'CARD_PAYMENT')
                )
            )
            
            entry_method = st.selectbox(
                "Entry Method",
                options=['chip', 'cont', 'manu', 'misc', 'mags'],
                index=0 if not sample_data else ['chip', 'cont', 'manu', 'misc', 'mags'].index(
                    sample_data.get('transaction_data', {}).get('entry_method', 'chip')
                )
            )
            
            source = st.selectbox(
                "Source",
                options=['GAIA', 'HERA', 'INTERNAL', 'LETO', 'MINOS'],
                index=0 if not sample_data else ['GAIA', 'HERA', 'INTERNAL', 'LETO', 'MINOS'].index(
                    sample_data.get('transaction_data', {}).get('source', 'GAIA')
                )
            )
            
            is_crypto = st.checkbox(
                "Cryptocurrency Transaction",
                value=sample_data.get('transaction_data', {}).get('is_crypto', False)
            )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")
        
        if submitted:
            # Prepare data
            user_data = {
                "id": user_id,
                "has_email": has_email,
                "phone_country": country,
                "country": country,
                "birth_year": birth_year,
                "kyc": kyc_status,
                "failed_sign_in_attempts": failed_attempts
            }
            
            transaction_data = {
                "id": transaction_id,
                "user_id": user_id,
                "amount_usd": amount,
                "currency": currency,
                "merchant_category": merchant_category,
                "type": transaction_type,
                "entry_method": entry_method,
                "source": source,
                "is_crypto": is_crypto
            }
            
            # Make prediction
            with st.spinner("Analyzing transaction..."):
                result = make_prediction(user_data, transaction_data)
            
            if result:
                st.success("Analysis completed!")
                st.divider()
                render_prediction_result(result)
            
            # Clear sample data after use
            if 'sample_data' in st.session_state:
                del st.session_state['sample_data']


def render_dashboard():
    """Render the monitoring dashboard."""
    st.header("📊 System Dashboard")
    
    # Get API stats
    stats = get_api_stats()
    
    if stats:
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Predictions",
                f"{stats['total_predictions']:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Fraud Rate",
                f"{stats['fraud_rate']:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Lock Rate",
                f"{stats['lock_rate']:.1%}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Alert Rate",
                f"{stats['alert_rate']:.1%}",
                delta=None
            )
        
        # Action distribution chart
        if stats['total_predictions'] > 0:
            action_data = {
                'Action': ['Lock User', 'Alert Agent', 'Approve'],
                'Count': [stats['lock_actions'], stats['alert_actions'], stats['approve_actions']],
                'Color': ['#dc3545', '#fd7e14', '#28a745']
            }
            
            fig_pie = px.pie(
                values=action_data['Count'],
                names=action_data['Action'],
                title="Action Distribution",
                color_discrete_sequence=action_data['Color']
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Last updated
        st.caption(f"Last updated: {stats['timestamp']}")
    
    else:
        st.warning("Unable to fetch system statistics. Check API connection.")


def main():
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ PayPal Fraud Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        Real-time machine learning fraud detection with advanced risk assessment
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["🔍 Fraud Detection", "📊 Dashboard"],
            index=0
        )
        
        # API Status
        st.markdown("### API Status")
        if check_api_connection():
            st.success("🟢 Online")
        else:
            st.error("🔴 Offline")
            st.markdown("Start API with: `python app/main.py`")
        
        # System Info
        st.markdown("### System Info")
        st.markdown(f"**Version:** 1.0.0")
        st.markdown(f"**Model:** RandomForest + GradientBoosting")
        st.markdown(f"**Features:** 50+ engineered features")
        
        # Quick Actions
        st.markdown("### Quick Actions")
        
        if st.button("🔄 Refresh Data"):
            st.experimental_rerun()

    
    # Main content based on selected page
    if page == "🔍 Fraud Detection":
        render_transaction_form()
    
    elif page == "📊 Dashboard":
        render_dashboard()
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
        PayPal Fraud Detection System | Built with Streamlit & FastAPI | 
        ML-Powered Real-time Risk Assessment
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()