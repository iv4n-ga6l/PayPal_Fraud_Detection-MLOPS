import pandas as pd
import matplotlib.pyplot as plt

def load_and_explore():
    """Load data and show essential insights."""
    print("🔍 PayPal Fraud Detection - Data Overview")
    print("=" * 45)
    
    # Load data
    try:
        users_df = pd.read_csv("data/paypal_users.csv")
        transactions_df = pd.read_csv("data/paypal_transactions.csv", nrows=10000)
        print(f"✅ Loaded: {users_df.shape[0]:,} users, {transactions_df.shape[0]:,} transactions")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Core fraud statistics
    if 'is_fraud' in users_df.columns:
        fraud_rate = users_df['is_fraud'].mean()
        print(f"\n🚨 Fraud Rate: {fraud_rate:.2%} ({users_df['is_fraud'].sum():,} fraudulent users)")
        
        # Top risk countries
        if 'country' in users_df.columns:
            country_risk = users_df.groupby('country')['is_fraud'].agg(['count', 'mean'])
            country_risk = country_risk[country_risk['count'] >= 50].sort_values('mean', ascending=False)
            print(f"\nTop risk countries:")
            for country, (count, rate) in country_risk.head(3).iterrows():
                print(f"  {country}: {rate:.1%}")
    
    # Key insights
    print(f"\n📊 Key Stats:")
    if 'kyc' in users_df.columns:
        kyc_fraud = users_df.groupby('kyc')['is_fraud'].mean()
        print(f"KYC fraud rates: {dict(kyc_fraud.round(3))}")
    
    if 'amount_usd' in transactions_df.columns:
        print(f"Transaction range: ${transactions_df['amount_usd'].min():.0f} - ${transactions_df['amount_usd'].max():,.0f}")
    
    if 'type' in transactions_df.columns:
        top_types = transactions_df['type'].value_counts().head(3)
        print(f"Top transaction types: {dict(top_types)}")
    
    # Feature recommendations
    print(f"\n💡 Key Features to Engineer:")
    features = [
        "Account age", "Transaction velocity", "Country risk scores", 
        "Amount anomalies", "KYC status", "Failed login patterns"
    ]
    print(f"  {', '.join(features)}")
    
    # Quick fraud visualization
    if 'is_fraud' in users_df.columns:
        plt.figure(figsize=(10, 3))
        
        plt.subplot(1, 2, 1)
        fraud_counts = users_df['is_fraud'].value_counts()
        plt.pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], 
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('User Distribution')
        
        if 'amount_usd' in transactions_df.columns:
            plt.subplot(1, 2, 2)
            plt.hist(transactions_df['amount_usd'], bins=30, alpha=0.7, color='blue')
            plt.xlabel('Amount (USD)')
            plt.ylabel('Count')
            plt.title('Transaction Amounts')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n✅ Ready for feature engineering and model training!")

if __name__ == "__main__":
    load_and_explore()
