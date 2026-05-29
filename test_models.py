import joblib
import numpy as np

# ── TEST 1: Churn Model ───────────────────────────────
print("Testing Churn Model...")
churn_model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler_churn.pkl')

# Sample customer data
sample = np.array([[12, 70, 840, 1, 2, 1, 0, 0, 1]])
sample_scaled = scaler.transform(sample)
prediction = churn_model.predict(sample_scaled)
probability = churn_model.predict_proba(sample_scaled)

print(f"  Churn Prediction : {'YES - Will Leave' if prediction[0]==1 else 'NO - Will Stay'}")
print(f"  Churn Probability: {probability[0][1]*100:.1f}%")

# ── TEST 2: Recommendation Engine ─────────────────────
print("\nTesting Recommendation Engine...")

# Load the rules
rules = joblib.load('models/recommendation_engine.pkl')

def recommend_package(monthly_charges, tenure):
    """Pick the right package based on rules"""
    if monthly_charges > rules["high"]["threshold"]:
        r = rules["high"]
    elif monthly_charges > rules["med_high"]["threshold"]:
        r = rules["med_high"]
    elif monthly_charges > rules["medium"]["threshold"]:
        r = rules["medium"]
    elif tenure < rules["new"]["threshold"]:
        r = rules["new"]
    else:
        r = rules["low"]
    return r["package"], r["reason"]

# Test with 3 customers
test_customers = [
    {"name": "Ashan",   "charges": 90, "tenure": 24},
    {"name": "Nimesha", "charges": 45, "tenure": 3},
    {"name": "Dilshan", "charges": 65, "tenure": 12},
]

print("\nPackage Recommendations:")
print("-" * 50)
for c in test_customers:
    pkg, reason = recommend_package(c['charges'], c['tenure'])
    print(f"  Customer : {c['name']}")
    print(f"  Package  : {pkg}")
    print(f"  Reason   : {reason}")
    print("-" * 50)

# ── TEST 3: Usage Model ───────────────────────────────
print("\nTesting Usage Prediction Model...")
usage_model = joblib.load('models/usage_model.pkl')
scaler_usage = joblib.load('models/scaler_usage.pkl')

# Sample: tenure=12, MonthlyCharges=70, Contract=1,
# InternetService=1, StreamingTV=1, StreamingMovies=0, OnlineSecurity=0
usage_sample = np.array([[12, 70, 1, 1, 1, 0, 0]])
usage_scaled = scaler_usage.transform(usage_sample)
predicted_spend = usage_model.predict(usage_scaled)

print(f"  Predicted Total Spend : ${predicted_spend[0]:.2f}")

print("\nAll models working perfectly!")