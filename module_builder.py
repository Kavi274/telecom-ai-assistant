import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os

# ── STEP 1: Load the dataset ──────────────────────────
print("Loading dataset...")
df = pd.read_csv('data/telecom.csv')
print(f"Loaded {len(df)} rows")

# ── STEP 2: Clean the data ────────────────────────────
# TotalCharges has some empty spaces — fix them
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Convert Churn Yes/No to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("Data cleaned")
print(df.shape)

# ── STEP 3: Encode text columns to numbers ────────────
print("\nEncoding data...")

# Make a copy for encoding
df_encoded = df.copy()

# Drop customerID — it's not useful for ML
df_encoded.drop('customerID', axis=1, inplace=True)

# Find all text (object) columns and encode them
label_encoders = {}
for column in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le

print("Encoding done")
print("Columns now:", list(df_encoded.columns))

# ── STEP 4: Churn Prediction Model ───────────────────
print("\nBuilding Churn Prediction Model...")

# Features (inputs) and Target (what we predict)
churn_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                  'Contract', 'PaymentMethod', 'InternetService',
                  'OnlineSecurity', 'TechSupport', 'StreamingTV']

X_churn = df_encoded[churn_features]
y_churn = df_encoded['Churn']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42
)

# Scale the numbers
scaler_churn = StandardScaler()
X_train_scaled = scaler_churn.fit_transform(X_train)
X_test_scaled = scaler_churn.transform(X_test)

# Train the model
churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
churn_model.fit(X_train_scaled, y_train)

# Test accuracy
y_pred = churn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Churn Model Accuracy: {accuracy * 100:.1f}%")

# Save the model
os.makedirs('models', exist_ok=True)
joblib.dump(churn_model, 'models/churn_model.pkl')
joblib.dump(scaler_churn, 'models/scaler_churn.pkl')
print("Churn model saved to models/")

# ── STEP 5: Usage Prediction Model ───────────────────
print("\nBuilding Usage Prediction Model...")

# We use MonthlyCharges to simulate usage prediction
# (Real dataset doesn't have GB usage — we use charges as proxy)
usage_features = ['tenure', 'MonthlyCharges', 'Contract',
                  'InternetService', 'StreamingTV',
                  'StreamingMovies', 'OnlineSecurity']

X_usage = df_encoded[usage_features]
y_usage = df_encoded['TotalCharges']  # predicting total spend

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_usage, y_usage, test_size=0.2, random_state=42
)

scaler_usage = StandardScaler()
X_train_u_scaled = scaler_usage.fit_transform(X_train_u)
X_test_u_scaled = scaler_usage.transform(X_test_u)

usage_model = LinearRegression()
usage_model.fit(X_train_u_scaled, y_train_u)

y_pred_u = usage_model.predict(X_test_u_scaled)
rmse = np.sqrt(mean_squared_error(y_test_u, y_pred_u))
print(f"Usage Model RMSE: {rmse:.2f}")

joblib.dump(usage_model, 'models/usage_model.pkl')
joblib.dump(scaler_usage, 'models/scaler_usage.pkl')
print("Usage model saved to models/")

# ── STEP 6: Package Recommendation Engine ─────────────
print("\nBuilding Package Recommendation Engine...")

def recommend_package(monthly_charges, tenure, contract_type):
    """
    Recommends a data package based on customer profile.
    contract_type: 0=Month-to-month, 1=One year, 2=Two year
    """
    if monthly_charges > 80:
        package = "Unlimited Pro - $49/month"
        reason = "High spender - gets maximum value from unlimited data"
    elif monthly_charges > 60:
        package = "Premium 25GB - $39/month"
        reason = "Medium-high user - Premium covers your usage perfectly"
    elif monthly_charges > 40:
        package = "Standard 10GB - $29/month"
        reason = "Medium user - Standard plan is cost-effective for you"
    elif tenure < 6:
        package = "Student Flex 5GB - $15/month"
        reason = "New customer - start with our affordable entry plan"
    else:
        package = "Night Owl 8GB - $12/month"
        reason = "Low usage detected - Night Owl gives great value"

    return package, reason

# Test it with 3 example customers
test_customers = [
    {"name": "Ashan",   "charges": 90, "tenure": 24, "contract": 2},
    {"name": "Nimesha", "charges": 45, "tenure": 3,  "contract": 0},
    {"name": "Dilshan", "charges": 65, "tenure": 12, "contract": 1},
]

print("\nPackage Recommendations:")
print("-" * 50)
for c in test_customers:
    pkg, reason = recommend_package(
        c['charges'], c['tenure'], c['contract']
    )
    print(f"Customer : {c['name']}")
    print(f"Package  : {pkg}")
    print(f"Reason   : {reason}")
    print("-" * 50)

# Save recommendation rules as a dictionary instead of a function
recommendation_rules = {
    "high":    {"threshold": 80, "package": "Unlimited Pro - $49/month",    "reason": "High spender - gets maximum value from unlimited data"},
    "med_high":{"threshold": 60, "package": "Premium 25GB - $39/month",     "reason": "Medium-high user - Premium covers your usage perfectly"},
    "medium":  {"threshold": 40, "package": "Standard 10GB - $29/month",    "reason": "Medium user - Standard plan is cost-effective for you"},
    "new":     {"threshold": 6,  "package": "Student Flex 5GB - $15/month", "reason": "New customer - start with our affordable entry plan"},
    "low":     {"threshold": 0,  "package": "Night Owl 8GB - $12/month",    "reason": "Low usage detected - Night Owl gives great value"},
}

joblib.dump(recommendation_rules, 'models/recommendation_engine.pkl')
print("Recommendation engine saved!")