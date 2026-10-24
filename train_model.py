import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("--- 1. LOADING DATASET ---")
url = "https://raw.githubusercontent.com/satishgunjal/House-Price-Prediction-Project/master/Bengaluru_House_Data.csv"
df = pd.read_csv(url)
print(f"Original dataset shape: {df.shape}")

print("\n--- 2. CLEANING DATA ---")
df = df.drop(
    columns=["area_type", "society", "balcony", "availability"]
).dropna()
df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))
df = df.drop(columns=["size"])
def convert_sqft_to_num(x):
  tokens = x.split("-")
  if len(tokens) == 2:
    return (float(tokens[0]) + float(tokens[1])) / 2
  try:
    return float(x)
  except:
    return None


df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)
df = df.dropna()
df["location"] = df["location"].apply(lambda x: x.strip())
location_stats = df.groupby("location")["location"].agg("count")
locations_less_than_10 = location_stats[location_stats <= 10]
df["location"] = df["location"].apply(
    lambda x: "Other" if x in locations_less_than_10 else x
)

print("\n--- 3. REMOVING OUTLIERS ---")
df = df[~(df.total_sqft / df.bhk < 300)]
df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]


def remove_pps_outliers(data):
  df_out = pd.DataFrame()
  for key, subdf in data.groupby("location"):
    m = np.mean(subdf["price_per_sqft"])
    st = np.std(subdf["price_per_sqft"])
    reduced_df = subdf[
        (subdf["price_per_sqft"] > (m - st))
        & (subdf["price_per_sqft"] <= (m + st))
    ]
    df_out = pd.concat([df_out, reduced_df], ignore_index=True)
  return df_out


df = remove_pps_outliers(df)
df = df.drop(columns=["price_per_sqft"])
print(f"Dataset shape after outlier removal: {df.shape}")

print("\n--- 4. ENCODING & TRAIN-TEST SPLIT ---")
df_encoded = pd.get_dummies(df, columns=["location"], drop_first=True)

X = df_encoded.drop(columns=["price"])
y = df_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n--- 5. TRAINING & COMPARING MODELS ---")
start_time = time.time()
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_time = time.time() - start_time

lr_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)
start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n================ RESULT COMPARISON ================")
print(
    f"Linear Regression -> Training Time: {lr_time:.2f}s | RMSE: ₹"
    f"{lr_rmse:.2f}L | R² Score: {lr_r2:.4f}"
)
print(
    f"Random Forest     -> Training Time: {rf_time:.2f}s | RMSE: ₹"
    f"{rf_rmse:.2f}L | R² Score: {rf_r2:.4f}"
)
print("===================================================")
import joblib

joblib.dump(rf_model, "house_price_model.pkl")
print("Model saved successfully as 'house_price_model.pkl'!")



def predict_price(location, sqft, bath, bhk):
  loc_index = (
      np.where(X.columns == f"location_{location}")[0][0]
      if f"location_{location}" in X.columns
      else -1
  )
  x = np.zeros(len(X.columns))
  x[0] = bath
  x[1] = sqft
  x[2] = bhk
  if loc_index >= 0:
    x[loc_index] = 1
  predicted_price = rf_model.predict([x])[0]
  return f"Estimated Price in {location}: ₹{predicted_price:.2f} Lakhs"
print(predict_price("Whitefield", 1200, 2, 2))
print(predict_price("Indira Nagar", 2000, 3, 3))
print(predict_price("Whitefield", 1500, 2, 3))
print(predict_price("Indira Nagar", 1800, 3, 3))
print(predict_price("Electronic City", 1000, 2, 2))
joblib.dump(rf_model, "house_price_model.pkl")
joblib.dump(X.columns, "columns.pkl")  
print("Model and columns saved successfully!")