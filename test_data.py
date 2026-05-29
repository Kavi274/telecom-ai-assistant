import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "data" / "telecom.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nFirst 3 rows:")
print(df.head(3))