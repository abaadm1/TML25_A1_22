import pandas as pd
import numpy as np
import requests

# Read test.csv (already in correct submission format: ids + score)
df = pd.read_csv("test.csv")

# Sanity check: print first few rows
print("[INFO] Submitting the following preview from test.csv:")
print(df.head())

# Submit to server
response = requests.post(
    "http://34.122.51.94:9090/mia",
    files={"file": open("test.csv", "rb")},
    headers={"token": "12910150"}  # Replace with actual token if needed
)

# Print server response
print("[INFO] Submission response:", response.json())
