import pandas as pd

# Load the CSV file
dataset = pd.read_csv("fastAPI-food-delivery-backend/dataset.csv", compression="gzip")

# View the first few rows to check the recipes
print(dataset.head())
