# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()

# Create a DataFrame from the dataset
# The data contains feature values, and we use the feature names as column names
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target values (species) to the DataFrame as a new column
iris_df['species'] = iris.target

# Map the target integers to their corresponding species names for better readability
iris_df['species'] = iris_df['species'].map(dict(enumerate(iris.target_names)))

# Display the first 5 rows of the DataFrame
print("First 5 rows of the dataset:")
print(iris_df.head())

# Basic exploration of the dataset
# 1. Shape of the dataset
print("\nShape of the dataset (rows, columns):", iris_df.shape)

# 2. Column names
print("\nColumn names:")
print(iris_df.columns)

# 3. Summary of the dataset
print("\nSummary statistics:")
print(iris_df.describe())

# 4. Check for missing values
print("\nCheck for missing values:")
print(iris_df.isnull().sum())

# 5. Count of each species (target class distribution)
print("\nClass distribution (species):")
print(iris_df['species'].value_counts())

# 6. Information about the dataset
print("\nDataset information:")
print(iris_df.info())
