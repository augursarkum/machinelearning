# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['species'] = iris.target

# Display the first few rows of the dataset
print("First few rows of the Iris dataset:")
print(iris_data.head())

# Basic statistics of the dataset
print("\nBasic statistics of the dataset:")
print(iris_data.describe())

# Count of each species
print("\nCount of each species:")
print(iris_data['species'].value_counts())

# Pairplot to visualize the relationships between features
sns.pairplot(iris_data, hue='species')
plt.show()

# Correlation matrix
print("\nCorrelation matrix:")
print(iris_data.corr())

# Heatmap of the correlation matrix
sns.heatmap(iris_data.corr(), annot=True, cmap='coolwarm')
plt.show()
