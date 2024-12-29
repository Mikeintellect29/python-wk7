# Task 1: Load and Explore the Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_data = load_iris(as_frame=True)
df = iris_data.frame

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Info:")
print(df.info())
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean the dataset (if there were missing values)
# In this case, there are no missing values in the Iris dataset.

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics of the Numerical Columns:")
print(df.describe())

# Perform groupings
grouped = df.groupby('target')['sepal length (cm)'].mean()
print("\nMean Sepal Length per Species:")
print(grouped)

# Identify patterns
print("\nPatterns:")
print("From the grouping, we observe that the average sepal length differs between species.")

# Task 3: Data Visualization

# Line chart - Example with a fake time-series column for visualization
df['index'] = df.index  # Fake index for demonstration
plt.figure(figsize=(10, 5))
plt.plot(df['index'], df['sepal length (cm)'], label="Sepal Length", color='blue')
plt.title("Trend of Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar chart - Average petal length per species
plt.figure(figsize=(10, 5))
df['species'] = df['target'].replace({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
avg_petal_length = df.groupby('species')['petal length (cm)'].mean()
avg_petal_length.plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram - Distribution of Sepal Width
plt.figure(figsize=(10, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='purple', alpha=0.7)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot - Sepal Length vs. Petal Length
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='bright')
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
