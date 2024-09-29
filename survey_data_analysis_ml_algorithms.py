#%% 
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

#%% 
# Load the datasets
df1 = pd.read_csv('dev_s19/survey_results_public.csv')
df2 = pd.read_csv('dev_s20/survey_results_public.csv')

# Display the first few rows
print("Dataset 1:")
display(df1.head())
print("Dataset 2:")
display(df2.head())

#%% 
# Data Preprocessing
# Check for missing values
print(df1.isnull().sum())
print(df2.isnull().sum())

# Fill missing values with 'Unknown' or mode/mean as appropriate
df1.fillna('Unknown', inplace=True)
df2.fillna('Unknown', inplace=True)

#%% 
# Feature Engineering
# Label Encoding categorical columns for machine learning purposes
label_encoder = LabelEncoder()

categorical_columns = ['MainBranch', 'Hobbyist', 'Country', 'Student', 'EdLevel', 
                       'Employment', 'Gender', 'Ethnicity']
for col in categorical_columns:
    df1[col] = label_encoder.fit_transform(df1[col])
    df2[col] = label_encoder.fit_transform(df2[col])

#%% 
# Descriptive Statistics
# Basic Statistics
print("Descriptive statistics for Dataset 1:")
print(df1.describe())

print("Descriptive statistics for Dataset 2:")
print(df2.describe())

#%% 
# Visualization: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df1.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap for Dataset 1')
plt.show()

#%% 
# Advanced Visualization: PCA for Dimensionality Reduction
# Apply PCA to the numeric columns and visualize the top 2 components

# Scaling numeric data for PCA
numeric_columns = df1.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df1_scaled = scaler.fit_transform(df1[numeric_columns])

# Applying PCA
pca = PCA(n_components=2)
df1_pca = pca.fit_transform(df1_scaled)

# PCA scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df1_pca[:, 0], df1_pca[:, 1], c=df1['MainBranch'], cmap='viridis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA of Dataset 1')
plt.colorbar(label='MainBranch')
plt.show()

#%% 
# Clustering with K-Means
# Apply KMeans clustering on the first dataset
kmeans = KMeans(n_clusters=3, random_state=42)
df1['Cluster'] = kmeans.fit_predict(df1_scaled)

# Visualize clusters using the PCA-reduced components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df1_pca[:, 0], y=df1_pca[:, 1], hue=df1['Cluster'], palette='Set1')
plt.title('K-Means Clusters for Dataset 1')
plt.show()

#%% 
# Machine Learning: Predictive Modeling
# We will use the dataset to predict whether someone is a developer (MainBranch)
X = df1.drop(['MainBranch'], axis=1)  # Features
y = df1['MainBranch']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply RandomForest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#%% 
# Feature Importance
# Visualize the most important features in the random forest model
importance = rf_classifier.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=feature_names)
plt.title('Feature Importance in Random Forest Model')
plt.show()

#%% 
# Salary Prediction Model (Regression Example)
# Let's attempt to predict compensation using a regression model from Dataset 2

# Selecting relevant columns for predicting compensation
X_salary = df2[['YearsCodePro', 'WorkWeekHrs', 'Age', 'Employment', 'EdLevel']]
y_salary = df2['ConvertedComp']

# Fill missing values
X_salary.fillna(X_salary.mean(), inplace=True)
y_salary.fillna(y_salary.mean(), inplace=True)

# Split data into training and testing
X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_salary_scaled = scaler.fit_transform(X_train_salary)
X_test_salary_scaled = scaler.transform(X_test_salary)

# Using Random Forest for regression
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_salary_scaled, y_train_salary)

# Predictions and evaluation
y_pred_salary = rf_regressor.predict(X_test_salary_scaled)

# Mean Absolute Error (MAE) and Mean Squared Error (MSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test_salary, y_pred_salary)
mse = mean_squared_error(y_test_salary, y_pred_salary)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

#%% 
# Conclusion
# In this notebook, it is performed basic descriptive statistics, PCA analysis, clustering with KMeans, 
# and predictive modeling with Random Forest for both classification and regression tasks. 
# I also evaluated the importance of features and model performance through metrics like accuracy, MAE, and MSE.
