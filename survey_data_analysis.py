# %% 
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# %% 
# Load the survey data and schema file
df = pd.read_csv('dev_s19/survey_results_public.csv')
schema_df  = pd.read_csv('dev_s19/survey_results_schema.csv', index_col='Column')

# %%
# Set display options to show more rows and columns
pd.set_option('display.max_columns', 85)
pd.set_option('display.max_rows', 85)

# %%
# Explore the first few rows of the schema to understand the questions
schema_df.head()

# %%
# Display the entire schema dataframe for reference
schema_df

# %%
# Check the shape of the survey data
df.shape

# %%
# Check the column names of the survey data
df.columns

# %%
# Check the distribution of responses to the "Hobbyist" question
df['Hobbyist'].value_counts()

# %%
# Select a range of rows and columns to preview some of the data
df.loc[0:2, 'Hobbyist':'Country']

# %%
# Create a small dataframe manually for testing purposes
people = { 
    'First': ['Daler', 'Tima', 'Zafar'],
    'Last': ['Khamidov','Atajanov', 'Zugurov'],
    'Age':[24,25,23]
}
df_people = pd.DataFrame(people)
df_people['First']

# %%
# Check the types of the "Age" column and the dataframe
print(type(df_people['Age']),  type(df_people))

# %%
# Display selected columns from the people dataframe
print(df_people[['First','Age']])

# %%
# Check the type of the selected subset of the dataframe
type(df_people[['First','Age']])

# %%
# Use .iloc to retrieve the first row and check the shape of the dataframe
print(df.iloc[0])
print(df.shape)

# %%
# Reload the survey data with 'Respondent' as the index
df = pd.read_csv('dev_s19/survey_results_public.csv', index_col='Respondent')
schema_df = pd.read_csv('dev_s19/survey_results_schema.csv', index_col='Column')

# %%
# Check a specific entry in the schema to understand a particular question
schema_df.loc['MgrIdiot', 'QuestionText']

# %%
# Sort the schema by index in descending order
schema_df.sort_index(ascending=False)

# %%
# Re-create the small 'people' dataframe
df1 = pd.DataFrame(people)
df1

# %%
# Apply a filter on the dataframe
filt = (df1['Last'] == 'Khamidov')

# %%
# Display filtered data and the 'Age' column for the filtered results
df1[filt]
print(df1.loc[filt, 'Age'])

# %%
# Reload the survey data and set the 'Respondent' column as the index
df = pd.read_csv('dev_s19/survey_results_public.csv')
schema_df = pd.read_csv('dev_s19/survey_results_schema.csv', index_col='Column')
pd.set_option('display.max_columns', 85)
pd.set_option('display.max_rows', 85)
df.set_index('Respondent', inplace=True)

# %%
# Filtering high salaries (>500,000) and selecting specific countries
high_salary = (df['ConvertedComp'] > 500000)
countries = ['United States', 'India', 'Uzbekistan']
filt_c = df['Country'].isin(countries)

# %%
# Display rows with high salaries along with the 'Country' and 'ConvertedComp' columns
df.loc[high_salary, ['Country', 'ConvertedComp']]

# %%
# Display rows from the selected countries
df.loc[filt_c, ['Country']]

# %%
# Filter rows where the respondent worked with Python
filt_by_str = df['LanguageWorkedWith'].str.contains('Python', na=False)
filt_by_str

# %% [Advanced Data Analysis and Machine Learning]
# Feature selection for machine learning - predicting salary based on selected features
# We will use columns like 'Age', 'YearsCodePro', 'Country', 'WorkWeekHrs', etc.

# Keep relevant columns for analysis
ml_df = df[['Age', 'YearsCodePro', 'WorkWeekHrs', 'Country', 'ConvertedComp']].copy()

# Drop rows with missing values
ml_df.dropna(inplace=True)

# Convert categorical 'Country' into dummy variables
ml_df = pd.get_dummies(ml_df, columns=['Country'], drop_first=True)

# Split data into features (X) and target (y)
X = ml_df.drop('ConvertedComp', axis=1)
y = ml_df['ConvertedComp']

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# %%
# Plot the actual vs predicted salaries
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', lw=2)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary using RandomForest")
plt.show()

