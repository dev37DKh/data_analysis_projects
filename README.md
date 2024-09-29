Survey Data Analysis and Machine Learning
Overview

This project performs exploratory data analysis and applies machine learning techniques to survey data from the Stack Overflow Developer Surveys. It includes data preprocessing, visualization, clustering, and predictive modeling to extract insights from the data.
Contents

    Data Preprocessing: Handles missing values and encodes categorical features.
    Exploratory Data Analysis (EDA): Provides descriptive statistics and visualizations, including correlation heatmaps and PCA.
    Clustering: Applies K-Means clustering to categorize respondents based on features.
    Predictive Modeling: Utilizes Random Forest for classification and regression tasks to predict job roles and compensation.

Datasets

    dev_s19/survey_results_public.csv: Contains survey results from 2019.
    dev_s20/survey_results_public.csv: Contains survey results from 2020.

Requirements

To run this project, ensure you have the following libraries installed:

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn

You can install the required libraries using pip:

bash

pip install pandas numpy matplotlib seaborn scikit-learn

Usage

    Clone the repository:

    bash

git clone https://github.com/yourusername/survey-data-analysis.git
cd survey-data-analysis

Ensure you have the datasets in the correct directory structure:

bash

dev_s19/survey_results_public.csv
dev_s20/survey_results_public.csv

Run the Jupyter Notebook:

bash

    jupyter notebook

    Open the notebook file and execute the cells to perform the analysis.

Results

The project provides insights into the relationships between different variables in the survey data and includes predictions on job roles and salary compensation based on various features.

    Correlation heatmap showing relationships between numeric variables.
    PCA scatter plots visualizing dimensionality reduction.
    Clustering visualization indicating groupings within the data.
    Model evaluation metrics such as classification reports and confusion matrices.

Conclusion

This project demonstrates the application of data analysis and machine learning techniques to gain insights from survey data. It showcases how to preprocess data, visualize relationships, perform clustering, and build predictive models.
