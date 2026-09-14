# Automotive-Data-Analysis-Prediction

## Project Overview
This repository contains a comprehensive four-week data science pipeline focused on automotive analytics. The project transitions from raw vehicle data acquisition to advanced predictive modeling, leveraging the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. The primary objective is to analyze physical vehicle attributes—such as weight, cylinder count, and engine displacement—to predict fuel efficiency (MPG) and extrapolate these findings into actionable business strategies for automotive manufacturing and predictive maintenance.

**Current Build Environment:** Python 3.13 (August 2026)

## Repository Structure
The project is decoupled into weekly milestone directories to separate documentation, raw data, data processing scripts, and modeling artifacts:

* `week1_research_planning/`: Contains the initial problem statement, project scope, and statistical methodology planning.
* `week2_data_and_eda/`: Houses the data acquisition scripts, cleaned datasets, and exploratory visualizations.
* `week3_predictive_modeling/`: Will contain machine learning pipelines and model evaluation scripts.
* `week4_evaluation_strategy/`: Will hold final business insights and strategic recommendations.

## Execution Phases
* **Week 1: Research & Strategy** 
  Defined the analytical approach, selecting tree-based machine learning algorithms and regression techniques for non-linear automotive sensor data.
* **Week 2: Data Cleaning & Exploratory Data Analysis (EDA)** 
  Acquired the public Auto MPG dataset. Handled null values via cylinder-grouped median imputation and identified anomalies using the statistical Interquartile Range (IQR) method. Extracted multi-variable correlations (e.g., Weight vs. MPG).
* **Week 3: Advanced Modeling** 
  Implementation of feature scaling, stratified train-test splitting, and multivariable regression/classification algorithms using Scikit-Learn.
* **Week 4: Business Intelligence** 
  Consolidating model performance metrics into strategic, real-world recommendations for the automotive sector.

## Local Setup & Installation
To run this project locally and reproduce the analytical environment, execute the following commands in your terminal:

1. Clone the repository:
   `git clone https://github.com/Ashishat404/Automotive-Data-Analysis-Prediction.git`
2. Navigate to the directory:
   `cd Automotive-Data-Analysis-Prediction`
3. Create an isolated virtual environment:
   `python -m venv .venv`
4. Activate the environment:
   `.\.venv\Scripts\activate` (Windows)
5. Install necessary dependencies:
   `pip install pandas numpy matplotlib seaborn scikit-learn`