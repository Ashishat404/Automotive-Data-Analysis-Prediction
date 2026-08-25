"""
Week 2 - Automotive Data Analysis Internship
Data acquisition, cleaning and exploratory analysis on the Auto MPG dataset.

I'm using the public Auto MPG dataset (mirrored on GitHub from the original
UCI source) since it has a good mix of numeric specs (weight, horsepower,
displacement) and a categorical field (origin) to explore.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. Get the data
# ---------------------------------------------------------------
print("Fetching Auto MPG dataset...")
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
df = pd.read_csv(url)

# keep an untouched copy before I start cleaning anything
df.to_csv('week2_data_and_eda/auto_mpg_raw.csv', index=False)
print(f"Data acquired successfully. Shape: {df.shape}")

# ---------------------------------------------------------------
# 2. Quick look at what I'm working with
# ---------------------------------------------------------------
print("\n--- Initial Inspection ---")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())
# horsepower is the only column with gaps - 6 rows out of 398

# ---------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------
# Fill missing horsepower with the median horsepower of vehicles that have
# the same number of cylinders. Using the overall median felt wrong here
# since a 4-cylinder and an 8-cylinder engine have very different power
# outputs, so grouping first makes the fill more realistic.
df['horsepower'] = df.groupby('cylinders')['horsepower'].transform(
    lambda x: x.fillna(x.median())
)

# Outlier check using the standard IQR rule (anything outside
# Q1 - 1.5*IQR to Q3 + 1.5*IQR gets flagged). Not removing these - in a car
# dataset they're usually real vehicles (sports cars, economy cars) rather
# than bad data, but good to know how many there are.
print("\n--- Outlier Check (IQR method) ---")
for col in ['horsepower', 'weight', 'mpg']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Outliers detected in {col}: {len(outliers)}")

# Correlation with mpg - this is what actually goes into the report's
# findings section, so printing it clearly here.
numeric_df = df.select_dtypes(include=[np.number])
print("\n--- Correlation with MPG (sorted) ---")
print(numeric_df.corr()['mpg'].sort_values())

# save the cleaned version for the report and for next week's modeling
df.to_csv('week2_data_and_eda/auto_mpg_cleaned.csv', index=False)
print("\nCleaned data saved as 'auto_mpg_cleaned.csv'.")

# ---------------------------------------------------------------
# 4. EDA charts
# ---------------------------------------------------------------

# Fig 1 - how is mpg distributed overall?
plt.figure(figsize=(8, 5))
sns.histplot(df['mpg'], kde=True, color='teal', bins=20)
plt.title('Distribution of Vehicle Miles Per Gallon (MPG)', fontsize=14)
plt.xlabel('Miles Per Gallon (MPG)')
plt.ylabel('Frequency')
plt.savefig('week2_data_and_eda/fig1_mpg_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 2 - does engine size (cylinders) affect fuel economy?
plt.figure(figsize=(8, 5))
sns.boxplot(x='cylinders', y='mpg', data=df, palette='Set2')
plt.title('Fuel Economy (MPG) by Engine Cylinder Count', fontsize=14)
plt.xlabel('Number of Cylinders')
plt.ylabel('Miles Per Gallon (MPG)')
plt.savefig('week2_data_and_eda/fig2_cylinders_vs_mpg.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 3 - weight vs mpg, split by origin, sized by displacement
plt.figure(figsize=(8, 5))
sns.scatterplot(x='weight', y='mpg', hue='origin', size='displacement', data=df, alpha=0.8, palette='tab10')
plt.title('Vehicle Weight vs. MPG (Grouped by Origin)', fontsize=14)
plt.xlabel('Vehicle Weight (lbs)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.savefig('week2_data_and_eda/fig3_weight_vs_mpg.png', dpi=300, bbox_inches='tight')
plt.close()

# Fig 4 - correlation heatmap across all numeric columns
plt.figure(figsize=(9, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Automotive Attributes', fontsize=14)
plt.savefig('week2_data_and_eda/fig4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll visualizations generated and saved in 'week2_data_and_eda/' folder.")