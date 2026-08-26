"""
Week 3 - Automotive Data Analysis Internship
Predictive modeling: training and evaluating ML models to predict MPG(Miles per gallon).

Building on Week 2's EDA, weight/displacement/horsepower/cylinders showed strongcorrelation with mpg (all -0.75 to -0.83) 
and model_year showed a positive trend(+0.58). Also flagged multicollinearity between the engine-spec variables.

Plan:
  1. Baseline model: Linear Regression - interpretable, and the relationships
     found in Week 2 were largely linear.
  2. Advanced model: Random Forest Regressor - handles multicollinearity better
     than linear regression and can capture non-linear interactions
     (e.g. weight vs mpg wasn't perfectly linear in the Week 2 scatter plot).
  3. Compare both using RMSE, MAE and R^2 on a held-out test set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 1. Load the cleaned dataset from Week 2
# ---------------------------------------------------------------
df = pd.read_csv('week2_data_and_eda/auto_mpg_cleaned.csv')
print(f"Loaded cleaned dataset: {df.shape}")

# ---------------------------------------------------------------
# 2. Feature selection
# ---------------------------------------------------------------
# Dropping 'name' (just a text identifier, not predictive on its own).
# Keeping all the numeric engine specs even though Week 2 showed they're
# correlated with each other - Random Forest handles that fine, and for
# Linear Regression it's still useful to show the effect of multicollinearity
# rather than hide it.
features = ['cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'model_year', 'origin']
target = 'mpg'

X = df[features]
y = df[target]

numeric_features = ['cylinders', 'displacement', 'horsepower', 'weight',
                     'acceleration', 'model_year']
categorical_features = ['origin']

# ---------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------
# 80/20 split, fixed random_state so results are reproducible run to run.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

# ---------------------------------------------------------------
# 4. Preprocessing pipeline
# ---------------------------------------------------------------
# Scale numeric features (matters for Linear Regression, harmless for RF)
# and one-hot encode 'origin' since it's categorical (usa/japan/europe).
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# ---------------------------------------------------------------
# 5. Model 1 - Linear Regression (baseline)
# ---------------------------------------------------------------
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print("\n--- Linear Regression ---")
print(f"RMSE: {lr_rmse:.3f}")
print(f"MAE:  {lr_mae:.3f}")
print(f"R^2:  {lr_r2:.3f}")

# ---------------------------------------------------------------
# 6. Model 2 - Random Forest Regressor (advanced)
# ---------------------------------------------------------------
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("\n--- Random Forest Regressor ---")
print(f"RMSE: {rf_rmse:.3f}")
print(f"MAE:  {rf_mae:.3f}")
print(f"R^2:  {rf_r2:.3f}")

# ---------------------------------------------------------------
# 7. Save metrics summary
# ---------------------------------------------------------------
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE': [lr_rmse, rf_rmse],
    'MAE': [lr_mae, rf_mae],
    'R2': [lr_r2, rf_r2]
})
results.to_csv('Week3_Predictive_Modeling/model_comparison.csv', index=False)
print("\n--- Model Comparison ---")
print(results)

# ---------------------------------------------------------------
# 8. Visualizations
# ---------------------------------------------------------------

# Fig 1 - Actual vs Predicted (both models side by side)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, name in zip(axes, [lr_preds, rf_preds], ['Linear Regression', 'Random Forest']):
    ax.scatter(y_test, preds, alpha=0.6, color='#2b6cb0')
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)  # perfect-prediction line
    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.set_title(name)
plt.suptitle('Actual vs. Predicted MPG', fontsize=14)
plt.tight_layout()
plt.savefig('Week3_Predictive_Modeling/fig1_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2 - Residual plot (Random Forest, the better model)
residuals = y_test - rf_preds
plt.figure(figsize=(8, 5))
sns.scatterplot(x=rf_preds, y=residuals, alpha=0.6, color='#dd6b20')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Predicted MPG')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residual Plot - Random Forest Model')
plt.savefig('Week3_Predictive_Modeling/fig2_residuals.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3 - Model comparison bar chart
plt.figure(figsize=(8, 5))
x = np.arange(len(results))
width = 0.35
plt.bar(x - width/2, results['RMSE'], width, label='RMSE', color='#2b6cb0')
plt.bar(x + width/2, results['MAE'], width, label='MAE', color='#dd6b20')
plt.xticks(x, results['Model'])
plt.ylabel('Error (mpg)')
plt.title('Model Comparison - Prediction Error')
plt.legend()
plt.savefig('Week3_Predictive_Modeling/fig3_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 4 - Feature importance (Random Forest only - it exposes this directly)
feature_names = (numeric_features +
                  list(rf_pipeline.named_steps['preprocessor']
                       .named_transformers_['cat']
                       .get_feature_names_out(categorical_features)))
importances = rf_pipeline.named_steps['model'].feature_importances_

imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(imp_df['feature'], imp_df['importance'], color='#38a169')
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest Model')
plt.tight_layout()
plt.savefig('Week3_Predictive_Modeling/fig4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations and model_comparison.csv saved in 'Week3_Predictive_Modeling/' folder.")