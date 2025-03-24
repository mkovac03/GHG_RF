# 🌍 GHG Flux Modeling with Random Forests

This repository simulates and models greenhouse gas (GHG) flux data using synthetic environmental variables. It includes tools for modeling, imputing missing data, evaluating model performance, and interpreting predictions using SHAP values.

---

## 📁 Contents
- `notebook.ipynb` or script file with the complete workflow
- `shap_force_plot_sample.html` – SHAP force plot for one instance
- `shap_force_plot_full.html` – SHAP force plots for all test samples
- `README.md` – Project description and workflow

---

## 🚀 Features
- ✅ Generate synthetic seasonal environmental data
- ✅ Simulate GHG fluxes with noise and nonlinearity
- ✅ Impute missing GHG flux values using Random Forest
- ✅ Train & tune models via `GridSearchCV`
- ✅ Visualize predictions with uncertainty
- ✅ Interpret model using SHAP (summary, dependence, force, waterfall)
- ✅ Export interactive plots (e.g., SHAP force plot as HTML)

---

## 🛠️ Requirements
Install required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn shap
```

---

## 🧪 Workflow Overview

### 🔹 Data Simulation
- Synthetic dataset over 1000 days
- Environmental variables with distinct seasonal cycles:
  - Soil Temperature
  - Soil Moisture
  - NDVI
  - Air Temperature
  - Precipitation
  - Land Cover (categorical)
- GHG flux computed as a nonlinear function of these variables + noise

### 🔹 Missing Data & Imputation
- 20% of GHG flux values are removed randomly
- Random Forest trained on non-missing data
- Imputed values are validated using:
  - MAE, RMSE, R²
  - Scatter plot and time series visualization with error bars and ribbons

### 🔹 Model Training & Evaluation
- `RandomForestRegressor` optimized using `GridSearchCV`
- Best model is selected based on 5-fold cross-validation R² score
- Performance metrics on test set:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score

### 🔹 Model Interpretability with SHAP
- Global feature importance (`summary_plot` with bar & dot styles)
- Dependence plots (e.g., `soil_temp` vs SHAP value)
- Waterfall plots (breakdown of individual predictions)
- Force plots (single instance & batch)
- All SHAP force plots saved as HTML

### 🔹 Uncertainty Quantification
- Per-point uncertainty via standard deviation across all trees
- Predicted vs. observed plot includes uncertainty error bars

### 🔹 Tree Visualization
- One decision tree from the trained forest is visualized using `plot_tree()`

---

## 📊 Example Visuals
> Include screenshots or links to figures here (optional)

---

## 📌 Notes
- Designed for experimentation and teaching
- Can be adapted for real-world GHG or remote sensing data
- Interactive SHAP plots require running in a Jupyter Notebook and trusting the notebook

---

## 👨‍💻 Author
Created by Gyula Mate Kovacs 
---

## 📄 License
MIT License
