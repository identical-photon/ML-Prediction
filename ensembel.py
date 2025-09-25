import pandas as pd
import numpy as np
from pymatgen.io.vasp import Poscar
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------- Manual Feature Extraction ----------------
def extract_manual_features(structure):
    try:
        lattice = structure.lattice
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        atomic_numbers = [site.specie.Z for site in structure.sites]
        return pd.DataFrame([{
            "mean_atomic_numbers": np.mean(atomic_numbers),
            "max_atomic_numbers": np.max(atomic_numbers),
            "min_atomic_numbers": np.min(atomic_numbers),
            "std_atomic_numbers": np.std(atomic_numbers),
            "a_parameters": a,
            "b_parameters": b,
            "c_parameters": c,
            "alpha_parameters": alpha,
            "beta_parameters": beta,
            "gamma_parameters": gamma,
            "mean_distance_matrix": np.mean(structure.distance_matrix),
            "max_distance_matrix": np.max(structure.distance_matrix),
            "min_distance_matrix": np.min(structure.distance_matrix),
            "std_distance_matrix": np.std(structure.distance_matrix),
        }])
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# ---------------- Load dataset ----------------
df = pd.read_csv("Band_gap_dataset.csv")
X = df.drop(columns=["Formula", "Band Gap"])
y = df["Band Gap"]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- Initial Models ----------------
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
ada = AdaBoostRegressor(random_state=42)

# ---------------- Hyperparameter Tuning ----------------
print("Tuning Random Forest...")
rf_param_grid = {
    "n_estimators": [300, 600, 1000],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]  # no 'auto'
}

rf_search = RandomizedSearchCV(
    rf, rf_param_grid, n_iter=15, cv=3, scoring="r2", n_jobs=-1, random_state=42
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print("Best RF Params:", rf_search.best_params_)

print("Tuning XGBoost...")
xgb_param_grid = {
    "n_estimators": [300, 600, 1000],
    "max_depth": [6, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2]
}
xgb_search = RandomizedSearchCV(xgb, xgb_param_grid, n_iter=15, cv=3, scoring="r2", n_jobs=-1, random_state=42)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print("Best XGB Params:", xgb_search.best_params_)

# ---------------- Stacking Ensemble ----------------
ensemble = StackingRegressor(
    estimators=[("rf", best_rf), ("xgb", best_xgb), ("ada", ada)],
    final_estimator=LinearRegression(),
    n_jobs=-1
)
print("Training Ensemble...")
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nEnsemble - RMSE: {rmse:.3f}, R²: {r2:.3f}")

# ---------------- Feature Importance ----------------
# RF
rf_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
rf_importances.to_csv("rf_feature_importance.csv", header=["Importance"])
plt.figure(figsize=(8,6))
rf_importances.sort_values(ascending=False).head(15).plot(kind="barh")
plt.title("Random Forest Feature Importance (Top 15)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.close()

# XGB
xgb_importances = pd.Series(best_xgb.feature_importances_, index=X.columns)
xgb_importances.to_csv("xgb_feature_importance.csv", header=["Importance"])
plt.figure(figsize=(8,6))
xgb_importances.sort_values(ascending=False).head(15).plot(kind="barh", color="orange")
plt.title("XGBoost Feature Importance (Top 15)")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png")
plt.close()

# Ensemble permutation importance
perm_importance = permutation_importance(ensemble, X_test, y_test, n_repeats=10, random_state=42)
perm_series = pd.Series(perm_importance.importances_mean, index=X.columns)
perm_series.to_csv("ensemble_feature_importance.csv", header=["Importance"])
plt.figure(figsize=(8,6))
perm_series.sort_values(ascending=False).head(15).plot(kind="barh", color="green")
plt.title("Ensemble Permutation Importance (Top 15)")
plt.tight_layout()
plt.savefig("ensemble_feature_importance.png")
plt.close()

# ---------------- Correlation Heatmap ----------------
print("Generating correlation heatmap...")
corr_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.jpg", dpi=300)
plt.close()
print("Correlation heatmap saved as correlation_heatmap.jpg")
# ---------------- Save Correlation Matrices ----------------
# Feature-to-feature correlation
corr_matrix = X.corr()
corr_matrix.to_csv("feature_correlation_matrix.csv")
print("Correlation matrix saved as feature_correlation_matrix.csv")

# Feature + Target correlation
df_corr = pd.concat([X, y], axis=1)
corr_matrix_with_target = df_corr.corr()
corr_matrix_with_target.to_csv("feature_target_correlation_matrix.csv")
print("Correlation matrix (with target) saved as feature_target_correlation_matrix.csv")

# ---------------- Save Model ----------------
joblib.dump(ensemble, "ensemble_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ---------------- Predict new VASP ----------------
vasp_files = ["BaTa4Te3O17.vasp"]
for f in vasp_files:
    try:
        structure = Poscar.from_file(f).structure
        new_feat = extract_manual_features(structure)
        if new_feat is not None:
            new_feat = new_feat.reindex(columns=X.columns, fill_value=0)
            new_feat_scaled = scaler.transform(new_feat)
            pred = ensemble.predict(new_feat_scaled)[0]
            print(f"Predicted Bandgap for {f}: {pred:.3f} eV")
        else:
            print(f"Feature extraction failed for {f}")
    except Exception as e:
        print(f"Error predicting {f}: {e}")
