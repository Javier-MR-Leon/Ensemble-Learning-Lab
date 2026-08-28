import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_and_treat_outliers(filepath):
    df = pd.read_csv(filepath)
    
    # Dummies para categóricas vistas en el EDA
    dummies = pd.get_dummies(df["feature_22"], prefix="feature_22", drop_first=True).astype(int)
    df = pd.concat([df, dummies], axis=1)
    
    # Eliminar colinealidad vistas en el EDA
    df = df.drop(columns=["feature_0", "feature_21", "feature_22"])
    
    num_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Detección de Outliers (IQR) y reemplazo con NaN
    Q1 = df[num_vars].quantile(0.25)
    Q3 = df[num_vars].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (df[num_vars] < lower_bound) | (df[num_vars] > upper_bound)
    
    df_for_impute = df[num_vars].mask(outlier_mask, np.nan)
    
    # KNN Imputation
    print("[*] Imputando outliers con KNN Imputer...")
    imputer = KNNImputer(n_neighbors=5)
    imputed_array = imputer.fit_transform(df_for_impute)
    
    df_imputed = df.copy()
    df_imputed[num_vars] = pd.DataFrame(imputed_array, columns=num_vars, index=df.index)
    
    X = df_imputed.drop(columns=["target"]).select_dtypes(exclude=["object"])
    y = df["target"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_base_models(X_train, y_train, X_test, y_test, results_dict):
    base_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
    }
    
    print(f"{'Modelo':<22} | {'R²':<8} | {'RMSE':<8}")
    print("-" * 44)
    
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Guardar en el diccionario global
        results_dict[name] = {
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae
        }
        
        print(f"{name:<22} | {r2:.4f}   | {rmse:.4f}")

def main():
    X_train, X_test, y_train, y_test = load_and_treat_outliers("../data/dataset_regresion.csv")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- MODELOS BASE ---
    evaluate_base_models(X_train_scaled, y_train, X_test_scaled, y_test, global_results)
    
    # --- GRADIENT BOOSTING REGRESSOR ---
    print("\n[*] Entrenando Gradient Boosting Regressor (GridSearch)...")
    gbr = GradientBoostingRegressor(random_state=42)
    param_grid_gb = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1]
    }
    grid_search = GridSearchCV(gbr, param_grid_gb, scoring="r2", cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_gb = grid_search.best_estimator_
    y_pred_gb = best_gb.predict(X_test_scaled)
    r2_gb = r2_score(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    
    print("--- GRADIENT BOOSTING RESULTADOS ---")
    print(f"Mejores Parámetros: {grid_search.best_params_}")
    print(f"R²: {r2_gb:.4f} | RMSE: {rmse_gb:.4f}")
    
    # --- STACKING REGRESSOR ---
    base_learners = [
        ("lr", LinearRegression()),
        ("dt", DecisionTreeRegressor(random_state=42)),
        ("knn", KNeighborsRegressor(n_neighbors=5))
    ]
    meta_regressor = Ridge(alpha=1.0)
    stacking_regressor = StackingRegressor(
        estimators=base_learners, 
        final_estimator=meta_regressor, 
        cv=5, 
        n_jobs=-1
    )
    
    stacking_regressor.fit(X_train_scaled, y_train)
    y_pred_stack = stacking_regressor.predict(X_test_scaled)
    r2_stack = r2_score(y_test, y_pred_stack)
    rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    mae_stack = mean_absolute_error(y_test, y_pred_stack)
    
    global_results["Stacking Regressor"] = {
        "R²": r2_stack,
        "RMSE": rmse_stack,
        "MAE": mae_stack
    }

    # --- COMPARACIÓN GLOBAL ---
    print("\n" + "="*50)
    print("--- COMPARACIÓN GLOBAL DE MODELOS (REGRESIÓN) ---")
    print("="*50)
    
    summary_df = pd.DataFrame(global_results).T.sort_values(by="R²", ascending=False)
    print(summary_df.round(4).to_string())

if __name__ == "__main__":
    main()
