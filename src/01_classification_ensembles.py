import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import classification_report, f1_score

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Feature Engineering & Encoding
    df["feature_21_encoded"] = OrdinalEncoder(
        categories=[["CatA_0", "CatA_1", "CatA_2", "CatA_3"]]
    ).fit_transform(df[["feature_21"]])
    
    df = pd.get_dummies(df, columns=["feature_22"], drop_first=True)
    
    # Eliminar colinealidad (feature_6 - Esto lo vimos en el EDA (que no se incluye)) y variables categóricas originales
    df = df.drop(columns=["feature_6", "feature_21"])
    
    X = df.drop(columns=["target"]).select_dtypes(exclude=["object"])
    y = df["target"]
    
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def evaluate_base_models(X_train, y_train, X_test, y_test):
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naïve Bayes": GaussianNB()
    }
    
    print(f"{'Modelo':<22} | {'F1 Macro':<10}")
    print("-" * 36)
    
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        print(f"{name:<22} | {f1_macro:.4f}")

def train_strategic_cascading(X_train, y_train, X_test):
    models = [
        LogisticRegression(max_iter=1000, random_state=42),
        KNeighborsClassifier(n_neighbors=5),
        GradientBoostingClassifier(learning_rate=0.2, max_depth=7, n_estimators=200, random_state=42)
    ]
    
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    y_pred_cascade = None
    
    for i, model in enumerate(models):
        model.fit(X_train_aug, y_train)
        
        if i == len(models) - 1:
            y_pred_cascade = model.predict(X_test_aug)
            break
            
        # Enriquecimiento de features para el siguiente modelo
        probs_train = model.predict_proba(X_train_aug)
        pred_train = np.argmax(probs_train, axis=1).reshape(-1, 1)
        
        probs_test = model.predict_proba(X_test_aug)
        pred_test = np.argmax(probs_test, axis=1).reshape(-1, 1)
        
        X_train_aug = np.hstack([X_train_aug, pred_train])
        X_test_aug = np.hstack([X_test_aug, pred_test])
        
    return y_pred_cascade

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/dataset_clasificacion.csv")
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balanceo de clases (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    # --- MODELOS BASE ---
    evaluate_base_models(X_train_bal, y_train_bal, X_test_scaled, y_test)
    
    # --- GRADIENT BOOSTING (Best Model) ---
    gb_clf = GradientBoostingClassifier(random_state=42)
    param_grid_gb = {
        "n_estimators": [100, 200],
        "learning_rate": [0.1, 0.2],
        "max_depth": [4, 7]
    }
    grid_search = GridSearchCV(gb_clf, param_grid_gb, scoring="f1_macro", cv=5, n_jobs=-1)
    grid_search.fit(X_train_bal, y_train_bal)
    
    best_gb = grid_search.best_estimator_
    y_pred_gb = best_gb.predict(X_test_scaled)
    
    print("\n--- GRADIENT BOOSTING RESULTADOS ---")
    print(f"Mejores Hiperparámetros: {grid_search.best_params_}")
    print(classification_report(y_test, y_pred_gb))
    
    # --- STACKING CLASSIFIER ---
    base_learners = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("knn", KNeighborsClassifier()),
        ("nb", GaussianNB())
    ]
    stack_model = StackingClassifier(estimators=base_learners, cv=5, n_jobs=-1, passthrough=True)
    stack_model.fit(X_train_bal, y_train_bal)
    y_pred_stack = stack_model.predict(X_test_scaled)
    
    print("--- STACKING RESULTADOS ---")
    print(classification_report(y_test, y_pred_stack))

    # --- CASCADING ESTRATÉGICO ---
    y_pred_cascade = train_strategic_cascading(X_train_bal, y_train_bal, X_test_scaled)
    global_results["Cascading (Estratégico)"] = {
        "Accuracy": accuracy_score(y_test, y_pred_cascade),
        "F1 Macro": f1_score(y_test, y_pred_cascade, average="macro"),
        "F1 Weighted": f1_score(y_test, y_pred_cascade, average="weighted")
    }

    # --- COMPARACIÓN GLOBAL ---
    print("\n" + "="*50)
    print("n--- COMPARACIÓN GLOBAL DE MODELOS ---")
    print("="*50)
    
    # Convertir diccionario a DataFrame y ordenar por F1 Macro
    summary_df = pd.DataFrame(global_results).T.sort_values(by="F1 Macro", ascending=False)
    print(summary_df.round(4).to_string())

if __name__ == "__main__":
    main()
