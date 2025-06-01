# ------------------------------------------------------------------
# 0. Librerías
# ------------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

# --- NUEVO / XGBoost ---
from xgboost import XGBClassifier        # pip install xgboost  (si aún no lo tienes instalado)

# ------------------------------------------------------------------
# 1. Cargar datos y preparar variables
# ------------------------------------------------------------------
file_path = r"C:\Users\ricar\Downloads\DATOS COMPLETOS 25-05.xlsx"
data      = pd.read_excel(file_path, sheet_name='Hoja1')

selected_features = ['Superficie lingual cm2', 'Distancia Piel a Epiglotis', 'Distancia Piel a Hueso Hioides',
                     'Grosor de la lengua', 'EDAD', 'IMC kg/m2']
X = data[selected_features]
y = data['DIFICULTAD VLD']

# Rangos de referencia (por si quieres mostrarlos en tu interfaz)
reference_ranges = {f: f"Rango: {data[f].min()} - {data[f].max()}" for f in selected_features}

# Normalizar (solo las seleccionadas)
scaler        = StandardScaler()
X_normalized  = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train / test
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.20, random_state=42, stratify=y
)

# Validación cruzada estratificada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ------------------------------------------------------------------
# 2. Logistic Regression
# ------------------------------------------------------------------
param_grid_log_reg = {'C': [0.01, 0.1, 1, 10, 100],
                      'penalty': ['l1'],
                      'solver': ['liblinear']}

grid_log_reg = GridSearchCV(
    LogisticRegression(random_state=42, multi_class='ovr'),
    param_grid_log_reg, cv=cv, scoring='roc_auc_ovr'
)
grid_log_reg.fit(X_train, y_train)

log_reg_best   = grid_log_reg.best_estimator_
y_pred_log_reg = log_reg_best.predict(X_test)
y_prob_log_reg = log_reg_best.predict_proba(X_test)
cv_scores_log_reg = cross_val_score(log_reg_best, X_normalized, y, cv=cv,
                                    scoring='roc_auc_ovr')

# ------------------------------------------------------------------
# 3. SVM
# ------------------------------------------------------------------
param_grid_svm = {'C': [0.1, 1, 10],
                  'kernel': ['linear', 'rbf'],
                  'gamma': [0.01, 0.1, 1]}

grid_svm = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid_svm, cv=cv, scoring='roc_auc_ovr'
)
grid_svm.fit(X_train, y_train)

svm_best   = grid_svm.best_estimator_
y_pred_svm = svm_best.predict(X_test)
y_prob_svm = svm_best.predict_proba(X_test)
cv_scores_svm = cross_val_score(svm_best, X_normalized, y, cv=cv,
                                scoring='roc_auc_ovr')

# ------------------------------------------------------------------
# 4. Random Forest (variables seleccionadas)
# ------------------------------------------------------------------
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth'   : [None, 4, 8],
    'max_features': ['sqrt', 'log2'],
    'criterion'   : ['gini', 'entropy']
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf, cv=cv, scoring='roc_auc_ovr'
)
grid_rf.fit(X_train, y_train)

rf_best   = grid_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)
y_prob_rf = rf_best.predict_proba(X_test)
cv_scores_rf = cross_val_score(rf_best, X_normalized, y, cv=cv,
                               scoring='roc_auc_ovr')

# ------------------------------------------------------------------
# 5. XGBoost
# ------------------------------------------------------------------
num_classes = len(np.unique(y))
param_grid_xgb = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [3, 4, 5],
    'learning_rate'    : [0.01, 0.1, 0.2],
    'subsample'        : [0.8, 1.0],
    'colsample_bytree' : [0.8, 1.0],
    'gamma'            : [0, 0.1, 0.2]
}

grid_xgb = GridSearchCV(
    XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=num_classes,
        random_state=42,
        use_label_encoder=False
    ),
    param_grid_xgb, cv=cv, scoring='roc_auc_ovr', n_jobs=-1, verbose=0
)
grid_xgb.fit(X_train, y_train)

xgb_best   = grid_xgb.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
y_prob_xgb = xgb_best.predict_proba(X_test)
cv_scores_xgb = cross_val_score(xgb_best, X_normalized, y, cv=cv,
                                scoring='roc_auc_ovr')

# ------------------------------------------------------------------
# 6. Importancia de variables (solo seleccionadas)
# ------------------------------------------------------------------
feature_importances = rf_best.feature_importances_
importance_df = (
    pd.DataFrame({'Variable': X.columns,
                  'Importancia': feature_importances})
      .sort_values(by='Importancia', ascending=False)
      .reset_index(drop=True)
)

print("\nImportancia de las variables en el modelo Random Forest (seleccionadas):")
print(importance_df)

# ------------------------------------------------------------------
# 6-bis. Random Forest con *todas* las columnas predictoras
# ------------------------------------------------------------------
# Asegúrate de que todas las columnas sean numéricas o previamente codificadas
X_all = data.drop(columns=['DIFICULTAD VLD'])
scaler_all = StandardScaler()
X_all_norm = pd.DataFrame(scaler_all.fit_transform(X_all),
                          columns=X_all.columns)

# Train/Test y búsqueda de hiper-parámetros
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all_norm, y, test_size=0.20, random_state=42, stratify=y
)

grid_rf_all = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf, cv=cv, scoring='roc_auc_ovr'
)
grid_rf_all.fit(X_train_all, y_train_all)

rf_best_all = grid_rf_all.best_estimator_

# Importancia con *todas* las variables
feat_imp_all = rf_best_all.feature_importances_
importance_df_all = (
    pd.DataFrame({'Variable': X_all.columns,
                  'Importancia': feat_imp_all})
      .sort_values(by='Importancia', ascending=False)
      .reset_index(drop=True)
)

print("\nImportancia de las variables en el Random Forest (todas):")
print(importance_df_all)

# ------------------------------------------------------------------
# 6-ter.  Función genérica para graficar importancias
# ------------------------------------------------------------------
def plot_feature_importance(df, title):
    """
    Dibuja un bar-plot horizontal con las variables ordenadas
    de mayor a menor importancia.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(df['Variable'], df['Importancia'])
    plt.gca().invert_yaxis()          # la más importante arriba
    plt.xlabel('Importancia', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='x', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# --- Gráficos de importancias ---
plot_feature_importance(
    importance_df_all,
    'Importancia de variables (Random Forest – TODAS)'
)

plot_feature_importance(
    importance_df,
    'Importancia de variables (Random Forest – SELECCIONADAS)'
)

# ------------------------------------------------------------------
# 7. Métricas AUC y reportes
# ------------------------------------------------------------------
auc_log_reg = roc_auc_score(y_test, y_prob_log_reg, multi_class='ovr')
auc_svm     = roc_auc_score(y_test, y_prob_svm,     multi_class='ovr')
auc_rf      = roc_auc_score(y_test, y_prob_rf,      multi_class='ovr')
auc_xgb     = roc_auc_score(y_test, y_prob_xgb,     multi_class='ovr')

def print_summary(name, y_true, y_pred, cv_scores):
    print(f"\n{name} (mejor modelo):\n",
          classification_report(y_true, y_pred))
    print("AUC Promedio de Validación Cruzada:", cv_scores.mean())
    print("Desviación Estándar del AUC de Validación Cruzada:", cv_scores.std())

print_summary("Logistic Regression", y_test, y_pred_log_reg, cv_scores_log_reg)
print_summary("SVM",                y_test, y_pred_svm,      cv_scores_svm)
print_summary("Random Forest",      y_test, y_pred_rf,       cv_scores_rf)
print_summary("XGBoost",            y_test, y_pred_xgb,      cv_scores_xgb)

# ------------------------------------------------------------------
# 8. Curvas ROC
# ------------------------------------------------------------------
# Si tu problema es binario toma [:,1]; para multi-clase puedes promediar o
# graficar una curva por clase. Aquí se mantiene el enfoque binario.
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_prob_log_reg[:, 1], pos_label=1)
fpr_svm,     tpr_svm,     _ = roc_curve(y_test, y_prob_svm[:, 1],     pos_label=1)
fpr_rf,      tpr_rf,      _ = roc_curve(y_test, y_prob_rf[:, 1],      pos_label=1)
fpr_xgb,     tpr_xgb,     _ = roc_curve(y_test, y_prob_xgb[:, 1],     pos_label=1)

plt.figure(figsize=(12, 8))
plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {auc_log_reg:.2f})', linewidth=2)
plt.plot(fpr_svm,     tpr_svm,     label=f'SVM (AUC = {auc_svm:.2f})',                 linewidth=2)
plt.plot(fpr_rf,      tpr_rf,      label=f'Random Forest (AUC = {auc_rf:.2f})',        linewidth=2)
plt.plot(fpr_xgb,     tpr_xgb,     label=f'XGBoost (AUC = {auc_xgb:.2f})',             linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate',  fontsize=14)
plt.title('ROC Curve Comparison', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.show()

