import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    roc_auc_score
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 또는 'Qt5Agg', 'Agg' 등 다른 백엔드 사용 가능
import seaborn as sns
from scipy.io import arff
import logging

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
for handler in logging.root.handlers:handler.setLevel(logging.INFO)

def evaluate_model(name, model, X_test, y_test, y_pred):
    logging.info(f"=== {name} ===")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    logging.info(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    logging.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")


# 1. ARFF File Loading
arff_file = "dataset_37_diabetes.arff"
data, meta = arff.loadarff(arff_file)

# 2. Convert to pandas DataFrame
df = pd.DataFrame(data)
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].str.decode('utf-8')

logging.info("Data loaded successfully")
logging.info(f"Dataset Info: {df.info()}")

# 3. Data Preprocessing
X = df.drop('class', axis=1)
y = df['class'].map({'tested_positive': 1, 'tested_negative': 0})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Feature Importance Calculation
rf_temp_model = RandomForestClassifier(random_state=42)
rf_temp_model.fit(X_train_smote, y_train_smote)
importance = pd.DataFrame({
    'Feature': meta.names()[:-1],
    'Importance': rf_temp_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
logging.info(f"Feature Importance:\n{importance}")

threshold = 0.05
selected_features = importance[importance['Importance'] > threshold]['Feature'].tolist()
logging.info(f"Selected Features: {selected_features}\n")

X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]
X_train_smote_reduced = X_train_smote[selected_features]

# 4. Decision Tree Model
logging.info("==== Training Decision Tree ====")
param_grid_dt = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': [None, 'balanced']
}

# Step 1 logging messages
logging.info("Step 1: Decision Tree 학습 시작")
grid_search_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    scoring='recall',
    cv=5
)
grid_search_dt.fit(X_train_smote_reduced, y_train_smote)
logging.info("Step 1 완료: Decision Tree 학습 완료")

best_dt_model = grid_search_dt.best_estimator_
logging.info(f"Best Parameters for Decision Tree: {grid_search_dt.best_params_}")

# Evaluate Decision Tree
y_pred_best_dt = best_dt_model.predict(X_test_reduced)
evaluate_model("Decision Tree", best_dt_model, X_test_reduced, y_test, y_pred_best_dt)

# 5. Random Forest Model
logging.info("====Training Random Forest====")

# Step 1 logging messages
logging.info("Step 1: Random Forest 학습 시작")
rf_model_recall = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight='balanced_subsample'
)
rf_model_recall.fit(X_train_smote_reduced, y_train_smote)
logging.info("Step 1 완료: Random Forest 학습 완료")

# Step 2 logging messages
logging.info("Step 2: Random Forest 평가 시작")
y_pred_rf_recall = rf_model_recall.predict(X_test_reduced)
evaluate_model("Random Forest", rf_model_recall, X_test_reduced, y_test, y_pred_rf_recall)
logging.info("Step 2 완료: Random Forest 평가 완료")

# 6. Gradient Boosting Model
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8]
}
logging.info("Training Gradient Boosting with Hyperparameter Tuning")

# Step 1 logging messages
logging.info("Step 1: Gradient Boosting 학습 시작")
random_search_gb = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_grid_gb,
    n_iter=12,
    scoring='recall',
    cv=5
)

# Fitting the model
random_search_gb.fit(X_train_smote_reduced, y_train_smote)
logging.info("Step 1 완료: Gradient Boosting 학습 완료")

# Logging best parameters
best_gb_model = random_search_gb.best_estimator_
logging.info(f"Best Parameters for Gradient Boosting: {random_search_gb.best_params_}")

# Step 2 logging messages
logging.info("Step 2: Gradient Boosting 평가 시작")

# Model evaluation
y_pred_gb = best_gb_model.predict(X_test_reduced)
evaluate_model("Gradient Boosting", best_gb_model, X_test_reduced, y_test, y_pred_gb)
logging.info("Step 2 완료: Gradient Boosting 평가 완료")

# 7. SVM Model
param_grid_svm = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale']
}
logging.info("Training SVM with Hyperparameter Tuning")

# Step 1 logging messages
logging.info("Step 1: SVM 학습 시작")
grid_search_svm = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    param_distributions=param_grid_svm,
    n_iter=4,
    scoring='recall',
    cv=5
)

# Fitting the model
grid_search_svm.fit(X_train_smote_reduced, y_train_smote)
logging.info("Step 1 완료: SVM 학습 완료")

# Logging best parameters
best_svm_model = grid_search_svm.best_estimator_
logging.info(f"Best Parameters for SVM: {grid_search_svm.best_params_}")

# Step 2 logging messages
logging.info("Step 2: SVM 평가 시작")

# Model evaluation
y_pred_svm = best_svm_model.predict(X_test_reduced)
evaluate_model("SVM", best_svm_model, X_test_reduced, y_test, y_pred_svm)
logging.info("Step 2 완료: SVM 평가 완료")

# 8. Naive Bayes Model
top_n_features = 5
logging.info(f"Selecting Top {top_n_features} Important Features")
selected_features_top_n = importance.head(top_n_features)['Feature'].tolist()
logging.info(f"Top {top_n_features} Important Features: {selected_features_top_n}")

# Preparing data with selected features
logging.info("Preparing data with selected features for Naive Bayes")
X_train_smote_reduced_top_n = X_train_smote[selected_features_top_n]
X_test_reduced_top_n = X_test[selected_features_top_n]

# Step 1: Training Naive Bayes Model
logging.info("Step 1: Naive Bayes 학습 시작")
nb_model = GaussianNB()
nb_model.fit(X_train_smote_reduced_top_n, y_train_smote)
logging.info("Step 1 완료: Naive Bayes 학습 완료")

# Step 2: Evaluating Naive Bayes Model
logging.info("Step 2: Naive Bayes 평가 시작")
y_pred_nb = nb_model.predict(X_test_reduced_top_n)
evaluate_model("Naive Bayes", nb_model, X_test_reduced_top_n, y_test, y_pred_nb)
logging.info("Step 2 완료: Naive Bayes 평가 완료")


# 9. Voting Classifier with Optimized Weights
def optimize_weights():
    logging.info("Starting weight optimization for Voting Classifier")
    best_score = 0
    best_weights = None
    for dt_weight in range(1, 3):
        for rf_weight in range(1, 3):
            for gb_weight in range(1, 3):
                for nb_weight in range(1, 3):
                    svm_weight = 1
                    weights = [dt_weight, rf_weight, gb_weight, nb_weight, svm_weight]
                    voting_clf = VotingClassifier(
                        estimators=[
                            ('dt', best_dt_model),
                            ('rf', rf_model_recall),
                            ('gb', best_gb_model),
                            ('nb', nb_model),
                            ('svm', best_svm_model)
                        ],
                        voting='soft',
                        weights=weights
                    )
                    voting_clf.fit(X_train_smote_reduced, y_train_smote)
                    score = recall_score(y_test, voting_clf.predict(X_test_reduced))
                    logging.debug(f"Weights: {weights}, Recall Score: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                        logging.info(f"New Best Weights Found: {best_weights}, Score: {best_score:.4f}")
    logging.info("Weight optimization complete")
    return best_weights


# Optimizing weights
logging.info("Optimizing Voting Classifier Weights")
optimized_weights = optimize_weights()
logging.info(f"Optimized Weights: {optimized_weights}")

# Training Voting Classifier with optimized weights
logging.info("Step 1: Voting Classifier 학습 시작")
voting_clf_optimized = VotingClassifier(
    estimators=[
        ('dt', best_dt_model),
        ('rf', rf_model_recall),
        ('gb', best_gb_model),
        ('nb', nb_model),
        ('svm', best_svm_model)
    ],
    voting='soft',
    weights=optimized_weights
)
voting_clf_optimized.fit(X_train_smote_reduced, y_train_smote)
logging.info("Step 1 완료: Voting Classifier 학습 완료")

# Evaluating Voting Classifier
logging.info("Step 2: Voting Classifier 평가 시작")
y_pred_voting_optimized = voting_clf_optimized.predict(X_test_reduced)
evaluate_model("Voting Ensemble", voting_clf_optimized, X_test_reduced, y_test, y_pred_voting_optimized)
logging.info("Step 2 완료: Voting Classifier 평가 완료")

# 10. Model Comparison Heatmap
logging.info("===Model Comparison Heatmap===")
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Decision Tree': [
        accuracy_score(y_test, y_pred_best_dt),
        precision_score(y_test, y_pred_best_dt, zero_division=0),
        recall_score(y_test, y_pred_best_dt, zero_division=0),
        f1_score(y_test, y_pred_best_dt)
    ],
    'Random Forest': [
        accuracy_score(y_test, y_pred_rf_recall),
        precision_score(y_test, y_pred_rf_recall, zero_division=0),
        recall_score(y_test, y_pred_rf_recall, zero_division=0),
        f1_score(y_test, y_pred_rf_recall)
    ],
    'Gradient Boosting': [
        accuracy_score(y_test, y_pred_gb),
        precision_score(y_test, y_pred_gb, zero_division=0),
        recall_score(y_test, y_pred_gb, zero_division=0),
        f1_score(y_test, y_pred_gb)
    ],
    'SVM': [
        accuracy_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_svm, zero_division=0),
        recall_score(y_test, y_pred_svm, zero_division=0),
        f1_score(y_test, y_pred_svm)
    ],
    'Naive Bayes': [
        accuracy_score(y_test, y_pred_nb),
        precision_score(y_test, y_pred_nb, zero_division=0),
        recall_score(y_test, y_pred_nb, zero_division=0),
        f1_score(y_test, y_pred_nb)
    ],
    'Voting Ensemble': [
        accuracy_score(y_test, y_pred_voting_optimized),
        precision_score(y_test, y_pred_voting_optimized, zero_division=0),
        recall_score(y_test, y_pred_voting_optimized, zero_division=0),
        f1_score(y_test, y_pred_voting_optimized)
    ]
})
comparison_df.set_index('Metric', inplace=True)
logging.info("Step 1 완료: Metrics calculated and DataFrame prepared")

logging.info("Step 2: Generating Model Comparison Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(
    comparison_df,
    annot=True,
    cmap='coolwarm',
    fmt=".3f",
    linewidths=0.5,
    annot_kws={"size": 12},
    cbar_kws={'shrink': 0.7}
)
plt.title('Model Comparison with Optimized Ensemble', fontsize=16)
plt.ylabel('Evaluation Metric', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show(block=False)
logging.info("Step 2 완료: Model Comparison Heatmap generated and displayed\n")


# 11. ROC-AUC Visualization
logging.info("===ROC-AUC 시각화 시작===")
plt.figure(figsize=(12, 8))
models = {
    "Decision Tree": (best_dt_model, X_test_reduced),
    "Random Forest": (rf_model_recall, X_test_reduced),
    "Gradient Boosting": (best_gb_model, X_test_reduced),
    "SVM": (best_svm_model, X_test_reduced),
    "Naive Bayes": (nb_model, X_test_reduced_top_n),
    "Voting Ensemble": (voting_clf_optimized, X_test_reduced),
}

for name, (model, X_test_data) in models.items():
    try:
        logging.info(f"ROC-AUC 계산 중: {name}")
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_data)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test_data)
        else:
            logging.error(f"{name} does not support predict_proba or decision_function.")
            continue
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    except Exception as e:
        logging.error(f"{name}에서 에러 발생: {e}")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")
plt.title("ROC Curve Comparison of Models", fontsize=16)
plt.xlabel("False Positive Rate (FPR)", fontsize=14)
plt.ylabel("True Positive Rate (TPR)", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show(block=False)
logging.info("ROC-AUC Visualization 저장 완료")
