# equipment_maintenance_synthetic_data.py
"""
Full analysis and model training script for equipment maintenance.
This script performs:
1.  Data loading and Exploratory Data Analysis (EDA), saving plots.
2.  Data splitting (1000 rows hold-out).
3.  Noise injection to create a robust training set.
4.  Model training (Logistic, Tree, RF, XGB) on noisy data.
5.  Model evaluation, saving reports, plots, and .pkl models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, jarque_bera
import os
import joblib
import warnings

# --- ML Imports ---
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    log_loss
)
from sklearn.datasets import make_classification

warnings.filterwarnings('ignore')
plt.ioff()  # Turn off interactive plotting

print("Starting full analysis and artifact creation...")

# --- Setup Artifacts Directory (Using your specific path) ---
# Use 'r' prefix for raw string to handle Windows backslashes
ARTIFACTS_DIR = r"D:\projects\equipment_failure_prediction\artifacts"
DATA_PATH = r'D:\projects\equipment_failure_prediction\dataset\automotive_machine_M0001_2025_5min_drift_failure_spike.csv'

# Check if directory exists, if not, create it
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)
    print(f"Created directory: '{ARTIFACTS_DIR}'")
else:
    print(f"Artifacts will be saved to existing directory: '{ARTIFACTS_DIR}'")


# =============================================================================
# --- EDA Functions (Modified to save plots) ---
# =============================================================================

def numerical_analysis(dataframe, column_name, cat_col=None, bins="auto", output_dir="."):
    """ Plots a comprehensive view of a numerical column and saves it. """
    fig = plt.figure(figsize=(15, 10))
    grid = GridSpec(nrows=2, ncols=2, figure=fig)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    sns.kdeplot(data=dataframe, x=column_name, hue=cat_col, ax=ax1)
    ax1.set_title(f'KDE Plot of {column_name}')
    sns.boxplot(data=dataframe, x=column_name, hue=cat_col, ax=ax2)
    ax2.set_title(f'Boxplot of {column_name}')
    sns.histplot(data=dataframe, x=column_name, bins=bins, hue=cat_col, kde=True, ax=ax3)
    ax3.set_title(f'Histogram of {column_name}')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"numerical_analysis_{column_name}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved numerical analysis plot to '{save_path}'")

def numerical_categorical_analysis(dataframe, cat_column_1, num_column, output_dir="."):
    """ Plots relationships between cat and num variables and saves it. """
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 7.5))
    
    sns.barplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax1[0])
    ax1[0].set_title(f'Average {num_column} by {cat_column_1}')
    sns.boxplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax1[1])
    ax1[1].set_title(f'Boxplot of {num_column} by {cat_column_1}')
    sns.violinplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax2[0])
    ax2[0].set_title(f'Violinplot of {num_column} by {cat_column_1}')
    sns.stripplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax2[1])
    ax2[1].set_title(f'Stripplot of {num_column} by {cat_column_1}')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"num_cat_analysis_{cat_column_1}_vs_{num_column}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved num/cat analysis plot to '{save_path}'")

def categorical_analysis(dataframe, column_name, output_dir="."):
    """ Prints value counts and saves a countplot. """
    analysis_df = pd.DataFrame({
        "Count": dataframe[column_name].value_counts(),
        "Percentage": (
            dataframe[column_name]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .astype("str")
            .add("%")
        )
    })
    print(f"\n--- Analysis of '{column_name}' ---")
    print(analysis_df)
    print("*" * 50)
    
    unique_categories = dataframe[column_name].unique().tolist()
    number_of_categories = dataframe[column_name].nunique()
    print(f"The unique categories in {column_name} column are:\n{unique_categories}")
    print(f"The number of categories in {column_name} column are: {number_of_categories}")

    # Plot countplot
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(data=dataframe, x=column_name)
    plt.title(f'Countplot for {column_name}')
    plt.xticks(rotation=45)
    save_path = os.path.join(output_dir, f"categorical_analysis_{column_name}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved categorical analysis plot to '{save_path}'")

# =============================================================================
# --- 1. Load Data & Initial EDA ---
# =============================================================================

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded data from '{DATA_PATH}'")
except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_PATH}'")
    print("Please ensure the path is correct.")
    exit()

print(df.head())

rows, columns = df.shape
print(f'Dataset has {rows} rows and {columns} columns')

df = df.rename(str.lower, axis=1)
print("\nRenamed columns to lowercase:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

df_metadata = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes.values,
    'missing_values': df.isnull().sum().values,
    'unique_values': df.nunique().values
})
print("\nDataset Metadata:")
print(df_metadata)

print("\nEquipment Failure Value Counts:")
print(df.equipment_failure.value_counts())

numeric_cols = df.select_dtypes(include='number').columns
df_numeric = df[numeric_cols]
print("\nNumeric Columns Description:")
print(df_numeric.describe())

# --- Scatter Plots ---
print("\nGenerating scatter plots...")
scatter_features = ['sensor_5', 'sensor_8', 'sensor_15', 'sensor_16', 
                    'sensor_17', 'sensor_18', 'sensor_19', 'process_errors']

for feature in scatter_features:
    fig = plt.figure()
    df.plot.scatter(x=feature, y='equipment_failure', title=f'Scatter Plot: {feature} vs Failure')
    save_path = os.path.join(ARTIFACTS_DIR, f"scatter_{feature}_vs_failure.png")
    plt.savefig(save_path)
    plt.close(fig)
print(f"Saved scatter plots to '{ARTIFACTS_DIR}'")

# --- Column-wise Boxplots ---
print("\nGenerating column-wise boxplots...")
for col in ['age_of_equipment', 'process_errors', 'sensor_8']:
    fig = plt.figure()
    sns.boxplot(df[col])
    plt.title(f"Boxplot for {col}")
    save_path = os.path.join(ARTIFACTS_DIR, f"boxplot_{col}.png")
    plt.savefig(save_path)
    plt.close(fig)
print(f"Saved boxplots to '{ARTIFACTS_DIR}'")

# --- Correlation Matrix ---
print("\nGenerating correlation matrix...")
corr = df.select_dtypes(include='number').corr(method='spearman')
fig = plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix (Numerical Features)')
save_path = os.path.join(ARTIFACTS_DIR, "correlation_heatmap.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close(fig)
print(f"Saved correlation heatmap to '{save_path}'")

# --- Pairplot (This may take a moment) ---
print("\nGenerating pairplot... (This may take a while)")
try:
    pairplot_fig = sns.pairplot(df.select_dtypes(include='number'))
    save_path = os.path.join(ARTIFACTS_DIR, "pairplot.png")
    pairplot_fig.savefig(save_path)
    plt.close('all') # Close all figures
    print(f"Saved pairplot to '{save_path}'")
except Exception as e:
    print(f"Could not generate pairplot: {e}")
    plt.close('all')

# --- Missing and Duplicate Rows ---
missing_rows = df.isnull().any(axis=1).sum()
print(f'\nThere are {missing_rows} rows with missing values in the data.')
print(f"It accounts for {(missing_rows/df.shape[0])*100:.2f}% of the data")

df_final = df
print(f"\nTotal duplicate rows: {df_final.duplicated().sum()}")

# --- Run Analysis Functions ---
categorical_analysis(df_final, 'equipment_failure', output_dir=ARTIFACTS_DIR)

# --- PCA on All Data ---
print("\nGenerating PCA plot...")
features = ['process_errors', 'sensor_5', 'sensor_8',
            'sensor_15', 'sensor_16', 'sensor_17',
            'sensor_18', 'sensor_19', 'age_of_equipment']
X_pca_raw = df[features]
y_pca_raw = df['equipment_failure'].values

X_std = StandardScaler().fit_transform(X_pca_raw)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['equipment_failure'] = y_pca_raw

fig = plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='equipment_failure',
                data=pca_df, palette='coolwarm', alpha=0.7)
plt.title('PCA – Sensor & Process Data (Full Dataset)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
plt.grid(True)
save_path = os.path.join(ARTIFACTS_DIR, "pca_full_dataset.png")
plt.savefig(save_path)
plt.close(fig)
print(f"Saved PCA plot to '{save_path}'")
print('Explained variance by PC1 and PC2:', pca.explained_variance_ratio_)


# =============================================================================
# --- 2. Data Splitting (Clean) ---
# =============================================================================

print("\n--- Splitting Data (1000 row hold-out) ---")
split_index = 1000
# Dataset 1: The first 1000 rows for future testing
df_future_testing = df.iloc[:split_index].reset_index(drop=True).copy()
# Dataset 2: The remaining rows for training
df_reduced = df.iloc[split_index:].reset_index(drop=True).copy()

print("Original shape:", df.shape)
print("Removed (Future Test) shape:", df_future_testing.shape)
print("Reduced (for training) shape:", df_reduced.shape)

# --- Save the future testing data (clean) ---
future_test_path = os.path.join(ARTIFACTS_DIR, "testing_dataset.csv")
df_future_testing.to_csv(future_test_path, index=False)
print(f"Saved clean testing data to '{future_test_path}'")

# --- Save the clean training data ---
clean_train_path = os.path.join(ARTIFACTS_DIR, "training_data_clean.csv")
df_reduced.to_csv(clean_train_path, index=False)
print(f"Saved clean training data to '{clean_train_path}'")

# =============================================================================
# --- 3. Model Training (on CLEAN data) ---
# =============================================================================

print("\n--- Training Models on CLEAN Data for Baseline ---")

X_clean = df_reduced[features]
y_clean = df_reduced['equipment_failure']
le = LabelEncoder()
y_enc_clean = le.fit_transform(y_clean)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clean, y_enc_clean, test_size=0.2, random_state=42, stratify=y_enc_clean
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

# --- Train and Evaluate (CLEAN) ---
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred_c = model.predict(X_test_c)

    print(f"\n🔹 {name} (CLEAN DATA)")
    print("Accuracy:", round(accuracy_score(y_test_c, y_pred_c), 3))
    print(classification_report(y_test_c, y_pred_c, target_names=le.classes_))

    disp = ConfusionMatrixDisplay(confusion_matrix(y_test_c, y_pred_c), display_labels=le.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{name} - Confusion Matrix (CLEAN DATA)")
    save_path = os.path.join(ARTIFACTS_DIR, f"clean_cm_{name.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')
    print(f"Saved clean CM to '{save_path}'")

# --- AUC-ROC (CLEAN) ---
print("\nGenerating AUC-ROC curve for CLEAN data...")
fig = plt.figure(figsize=(7, 6))
for name, model in models.items():
    y_prob_c = model.predict_proba(X_test_c)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_c, y_prob_c)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('AUC-ROC Curve (CLEAN Data - Train-Test Split)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
save_path = os.path.join(ARTIFACTS_DIR, "clean_auc_roc_train_test_split.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close(fig)
print(f"Saved clean AUC-ROC plot to '{save_path}'")

# --- K-fold CV AUC-ROC (CLEAN) ---
print("\nGenerating K-Fold AUC-ROC curve for CLEAN data...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fig = plt.figure(figsize=(7, 6))
for name, model in models.items():
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_idx, test_idx in cv.split(X_clean, y_enc_clean):
        X_train_cv, X_test_cv = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train_cv, y_test_cv = y_enc_clean[train_idx], y_enc_clean[test_idx]
        model.fit(X_train_cv, y_train_cv)
        y_prob_cv = model.predict_proba(X_test_cv)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_cv, y_prob_cv)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=f'{name} (Mean AUC = {mean_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('AUC-ROC Curve (CLEAN Data - 5-Fold Cross-Validation)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
save_path = os.path.join(ARTIFACTS_DIR, "clean_auc_roc_kfold.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close(fig)
print(f"Saved clean K-Fold AUC-ROC plot to '{save_path}'")

# --- Learning Curves (CLEAN) ---
print("\nGenerating Learning Curves for CLEAN data...")
fig = plt.figure(figsize=(10, 7))
for name, model in {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_clean, y_enc_clean, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_error = 1 - np.mean(train_scores, axis=1)
    test_error = 1 - np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_error, 'o--', label=f'{name} - Train Error')
    plt.plot(train_sizes, test_error, 'o-', label=f'{name} - Validation Error')
plt.title("Learning Curves (CLEAN Data)")
plt.xlabel("Training Set Size"); plt.ylabel("Error (1 - Accuracy)")
plt.legend(); plt.grid(True)
save_path = os.path.join(ARTIFACTS_DIR, "clean_learning_curves.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close(fig)
print(f"Saved clean Learning Curves plot to '{save_path}'")

# =============================================================================
# --- 4. Noise Injection ---
# =============================================================================

print("\n--- Applying Noise to Training Data ---")
df_train = df_reduced.copy()
# Define target variable string
target = 'equipment_failure'
X = df_train[features].copy()
y = df_train[target].copy()

# (Re-fit encoder just to be safe, though it's the same data)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_noisy = pd.Series(y_encoded, index=y.index, name=target)
top_sensors = ['sensor_5', 'sensor_18', 'sensor_17']
X_noisy = X.copy()

# Strong noise for top sensors
for col in top_sensors:
    feature_range = X_noisy[col].max() - X_noisy[col].min()
    if feature_range == 0: feature_range = 1 # Avoid div by zero
    X_noisy[col] += np.random.normal(0, 0.5 * feature_range, X_noisy[col].shape)
# Smaller noise for remaining sensors
other_sensors = [col for col in X_noisy.columns if col not in top_sensors]
for col in other_sensors:
    feature_range = X_noisy[col].max() - X_noisy[col].min()
    if feature_range == 0: feature_range = 1
    X_noisy[col] += np.random.normal(0, 0.2 * feature_range, X_noisy[col].shape)
# Add small label noise
label_noise_fraction = 0.04
n_noisy = int(label_noise_fraction * len(y_noisy))
noisy_indices = np.random.choice(y_noisy.index, n_noisy, replace=False)
y_noisy.loc[noisy_indices] = 1 - y_noisy.loc[noisy_indices] # Flip labels

df_noisy_train = X_noisy.copy()
df_noisy_train[target] = y_noisy

noisy_train_path = os.path.join(ARTIFACTS_DIR, "training_data_noisy.csv")
df_noisy_train.to_csv(noisy_train_path, index=False)
print(f"Noisy training data ready and saved to '{noisy_train_path}'")
print(df_noisy_train.head())

# =============================================================================
# Check datatypes of all columns in the noisy dataset
# =============================================================================
print(df_noisy_train.dtypes)


# =============================================================================
# --- 5. Final Model Training (on NOISY data) ---
# =============================================================================

print("\n--- Training Final Models on NOISY Data ---")

# --- Train-test split for evaluation ---
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    df_noisy_train[features],
    df_noisy_train[target], # Use the 'target' string here
    test_size=0.2,
    random_state=42,
    stratify=df_noisy_train[target] # And here
)

# --- Define Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

# --- Train, Evaluate, and Save Models (NOISY) ---
for name, model in models.items():
    print(f"Training {name} on noisy data...")
    # Train on the full noisy dataset
    model.fit(df_noisy_train[features], df_noisy_train[target]) # Use 'target' string
    
    # Evaluate on the noisy test split
    y_pred_n = model.predict(X_test_n)
    
    print(f"\n▼ {name} Evaluation Report (NOISY test split):")
    print("Accuracy:", round(accuracy_score(y_test_n, y_pred_n), 3))
    print(classification_report(y_test_n, y_pred_n, target_names=le.classes_))

    # --- Save Confusion Matrix ---
    cm = confusion_matrix(y_test_n, y_pred_n)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{name} - Confusion Matrix (NOISY DATA)")
    cm_path = os.path.join(ARTIFACTS_DIR, f"noisy_cm_{name.replace(' ', '_').lower()}.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=120)
    print(f"Saved noisy confusion matrix to '{cm_path}'")
    plt.close('all')

    # --- Save Model ---
    filename = f"{name.replace(' ', '_').lower()}_model.pkl"
    model_path = os.path.join(ARTIFACTS_DIR, filename)
    joblib.dump(model, model_path)
    print(f"Saved final {name} as '{model_path}'")

# --- Save the Label Encoder ---
le_path = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')
joblib.dump(le, le_path)
print(f"Saved final Label Encoder as '{le_path}'")

# --- Generate and Save AUC-ROC Curve (NOISY 5-Fold CV) ---
print("\nGenerating final AUC-ROC Curve (Noisy Data, 5-Fold CV)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fig = plt.figure(figsize=(7, 6))

for name, model in models.items():
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    # Re-init model for CV
    if name == "Logistic Regression":
        model_cv = LogisticRegression(max_iter=1000, random_state=42)
    elif name == "Decision Tree":
        model_cv = DecisionTreeClassifier(random_state=42)
    elif name == "Random Forest":
        model_cv = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model_cv = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

    for tr, ts in cv.split(X_noisy, y_noisy):
        model_cv.fit(X_noisy.iloc[tr], y_noisy.iloc[tr])
        y_prob = model_cv.predict_proba(X_noisy.iloc[ts])[:, 1]
        fpr, tpr, _ = roc_curve(y_noisy.iloc[ts], y_prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    plt.plot(mean_fpr, np.mean(tprs, axis=0), label=f'{name} (AUC={auc(mean_fpr, np.mean(tprs,axis=0)):.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('AUC-ROC Curve (Noisy Data - 5-Fold CV)')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend()
auc_path = os.path.join(ARTIFACTS_DIR, "noisy_auc_roc_curve_kfold.png")
plt.savefig(auc_path, bbox_inches="tight", dpi=120)
plt.close(fig)
print(f"✅ Final AUC-ROC curve saved to '{auc_path}'")

# =============================================================================
# --- 6. Extra Plots from Notebook ---
# =============================================================================

# --- Sample Learning Curve (from notebook) ---
print("\nGenerating sample learning curve plot...")
X_sample, y_sample = make_classification(n_samples=2000, n_features=10, n_informative=6, n_redundant=2, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)
model_s = LogisticRegression(max_iter=1000)
train_sizes = np.linspace(0.1, 1.0, 10)
train_errors, test_errors = [], []

for size in train_sizes:
    X_subset = X_train_s[:int(size * len(X_train_s))]
    y_subset = y_train_s[:int(size * len(y_train_s))]
    if len(y_subset) == 0: continue
    model_s.fit(X_subset, y_subset)
    y_train_pred_s = model_s.predict(X_subset)
    y_test_pred_s = model_s.predict(X_test_s)
    train_errors.append(1 - accuracy_score(y_subset, y_train_pred_s))
    test_errors.append(1 - accuracy_score(y_test_s, y_test_pred_s))

fig = plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_errors, marker='o', label='Train Error')
plt.plot(train_sizes, test_errors, marker='s', label='Test Error')
plt.xlabel('Training Set Size (%)'); plt.ylabel('Error (1 - Accuracy)')
plt.title('Sample Learning Curve: Logistic Regression'); plt.legend(); plt.grid(True)
save_path = os.path.join(ARTIFACTS_DIR, "sample_learning_curve.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close(fig)
print(f"Saved sample learning curve to '{save_path}'")

# --- Loss Curve (SGD) ---
print("\nGenerating SGD loss curve...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_n)
X_test_scaled = scaler.transform(X_test_n)

model_lr_sgd = SGDClassifier(loss='log_loss', random_state=42)
all_classes = np.unique(y_train_n)
train_losses, test_losses = [], []
n_epochs = 45

for epoch in range(n_epochs):
    model_lr_sgd.partial_fit(X_train_scaled, y_train_n, classes=all_classes)
    y_pred_train_proba = model_lr_sgd.predict_proba(X_train_scaled)
    y_pred_test_proba = model_lr_sgd.predict_proba(X_test_scaled)
    train_loss = log_loss(y_train_n, y_pred_train_proba)
    test_loss = log_loss(y_test_n, y_pred_test_proba)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

print(f"Final Test Loss (SGD): {test_losses[-1]:.4f}")

fig = plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test (Validation) Loss")
plt.title("Logistic Regression Loss Curve (via SGD)", fontsize=16)
plt.xlabel("Epoch"); plt.ylabel("Log Loss (Lower is better)")
plt.legend(); plt.grid(True)
save_path = os.path.join(ARTIFACTS_DIR, "sgd_loss_curve.png")
plt.savefig(save_path, bbox_inches="tight")
plt.close(fig)
print(f"✅ SGD Loss curve saved to '{save_path}'")

print("\n--- Full analysis and artifact creation complete. ---")