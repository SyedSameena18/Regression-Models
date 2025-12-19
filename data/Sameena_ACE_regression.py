# ==================================================
# Programme: Research Teaser for IoT’25
# Task Name: Intro to Machine Learning
# Author: Syed Sameena
# College: ACE Engineering College
# File: Sameena_ACE_recommender.py
# ==================================================

import os
import ast
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from time import time

# Try importing XGBoost; if not available, warn and skip XGB models
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print("⚠ XGBoost not available; XGB models will be skipped. Install with `pip install xgboost` to enable them.")
    XGBOOST_AVAILABLE = False

# ----------------------------
# Config / Paths / DB details
# ----------------------------
IOT_MAIN_CSV = "..\\iot_dataset.csv"           # adjust paths if needed
IOT_MAP_CSV = "..\\iot_dataset_mapping.csv"
PDF_OUTPUT = "Sameena_ACE_report.pdf"
PLOTS_DIR = "plots"

# PostgreSQL connection string - update password/host/db as needed
PG_CONN_STRING = "postgresql://postgres:Sameena@localhost:5432/postgres"

# MongoDB connection URI
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DBNAME = "iot25db"

# Create plots folder
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===============================
# Step 1: Load both datasets
# ===============================
print("Loading datasets...")
df_main = pd.read_csv(IOT_MAIN_CSV, sep=None, engine='python')  # engine='python' helps auto-detect separators
df_map = pd.read_csv(IOT_MAP_CSV, sep=None, engine='python')

print(f"Main dataset shape: {df_main.shape}")
print(f"Mapping dataset shape: {df_map.shape}")

# Show columns for debugging (optional)
print("Main columns:", df_main.columns.tolist())
print("Map columns:", df_map.columns.tolist())

# ===============================
# Step 2: Convert mapping file to lookup dict
# mapping file format:
# value_col | original_column
# original_column contains strings like "{'AQ': 'pm25', 'WF': 'flowrate', 'SL': 'active_power'}"
# ===============================
mapping_dict = {}
for _, row in df_map.iterrows():
    value_col = str(row['value_col']).strip()
    orig_col_str = row['original_column']
    if pd.isna(orig_col_str):
        continue
    # Parse the mapping string into a dict safely
    try:
        parsed = ast.literal_eval(orig_col_str)
    except Exception as e:
        # If parsing fails, try to fix common issues (e.g., single quotes mismatched)
        parsed = {}
        try:
            parsed = eval(orig_col_str)  # fallback (less safe) - only used if literal_eval failed
        except Exception:
            print(f"⚠ Could not parse mapping for {value_col}: {orig_col_str}")
            parsed = {}
    mapping_dict[value_col] = parsed

print("Parsed mapping keys (example):", list(mapping_dict.keys())[:6])

# ===============================
# Helper: rename columns for a given device type
# ===============================
def rename_columns_by_type(df, device_type):
    rename_map = {}
    for value_col, type_map in mapping_dict.items():
        # if mapping has the type and the mapping is non-empty string
        if device_type in type_map and type_map[device_type] and str(type_map[device_type]).strip():
            rename_map[value_col] = type_map[device_type].strip()
    # Rename only if those columns exist in df
    rename_map_existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=rename_map_existing)

# ===============================
# Step 3: Split per vertical & rename
# ===============================
if 'type' not in df_main.columns:
    raise KeyError("The main dataset does not contain a 'type' column. This script expects a 'type' column (e.g., 'AQ','WF','SL').")

verticals = df_main['type'].dropna().unique().tolist()
print("Detected verticals:", verticals)

vertical_data = {}
for v in verticals:
    df_v = df_main[df_main['type'] == v].copy()
    # Rename columns using mapping
    df_v = rename_columns_by_type(df_v, v)
    vertical_data[v] = df_v.reset_index(drop=True)
    print(f"Vertical {v}: shape {vertical_data[v].shape} - columns: {vertical_data[v].columns.tolist()[:12]}")

# ===============================
# Step 4: Insert into PostgreSQL & MongoDB per vertical
# ===============================
print("\nConnecting to databases...")
# PostgreSQL
pg_engine = create_engine(PG_CONN_STRING)

# MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DBNAME]

for v, df_v in vertical_data.items():
    table_name = f"{v}_table"
    collection_name = f"{v}_collection"
    df_to_store = df_v.copy()

    # Fill missing numeric/object columns (simple strategy)
    for col in df_to_store.columns:
        if pd.api.types.is_numeric_dtype(df_to_store[col]):
            df_to_store[col] = df_to_store[col].fillna(0)
        else:
            df_to_store[col] = df_to_store[col].fillna("Unknown")

    # MongoDB: wipe & insert
    try:
        mongo_db[collection_name].delete_many({})
        # convert numpy types to python native for Mongo insert
        records = df_to_store.replace({np.nan: None}).to_dict(orient='records')
        if records:
            mongo_db[collection_name].insert_many(records)
        print(f"Inserted {len(records)} documents into MongoDB collection '{collection_name}'")
    except Exception as e:
        print(f"⚠ MongoDB insert failed for {collection_name}: {e}")

    # PostgreSQL: write table
    try:
        # drop duplicates on node_id if present
        if 'node_id' in df_to_store.columns:
            df_to_store = df_to_store.drop_duplicates(subset=['node_id'])
        df_to_store.to_sql(table_name, pg_engine, if_exists='replace', index=False)
        # Add primary key if possible
        with pg_engine.begin() as conn:
            if 'node_id' in df_to_store.columns:
                try:
                    conn.execute(text(f'ALTER TABLE "{table_name}" ADD PRIMARY KEY ("node_id");'))
                except Exception as e:
                    print(f"⚠ Could not set node_id PK for {table_name} (maybe already exists): {e}")
            else:
                try:
                    conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN id SERIAL PRIMARY KEY;'))
                except Exception as e:
                    print(f"⚠ Could not add serial PK for {table_name}: {e}")
        print(f"Inserted table '{table_name}' into PostgreSQL.")
    except Exception as e:
        print(f"⚠ PostgreSQL insertion failed for {table_name}: {e}")

# ===============================
# Step 5: Build ML models per vertical, save plots, and create PDF
# ===============================
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "IoT’25 Machine Learning Report", ln=True, align='C')
pdf.ln(4)
pdf.set_font("Arial", '', 11)
pdf.multi_cell(0, 8, f"Datasets used: {IOT_MAIN_CSV} and {IOT_MAP_CSV}. Plots stored in ./{PLOTS_DIR}/")

# Model loop
for v, df_v in vertical_data.items():
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, f"Vertical: {v}", ln=True)
    pdf.ln(3)

    # Identify columns
    exclude_cols = ['node_id', 'type', 'name', 'created_at']
    # Choose target intelligently:
    numeric_cols = df_v.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df_v.select_dtypes(include=['object']).columns.tolist()

    # We need at least one feature and one target; choose target as first meaningful numeric column not in exclude
    target = None
    task_type = None
    # Prefer domain-specific targets like 'pm25', 'active_power', etc.
    preferred_targets = ['pm25','active_power','flowrate','total_flow','temperature','noise','pm10']
    for t in preferred_targets:
        if t in df_v.columns and t not in exclude_cols:
            target = t
            break

    if target is None:
        # fallback: pick first numeric column not in exclude
        for col in numeric_cols:
            if col not in exclude_cols:
                target = col
                break

    if target is None:
        # fallback: classification on first object column (if any)
        if object_cols:
            target = object_cols[0]
            task_type = "classification"

    if target is None:
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, f"⚠ No suitable target found for vertical {v}. Skipping ML for this vertical.")
        continue

    # Determine task type
    if task_type is None:
        if pd.api.types.is_numeric_dtype(df_v[target]):
            task_type = "regression"
        else:
            task_type = "classification"

    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, f"Target chosen: {target} | Task type: {task_type}")

    # Prepare features X and target y
    X = df_v.drop(columns=[target] + [c for c in exclude_cols if c in df_v.columns], errors='ignore')
    # Drop non-informative columns if present (latitude/longitude could be kept)
    # Remove columns that are identical to target
    if target in X.columns:
        X = X.drop(columns=[target], errors='ignore')

    y = df_v[target].copy()

    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        try:
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        except Exception as e:
            print(f"⚠ Label encoding failed for column {col}: {e}")

    # Encode y if classification and object dtype
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        y_le = LabelEncoder()
        y = y_le.fit_transform(y.astype(str))
    else:
        y = y.fillna(y.mean() if pd.api.types.is_numeric_dtype(y) else 0)

    X = X.fillna(0)

    # Split dataset
    if len(X) < 2 or len(y) < 2:
        pdf.multi_cell(0, 8, f"⚠ Not enough data after preprocessing for vertical {v}. Rows: {len(df_v)}. Skipping.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric features for regression
    scaler = None
    if task_type == "regression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Define models
    models = {}
    param_grids = {}

    if task_type == "regression":
        models["LinearRegression"] = LinearRegression()
        models["RandomForestRegressor"] = RandomForestRegressor(n_estimators=100, random_state=42)
        param_grids["RandomForestRegressor"] = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        if XGBOOST_AVAILABLE:
            models["XGBoostRegressor"] = XGBRegressor(eval_metric='rmse', random_state=42)
            param_grids["XGBoostRegressor"] = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    else:
        models["LogisticRegression"] = LogisticRegression(max_iter=1000)
        models["RandomForestClassifier"] = RandomForestClassifier(n_estimators=100, random_state=42)
        param_grids["RandomForestClassifier"] = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        if XGBOOST_AVAILABLE:
            models["XGBoostClassifier"] = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
            param_grids["XGBoostClassifier"] = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}

    results_summary = []

    for model_name, model in models.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 7, f"Model: {model_name}", ln=True)
        start_time = time()
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 7, f"⚠ Training failed for {model_name}: {e}")
            continue
        baseline_time = time() - start_time

        # Predictions
        y_pred = model.predict(X_test)

        if task_type == "regression":
            baseline_metric = mean_squared_error(y_test, y_pred)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 7, f"Baseline MSE: {baseline_metric:.4f} | Training time: {baseline_time:.2f}s")
        else:
            baseline_metric = accuracy_score(y_test, y_pred)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 7, f"Baseline Accuracy: {baseline_metric:.4f} | Training time: {baseline_time:.2f}s")

        best_params = "N/A"
        tuned_metric = "N/A"
        tuned_time_str = "N/A"

        # Hyperparameter tuning if grid available
        if model_name in param_grids:
            grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring='r2' if task_type == "regression" else 'accuracy', n_jobs=-1)
            t0 = time()
            try:
                grid.fit(X_train, y_train)
                tuned_time = time() - t0
                best_model = grid.best_estimator_
                y_pred_tuned = best_model.predict(X_test)
                if task_type == "regression":
                    tuned_metric = mean_squared_error(y_test, y_pred_tuned)
                else:
                    tuned_metric = accuracy_score(y_test, y_pred_tuned)
                best_params = grid.best_params_
                tuned_time_str = f"{tuned_time:.2f}s"
                pdf.multi_cell(0, 7, f"Tuned metric: {tuned_metric} | Best params: {best_params} | Tuning time: {tuned_time:.2f}s")
            except Exception as e:
                pdf.multi_cell(0, 7, f"⚠ GridSearchCV failed for {model_name}: {e}")
        else:
            pdf.multi_cell(0, 7, "Hyperparameter tuning: N/A for this model")

        # Save results summary
        results_summary.append({
            "Model": model_name,
            "BaselineMetric": baseline_metric,
            "TunedMetric": tuned_metric,
            "BestParams": best_params,
            "BaselineTime": baseline_time,
            "TunedTime": tuned_time_str
        })

        # ----------------------------
        # Create plot for this model
        # ----------------------------
        plot_filename = os.path.join(PLOTS_DIR, f"{v}_{model_name}.png")
        plt.figure(figsize=(6,4))
        if task_type == "regression":
            # Predicted vs Actual scatter
            try:
                plt.scatter(y_test, y_pred)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"{v} - {model_name}: Actual vs Predicted")
                plt.tight_layout()
                plt.savefig(plot_filename)
                plt.close()
            except Exception as e:
                plt.close()
                print(f"⚠ Could not create regression plot for {v} {model_name}: {e}")
                plot_filename = None
        else:
            # Classification confusion matrix
            try:
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(cm)
                disp.plot()
                plt.title(f"{v} - {model_name}: Confusion Matrix")
                plt.tight_layout()
                plt.savefig(plot_filename)
                plt.close()
            except Exception as e:
                plt.close()
                print(f"⚠ Could not create confusion matrix for {v} {model_name}: {e}")
                plot_filename = None

        # Embed plot into PDF (if created)
        if os.path.exists(plot_filename):
            try:
                # Add a small spacer
                pdf.ln(2)
                # Ensure image will fit: set width to pdf.w - 30 margin
                img_w = pdf.w - 30
                pdf.image(plot_filename, w=img_w)
                pdf.ln(4)
            except Exception as e:
                pdf.multi_cell(0, 7, f"⚠ Could not add plot {plot_filename} to PDF: {e}")
        else:
            pdf.multi_cell(0, 7, f"⚠ Plot not available for {model_name}")

    # After all models for this vertical, add a summary table in PDF
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Summary (Baseline vs Tuned)", ln=True)
    pdf.set_font("Arial", '', 11)
    for res in results_summary:
        pdf.multi_cell(0, 7, f"{res['Model']} | Baseline: {res['BaselineMetric']} | Tuned: {res['TunedMetric']} | BestParams: {res['BestParams']}")

# ===============================
# Step 6: Save Final PDF (UTF-8 Safe)
# ===============================
from fpdf import FPDF

# Optional helper to clean up problematic characters
def clean_text(text):
    """Replace unsupported symbols with safe ones for PDF."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "•": "-", "…": "...",
        "™": "(TM)", "®": "(R)", "©": "(C)"
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

# Clean every text cell (optional if already done earlier)
# Example: pdf.multi_cell(0, 10, clean_text("Some text"))

try:
    pdf.output(PDF_OUTPUT)
    print(f"\n✅ PDF Report generated successfully: {PDF_OUTPUT}")
except Exception as e:
    safe_output = PDF_OUTPUT.replace(".pdf", "_utf8safe.pdf")
    print(f"⚠ Error saving PDF ({e}). Retrying safely...")

    # Save again after cleaning non-UTF characters
    with open(safe_output, "wb") as f:
        data = pdf.output(dest="S").encode("latin-1", "replace")
        f.write(data)

    print(f"✅ PDF saved successfully as: {safe_output}")

print(f"✅ Plots saved in folder: {PLOTS_DIR}")
