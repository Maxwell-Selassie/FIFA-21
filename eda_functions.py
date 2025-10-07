import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s : %(message)s')
# ----load data-----
def load_data(filepath: str):    
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print('File Not Found! Please check filepath and try again!')
        raise

# ----------dataset overview-----
def dataset_overview(df: pd.DataFrame):
    logging.info(f'Number of observations : {df.shape[0]}')
    logging.info(f'Number of features : {df.shape[1]}')
    overview = pd.DataFrame({
        "Dtype": df.dtypes,
        "Non-Null Count": df.count(),
        "Null Count": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    logging.info(overview.head(10))
    return df.describe(include='all')

# -------duplicate data---------
def duplicates(df: pd.DataFrame):
    duplicates = df[df.duplicated()]
    logging.info(f'Number of duplicated rows : {len(duplicates)}')
    if len(duplicates) == 0:
        logging.info(f'No duplicates found')
    return duplicates

# -----missing data---------
def missing_data(df: pd.DataFrame):
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values' : missing_values,
        'Missing Pct' : missing_pct.round(2)
    }).sort_values(by='Missing Pct',ascending=False)
    logging.info(f'---------Missing Data(Top 10)----------\n')
    logging.info(missing_data.head(10))
    return missing_data

# ---column summaries--------
def column_summaries(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for i,col in enumerate(numeric_cols,1):
        logging.info(f'{i:<2}. {col:<17} -Min : {df[col].min():<4} -Max : {df[col].max()}')

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for i,col in enumerate(categorical_cols,1):
        uniques = df[col].unique()
        logging.info(f'{i}. {col} | Unique : {df[col].nunique()} | Examples : {uniques[:5]}')
    return numeric_cols, categorical_cols

# outlier detection using IQR
def check_outliers(df: pd.DataFrame, col: str):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outlier, upper_bound, lower_bound

def outlier_summary(df: pd.DataFrame, numeric_cols):
    results = []
    logging.info(f"\n Outlier Summary (IQR Method):")
    for i, col in enumerate(numeric_cols, 1):
        outlier, lower, upper = check_outliers(df, col)
        results.append({
            'column' : col,
            'Outlier_count' : len(outlier),
            'lower_bound' : lower,
            'upper_bound' : upper
        })
    summary_df = pd.DataFrame(results)
    logging.info(summary_df)
    return summary_df
        
# ---- Save Reports ----
import os
def save_summary(df: pd.DataFrame, name: str):
    os.makedirs("eda_reports", exist_ok=True)
    path = f"eda_reports/{name}.csv"
    df.to_csv(path, index=False)
    logging.info(f"Saved report: {path}")

def run_basic_eda(filepath: str):
    df = load_data(filepath)

    # ---- Basic cleaning ----
    if 'Hits' in df.columns:
        df['Hits'] = df['Hits'].astype(str).str.extract(r'(\d+)').astype(float)
        logging.info("Column 'Hits' cleaned and converted to numeric.")

    overview = dataset_overview(df)
    duplicate = duplicates(df)
    missing = missing_data(df)
    numeric_cols, category_cols = column_summaries(df)
    outlier_df = outlier_summary(df, numeric_cols)

    save_summary(missing, "missing_data")
    save_summary(outlier_df, "outlier_summary")

    return {
        'data': df,
        'overview': overview,
        'duplicate': duplicate,
        'missing': missing,
        'outliers': outlier_df,
        'numeric_cols': numeric_cols,
        'category_cols': category_cols
    }

if __name__ == '__main__':
    df = run_basic_eda("../data/fifa21 raw data v2.csv")
# Temporary compatibility fix for NumPy >= 2.0
import builtins
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = builtins.DeprecationWarning

import sweetviz as sv

from pkg_resources import resource_filename
# create report
data = df['data']
report = sv.analyze(data)

# generate and show report
report.show_html("fifa21_sweetviz_report.html")

logging.info("Sweetviz report saved as eda_reports/fifa21_sweetviz_report.html")
