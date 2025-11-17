"""
Machine Learning Generated Fundamental Analysis For Stock Market Listed Companies

This single-file Python module implements:
- Read company tickers from an Excel file (Nifty100Companies.xlsx)
- Fetch financial statements (Income Statement, Balance Sheet, Cash Flow) using yfinance
- Clean & normalize the data
- Compute basic financial ratios / features
- A simple ML pipeline (RandomForest) to produce a "fundamental score"
- Save analysis results to MySQL using SQLAlchemy
- CLI with realtime colored logging using rich
- Export final reports to Excel

Usage:
1. Install requirements: pip install -r requirements.txt
2. Prepare Excel: Nifty100Companies.xlsx with a column named 'Symbol' (e.g. RELIANCE or RELIANCE.NS). If tickers are plain (RELIANCE), the script will append '.NS' for NSE.
3. Provide MySQL URL via environment variable: DB_URL (e.g. mysql+pymysql://user:pass@localhost:3306/dbname)
4. Run: python ml_fundamental_analysis.py --input Nifty100Companies.xlsx --out results.xlsx

Note: This script uses yfinance which provides commonly available financial tables; some fields may be missing for certain tickers. The code includes robust cleaning.

Reference: Project brief uploaded by user (project.txt)."""

import os
import argparse
import logging
import math
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, DateTime
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from rich.console import Console
from rich.table import Table as RichTable
from rich.progress import track

console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Utilities
# -------------------------

def nse_ticker(sym: str) -> str:
    """Ensure .NS appended for NSE tickers when not present."""
    if sym.endswith('.NS') or sym.endswith('.BO') or '.' in sym:
        return sym
    return sym + '.NS'


def read_company_list(path: str, symbol_col: str = 'Symbol') -> pd.DataFrame:
    df = pd.read_excel(path)
    if symbol_col not in df.columns:
        raise ValueError(f"Excel must contain a '{symbol_col}' column with tickers.")
    df[symbol_col] = df[symbol_col].astype(str).str.strip()
    return df


# -------------------------
# Data fetching
# -------------------------

def fetch_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    """Fetch financial tables via yfinance for a given ticker.
    Returns a dict with keys: income, balance, cashflow. Each is a DataFrame (columns = years).
    """
    try:
        tk = yf.Ticker(ticker)
        income = tk.financials if tk.financials is not None else pd.DataFrame()
        balance = tk.balance_sheet if tk.balance_sheet is not None else pd.DataFrame()
        cashflow = tk.cashflow if tk.cashflow is not None else pd.DataFrame()
        # transpose for row-based years
        return {
            'income': income.T if not income.empty else pd.DataFrame(),
            'balance': balance.T if not balance.empty else pd.DataFrame(),
            'cashflow': cashflow.T if not cashflow.empty else pd.DataFrame(),
        }
    except Exception as e:
        logger.exception(f"Error fetching {ticker}: {e}")
        return {'income': pd.DataFrame(), 'balance': pd.DataFrame(), 'cashflow': pd.DataFrame()}


# -------------------------
# Cleaning & Feature Engineering
# -------------------------

def safe_get(df: pd.DataFrame, col: str) -> float:
    try:
        return float(df.get(col, np.nan))
    except Exception:
        return np.nan


def extract_latest_metrics(fin_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Given the dict of dataframes, return a flat dict of latest-year metrics used for modeling/reporting.
    We attempt to get common fields; absent values become NaN.
    """
    income = fin_dict.get('income', pd.DataFrame())
    balance = fin_dict.get('balance', pd.DataFrame())
    cashflow = fin_dict.get('cashflow', pd.DataFrame())

    # pick the most recent row (index 0 after transpose) if available
    def latest(df):
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df.iloc[0]

    inc = latest(income)
    bal = latest(balance)
    cf = latest(cashflow)

    metrics = {
        'TotalRevenue': safe_get(inc, 'Total Revenue') or safe_get(inc, 'TotalRevenue') or safe_get(inc, 'Revenue'),
        'NetIncome': safe_get(inc, 'Net Income') or safe_get(inc, 'NetIncome') or safe_get(inc, 'NetIncomeApplicableToCommonShares'),
        'OperatingCashflow': safe_get(cf, 'Total Cash From Operating Activities') or safe_get(cf, 'Operating Cash Flow') or safe_get(cf, 'TotalCashFromOperatingActivities'),
        'TotalAssets': safe_get(bal, 'Total Assets') or safe_get(bal, 'TotalAssets'),
        'TotalLiab': safe_get(bal, 'Total Liab') or safe_get(bal, 'TotalLiab') or safe_get(bal, 'Total Liabilities'),
        'Equity': safe_get(bal, 'Total Stockholder Equity') or safe_get(bal, 'TotalStockholderEquity') or safe_get(bal, 'Shareholders Equity')
    }

    # Additional computed fields
    try:
        metrics['ROE'] = metrics['NetIncome'] / metrics['Equity'] if metrics['Equity'] and not math.isclose(metrics['Equity'], 0.0) else np.nan
    except Exception:
        metrics['ROE'] = np.nan

    try:
        metrics['DebtToEquity'] = metrics['TotalLiab'] / metrics['Equity'] if metrics['Equity'] and not math.isclose(metrics['Equity'], 0.0) else np.nan
    except Exception:
        metrics['DebtToEquity'] = np.nan

    try:
        metrics['ProfitMargin'] = metrics['NetIncome'] / metrics['TotalRevenue'] if metrics['TotalRevenue'] and not math.isclose(metrics['TotalRevenue'], 0.0) else np.nan
    except Exception:
        metrics['ProfitMargin'] = np.nan

    try:
        metrics['OCF_to_NetIncome'] = metrics['OperatingCashflow'] / metrics['NetIncome'] if metrics['NetIncome'] and not math.isclose(metrics['NetIncome'], 0.0) else np.nan
    except Exception:
        metrics['OCF_to_NetIncome'] = np.nan

    return metrics


# -------------------------
# ML module
# -------------------------

def train_simple_model(df_features: pd.DataFrame, target_col: str = 'ROE') -> Pipeline:
    """Train a simple regression model to predict target_col. Returns sklearn Pipeline.
    If not enough rows, returns None.
    """
    if df_features.shape[0] < 10 or target_col not in df_features.columns:
        logger.warning("Not enough data to train model; returning None")
        return None

    X = df_features.drop(columns=[target_col])
    y = df_features[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols]

    # Simple imputer + scaler + RandomForest
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_num, y)
    logger.info("Trained RandomForest model")
    return (pipeline, numeric_cols)


# -------------------------
# DB integration
# -------------------------

def init_db(db_url: str):
    engine = create_engine(db_url)
    metadata = MetaData()
    ml_table = Table('ml', metadata,
                     Column('id', Integer, primary_key=True, autoincrement=True),
                     Column('symbol', String(64)),
                     Column('run_at', DateTime),
                     Column('score', Float),
                     Column('roe', Float),
                     Column('profit_margin', Float),
                     Column('debt_to_equity', Float),
                     Column('ocf_to_netincome', Float)
                     )
    metadata.create_all(engine)
    return engine, ml_table


def save_result(engine, ml_table, row: Dict[str, Any]):
    conn = engine.connect()
    try:
        ins = ml_table.insert().values(**row)
        conn.execute(ins)
        conn.close()
    except SQLAlchemyError as e:
        logger.exception(f"DB error: {e}")


# -------------------------
# CLI & Orchestration
# -------------------------

def analyze_companies(input_xlsx: str, output_xlsx: str, db_url: str = None):
    df_companies = read_company_list(input_xlsx, symbol_col='Symbol')
    results = []

    features_rows = []

    # Fetch loop
    for idx, r in track(df_companies.iterrows(), total=len(df_companies), description="Fetching companies"):
        sym_raw = r['Symbol']
        sym = nse_ticker(sym_raw)
        console.print(f"[bold]Processing[/bold] {sym}")
        fin = fetch_financials(sym)
        metrics = extract_latest_metrics(fin)
        metrics['Symbol'] = sym
        metrics['CompanyName'] = r.get('Company Name', '') if 'Company Name' in r else ''
        features_rows.append(metrics)

    features_df = pd.DataFrame(features_rows)

    # Compute a simple baseline 'score' as normalized ROE (for demo). We also train a model when possible.
    # Fill na with median for scoring
    if 'ROE' in features_df.columns:
        median_roe = features_df['ROE'].median(skipna=True)
        features_df['ROE_filled'] = features_df['ROE'].fillna(median_roe)
        # baseline score: ROE scaled to 0-100, clamp
        def roe_to_score(x):
            try:
                val = float(x)
                return max(0.0, min(100.0, (val * 100)))  # ROE 0.1 -> 10
            except Exception:
                return 50.0
        features_df['Score'] = features_df['ROE_filled'].apply(roe_to_score)
    else:
        features_df['Score'] = 50.0

    # Train model if possible
    pipeline_info = None
    try:
        pipeline_info = train_simple_model(features_df[['ROE', 'ProfitMargin', 'DebtToEquity', 'OCF_to_NetIncome']].rename(columns={'DebtToEquity':'DebtToEquity','ProfitMargin':'ProfitMargin','OCF_to_NetIncome':'OCF_to_NetIncome'}), target_col='ROE')
    except Exception as e:
        logger.warning(f"Model training skipped: {e}")

    # If trained, predict and combine
    if pipeline_info:
        pipeline, numeric_cols = pipeline_info
        X_pred = features_df[numeric_cols]
        preds = pipeline.predict(X_pred.fillna(0))
        # map preds to 0-100
        preds_score = [max(0.0, min(100.0, p * 100)) for p in preds]
        features_df['ModelScore'] = preds_score
        # Blend baseline and model
        features_df['FinalScore'] = (features_df['Score'] * 0.4) + (features_df['ModelScore'] * 0.6)
    else:
        features_df['FinalScore'] = features_df['Score']

    # CLI output with color coding
    for _, row in features_df.iterrows():
        score = float(row['FinalScore']) if not pd.isna(row['FinalScore']) else 50.0
        if score >= 70:
            style = 'green'
        elif score >= 40:
            style = 'yellow'
        else:
            style = 'red'
        console.print(f"{row['Symbol']}: [bold {style}]{score:.2f}[/bold {style}] (ROE: {row.get('ROE'):.4f} ProfitMargin: {row.get('ProfitMargin')})")

    # Save to Excel
    features_df.to_excel(output_xlsx, index=False)
    console.print(f"Saved results to {output_xlsx}")

    # Save to DB if provided
    if db_url:
        engine, ml_table = init_db(db_url)
        for _, r in features_df.iterrows():
            row = {
                'symbol': r['Symbol'],
                'run_at': datetime.utcnow(),
                'score': float(r['FinalScore']) if not pd.isna(r['FinalScore']) else None,
                'roe': float(r['ROE']) if not pd.isna(r['ROE']) else None,
                'profit_margin': float(r['ProfitMargin']) if not pd.isna(r['ProfitMargin']) else None,
                'debt_to_equity': float(r['DebtToEquity']) if not pd.isna(r['DebtToEquity']) else None,
                'ocf_to_netincome': float(r['OCF_to_NetIncome']) if not pd.isna(r['OCF_to_NetIncome']) else None
            }
            save_result(engine, ml_table, row)
        console.print("Saved results to DB")


# -------------------------
# Main
# -------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Fundamental Analysis pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input Excel file with Symbol column')
    parser.add_argument('--out', '-o', default='results.xlsx', help='Output Excel file')
    parser.add_argument('--db', '-d', default=os.getenv('DB_URL'), help='SQLAlchemy DB URL or set DB_URL env var')
    args = parser.parse_args()

    analyze_companies(args.input, args.out, args.db)

# End of file
