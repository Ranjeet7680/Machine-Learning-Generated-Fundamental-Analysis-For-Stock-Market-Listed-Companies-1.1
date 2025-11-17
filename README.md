# Machine Learning Generated Fundamental Analysis for Stock Market Listed Companies

This project automates **fundamental analysis** of stock market–listed companies (e.g., Nifty 100) using:

* Financial data fetched via **yfinance API**
* Feature engineering from financial statements (Income, Balance Sheet, Cash Flow)
* Machine Learning scoring (RandomForest)
* MySQL database storage via SQLAlchemy
* Excel report export
* CLI with colored output (rich)

---

## 🚀 Features

### ✔ Read Company Symbols from Excel

* Input file: `Nifty100Companies.xlsx`
* Must contain a column named **Symbol**.
* Automatically converts symbols like `RELIANCE` → `RELIANCE.NS`.

### ✔ Fetch Financial Statements

Using **yfinance**, the script downloads:

* Income Statement
* Balance Sheet
* Cash Flow Statement

Automatically extracts:

* Total Revenue
* Net Income
* Operating Cash Flow
* Total Assets
* Total Liabilities
* Shareholder Equity

### ✔ Financial Ratios Computed

* **ROE** → NetIncome / Equity
* **Debt to Equity** → TotalLiab / Equity
* **Profit Margin** → NetIncome / TotalRevenue
* **OCF / NetIncome** → OperatingCashFlow / NetIncome

### ✔ Machine Learning Model

* Simple regression using **RandomForestRegressor**
* Predicts `ROE` and maps predictions to a **0–100 score**
* Final score = 40% baseline + 60% ML model

### ✔ Database Integration

* Saves each company’s:

  * Symbol
  * Timestamp
  * ROE
  * Profit Margin
  * Debt-to-Equity
  * OCF-to-NetIncome
  * Final ML score

### ✔ CLI Support (rich)

* Shows score in colored format:

  * 🟢 Green: Score ≥ 70
  * 🟡 Yellow: 40 ≤ Score < 70
  * 🔴 Red: Score < 40

### ✔ Excel Export

* Full results exported to an output file like:

  * `results.xlsx`

---

## 📂 Project Structure

```
ml_fundamental_analysis.py  # Main pipeline script
README_ML_Fundamental_Analysis.md  # Documentation
Nifty100Companies.xlsx  # Input file (user-provided)
```

---

## 🛠 Installation

### 1. Install Dependencies

Create `requirements.txt` manually or install packages directly:

```
pip install pandas numpy yfinance rich scikit-learn sqlalchemy pymysql openpyxl
```

### 2. Prepare Excel Input

Your Excel file must contain:

```
| Company Name | Symbol |
| Reliance Industries | RELIANCE |
```

### 3. (Optional) Set Database URL

```
export DB_URL="mysql+pymysql://user:password@localhost:3306/yourdb"
```

---

## ▶️ Usage

Run the script:

```
python ml_fundamental_analysis.py --input Nifty100Companies.xlsx --out results.xlsx
```

With DB:

```
python ml_fundamental_analysis.py --input Nifty100Companies.xlsx --out results.xlsx --db $DB_URL
```

---

## 📊 Output

### Excel Columns include:

* Symbol
* CompanyName
* TotalRevenue
* NetIncome
* OperatingCashflow
* ROE
* ProfitMargin
* DebtToEquity
* OCF_to_NetIncome
* Score
* ModelScore
* FinalScore

---

## 🧪 Example Output (CLI)

```
Processing RELIANCE.NS
RELIANCE.NS:  82.34 (ROE: 0.21 ProfitMargin: 0.11)
HDFCBANK.NS:  74.92 (ROE: 0.18 ProfitMargin: 0.09)
...
Saved results to results.xlsx
```

---

## 🔧 Customization

You can extend the project by:

* Adding more ratios
* Training advanced ML models (XGBoost, LightGBM)
* Adding sector-wise benchmarking
* Creating a Streamlit dashboard
* Using paid APIs for high-accuracy financials

---

## 🧑‍💻 Author

Developed for automated ML-based stock market analysis.

Feel free to ask for:

* `requirements.txt`
* A packaged folder structure
* Streamlit web dashboard version
* MySQL table schema improvements
* A Dockerfile
