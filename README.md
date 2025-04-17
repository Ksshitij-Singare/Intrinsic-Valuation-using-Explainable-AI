# Intrinsic Valuation with Explainable AI (XAI)

This repository contains a Jupyter Notebook (`Intrinsic Valuation Ex AI.ipynb`) implementing an advanced Discounted Cash Flow (DCF) valuation model integrated with Explainable AI (XAI) techniques for S&P 500 companies. The project leverages financial data to estimate intrinsic values, assess undervaluation, and provide transparency using machine learning models and SHAP analysis.

## Project Overview

- **Objective**: Develop a robust framework for intrinsic valuation using DCF, enhanced with XAI to ensure transparency and interpretability of financial predictions.
- **Target**: S&P 500 companies, with a focus on metrics like Free Cash Flow (FCF), Weighted Average Cost of Capital (WACC), and undervaluation percentages.
- **Tools**: Python, yfinance, pandas, numpy, matplotlib, scikit-learn, lightgbm, catboost, xgboost, and shap.

## Features

- Fetches financial fundamentals (e.g., P/E ratio, ROE, EBITDA) from Yahoo Finance for all S&P 500 companies.
- Calculates historical FCF growth and estimates WACC with realistic bounds.
- Implements DCF to compute intrinsic value per share for each ticker.
- Trains multiple regression models (Random Forest, Gradient Boosting, XGBoost) to predict undervaluation percentages.
- Uses SHAP (TreeSHAP) for XAI to explain model predictions and identify key features.

## Usage

1. Open `Intrinsic Valuation Ex AI.ipynb` in Jupyter Notebook or Google Colab.
2. Run the cells sequentially to:
- Install dependencies.
- Fetch S&P 500 ticker data and financial fundamentals.
- Compute DCF valuations and train models.
- Generate SHAP explanations (note: may require sampling for performance).
3. The notebook outputs a `fundamentals_dataset.csv` file with collected data and prints model evaluation metrics (MAE, RMSE, R2).

## Dataset

- **Source**: Yahoo Finance (via yfinance) and Wikipedia (S&P 500 list).
- **Content**: Financial metrics (e.g., PE_ratio, PB_ratio, OperatingCF) and undervaluation predictions.
- **File**: `fundamentals_dataset.csv` is generated after running the data collection cells.

## Results

- Model performance is evaluated with MAE, RMSE, and R2 scores for Random Forest, Gradient Boosting, and XGBoost.
- SHAP summaries provide insights into which financial features most influence undervaluation predictions.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for enhancements or bug fixes. Ensure to update the README with details of changes.

## License

This project is under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by discussions on integrating DCF with XAI for transparency in financial modeling.
- Utilizes open-source libraries and data from Yahoo Finance and Wikipedia.
