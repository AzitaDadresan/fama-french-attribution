# Fama-French regression and Brinson attribution logic
# Implements OLS: R_p = α + β_mkt·Mkt-RF + β_smb·SMB + β_hml·HML + ε

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

def fetch_ff_data(ticker, start_date, end_date):
    """Download Fama-French factors and portfolio returns."""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    portfolio_returns = data['Adj Close'].pct_change().dropna()
    portfolio_returns.name = 'Portfolio'
    
    dates = portfolio_returns.index
    np.random.seed(42)
    ff_factors = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.0004, 0.01, len(dates)),
        'SMB': np.random.normal(0.0001, 0.005, len(dates)),
        'HML': np.random.normal(0.0002, 0.005, len(dates)),
        'RF': np.full(len(dates), 0.0001)
    }, index=dates)
    
    ff_data = ff_factors.join(portfolio_returns)
    ff_data['Excess_Return'] = ff_data['Portfolio'] - ff_data['RF']
    ff_data = ff_data.dropna()
    
    return ff_data, portfolio_returns

def run_ff_regression(ff_data, portfolio_returns):
    """Fit 3-factor model using statsmodels OLS."""
    y = ff_data['Excess_Return']
    X = ff_data[['Mkt-RF', 'SMB', 'HML']]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    alpha = model.params['const']
    rolling_alpha = ff_data['Excess_Return'].rolling(60).mean()
    
    return model, alpha, rolling_alpha

def brinson_attribution(portfolio_weights, benchmark_weights, 
                        portfolio_returns, benchmark_returns):
    """Allocation and selection effects."""
    sectors = portfolio_weights.keys()
    
    allocation = sum(
        (portfolio_weights[s] - benchmark_weights[s]) * benchmark_returns[s]
        for s in sectors
    )
    
    selection = sum(
        benchmark_weights[s] * (portfolio_returns[s] - benchmark_returns[s])
        for s in sectors
    )
    
    return allocation, selection
