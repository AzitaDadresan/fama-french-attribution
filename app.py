# Streamlit dashboard for Fama-French + Brinson attribution
# Author: Azita Dadresan | CQF Module 6 (Performance Attribution)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from attribution import fetch_ff_data, run_ff_regression, brinson_attribution

st.set_page_config(page_title="Fama-French Attribution", layout="wide")

st.title("📊 Fama-French 3-Factor Attribution Dashboard")
st.markdown("**CQF-grade factor decomposition for equity portfolios**")

st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Portfolio Ticker", value="SPY")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

if st.sidebar.button("Run Analysis"):
    with st.spinner("Fetching data and running regression..."):
        ff_data, portfolio_returns = fetch_ff_data(ticker, start_date, end_date)
        results, alpha, rolling_alpha = run_ff_regression(ff_data, portfolio_returns)
        
        st.subheader("Fama-French 3-Factor Regression")
        st.write(results.summary())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Annualized Alpha", f"{alpha * 252:.2%}")
            st.metric("Market Beta", f"{results.params['Mkt-RF']:.3f}")
        with col2:
            st.metric("SMB Loading", f"{results.params['SMB']:.3f}")
            st.metric("HML Loading", f"{results.params['HML']:.3f}")
        
        st.subheader("Rolling 60-Day Alpha")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_alpha.index, 
            y=rolling_alpha.values * 252,
            mode='lines',
            name='Alpha',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Annualized Alpha (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Brinson Attribution (vs Benchmark)")
        allocation, selection = brinson_attribution(
            portfolio_weights={'Tech': 0.4, 'Finance': 0.3, 'Energy': 0.3},
            benchmark_weights={'Tech': 0.35, 'Finance': 0.35, 'Energy': 0.3},
            portfolio_returns={'Tech': 0.15, 'Finance': 0.08, 'Energy': -0.05},
            benchmark_returns={'Tech': 0.12, 'Finance': 0.10, 'Energy': -0.03}
        )
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Allocation Effect", f"{allocation:.2%}")
        with col4:
            st.metric("Selection Effect", f"{selection:.2%}")
        
        st.success("Analysis complete!") 
