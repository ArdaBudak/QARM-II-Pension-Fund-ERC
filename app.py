import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cvxpy as cp
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
from fpdf import FPDF
import io
from sklearn.covariance import LedoitWolf

# Custom styling
st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")
try:
    st.logo("ERC Portfolio.png")
except:
    pass 

st.markdown(
    """
    <style>
    :root { --primary-color: #f0f0f0; }
    .stApp { background-color: #000000; color: #f0f0f0; font-family: 'Times New Roman', serif; }
    .stSidebar { background-color: #111111; color: #f0f0f0; font-family: 'Times New Roman', serif; }
    .stButton>button { background-color: #f0f0f0; color: #000000; border-radius: 8px; padding: 10px 20px; font-family: 'Times New Roman', serif; }
    .stButton>button:hover { background-color: #dddddd; }
    .stHeader { color: #f0f0f0; font-size: 32px; font-weight: bold; font-family: 'Times New Roman', serif; }
    .stExpander { background-color: #222222; color: #f0f0f0; font-family: 'Times New Roman', serif; }
    .stMultiSelect [data-testid=stMarkdownContainer] { color: #f0f0f0; font-family: 'Times New Roman', serif; }
    .stPlotlyChart { background-color: #000000; }
    .stDateInput label { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    .stTable { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    table { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    th, td { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    .stMetric, .stMetric label, .stMetricValue, [data-testid="stMetric"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    header { background-color: #000000 !important; }
    div[data-testid="stAlert"] { background-color: #111111 !important; color: #f0f0f0 !important; border-color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- DATA LOADING FUNCTIONS ---

@st.cache_data
def load_data_bundle():
    """
    Loads returns for Stocks and ETFs, and extracts the Risk Free Rate 
    from the Compustat file.
    Returns: (returns_df, rf_series)
    """
    try:
        # Load files
        comp = pd.read_parquet("compustat_git.parquet")
        etf = pd.read_parquet("etf_git.parquet")

        # 1. Extract Risk Free Rate (RF) from Compustat
        if "RF" in comp.columns:
            comp["date"] = pd.to_datetime(comp["date"])
            # Ensure RF is treated as numeric and sorted
            rf_raw = comp.groupby("date")["RF"].mean().sort_index()
            rf_series = rf_raw.fillna(0.0)
        else:
            rf_series = pd.Series(dtype=float)

        # 2. Process Stocks (Compustat)
        comp_ret = comp[["date", "company_name", "monthly_return"]].copy()
        comp_ret["date"] = pd.to_datetime(comp_ret["date"])
        comp_ret = comp_ret.rename(columns={"company_name": "asset", "monthly_return": "ret"})

        # 3. Process ETFs
        etf_ret = etf[["date", "ETF", "return_monthly"]].copy()
        etf_ret["date"] = pd.to_datetime(etf_ret["date"])
        etf_ret = etf_ret.rename(columns={"ETF": "asset", "return_monthly": "ret"})

        # 4. Merge and Pivot
        returns_long = pd.concat([comp_ret, etf_ret], ignore_index=True)
        returns_wide = returns_long.pivot(index="date", columns="asset", values="ret").sort_index()
        returns_wide.index = pd.to_datetime(returns_wide.index)

        return returns_wide, rf_series

    except Exception as e:
        st.error(f"Error loading data bundle: {e}")
        return pd.DataFrame(), pd.Series()

@st.cache_data
def load_country_mapping():
    try:
        comp = pd.read_parquet("compustat_git.parquet")
        if "country_code" in comp.columns:
            mapping = comp[["company_name", "country_code"]].drop_duplicates()
            mapping = mapping.set_index("company_name")["country_code"].to_dict()
            return mapping
        return {}
    except:
        return {}

def get_valid_assets(custom_data, start_date, end_date):
    # FIX 1: Snap dates to Month End
    start_date = pd.to_datetime(start_date) + MonthEnd(0)
    end_date = pd.to_datetime(end_date) + MonthEnd(0)
    
    if custom_data.empty: 
        return {"stocks": [], "etfs": []}

    subset = custom_data.loc[start_date:end_date]
    available_assets = subset.columns[subset.notna().any()].tolist()
    
    return {"stocks": available_assets, "etfs": []} 

def get_common_start_date(custom_data, selected_assets, user_start_date):
    # FIX 1: Snap dates to Month End
    user_start_date = pd.to_datetime(user_start_date) + MonthEnd(0)
    
    missing = [a for a in selected_assets if a not in custom_data.columns]
    if missing:
        st.error(f"Assets not in database: {missing}")
        return None
    
    first_valid = custom_data[selected_assets].apply(lambda col: col.first_valid_index())
    common_start = first_valid.max()
    
    if pd.isna(common_start):
        st.error("No common valid data found.")
        return None
        
    if common_start > user_start_date:
        st.warning(f"⚠️ Adjusted start date to **{common_start.date()}** (data availability).")
        
    return max(common_start, user_start_date)

def compute_rebalance_indices(dates, freq_label):
    freq_map = {"Quarterly": 3, "Semi-Annually": 6, "Annually": 12}
    step = freq_map.get(freq_label, 12)
    n = len(dates)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs

# --- OPTIMIZATION ENGINE ---

def solve_erc_weights(cov_matrix):
    n = cov_matrix.shape[0]
    
    def solve_with_rho(rho):
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix) - rho * cp.sum(cp.log(w)))
        constraints = [cp.sum(w) == 1, w >= 1e-6]
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL)
        except:
            prob.solve(solver=cp.SCS)
            
        if prob.status == "optimal":
            return np.array(w.value).flatten()
        return None

    def rc_variance(rho):
        w = solve_with_rho(rho)
        if w is None: return np.inf
        var = w @ cov_matrix @ w
        sigma = np.sqrt(var)
        if sigma <= 0: return np.inf
        mrc = cov_matrix @ w
        rc = w * mrc / sigma
        return np.var(rc)

    res = minimize_scalar(rc_variance, bounds=(1e-6, 1e-1), method="bounded", tol=1e-5)
    w_star = solve_with_rho(res.x)
    
    if w_star is None: raise RuntimeError("ERC Optimization Failed")
    w_star = np.where(np.abs(w_star) < 1e-6, 0, w_star)
    w_star /= w_star.sum()
    return w_star

def compute_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    return drawdowns.min() * 100

@st.cache_data(show_spinner=True)
def perform_optimization(selected_assets, start_date_user, end_date_user, rebalance_freq, _custom_data, _rf_data, lookback_months=36, ann_factor=12, tc_rate=0.001):
    """
    Main loop. 
    """
    custom_data = _custom_data 
    rf_data = _rf_data
    country_map = load_country_mapping()
    
    try:
        # FIX 1: Snap dates to Month End
        start_date_user = pd.to_datetime(start_date_user) + MonthEnd(0)
        end_date_user = pd.to_datetime(end_date_user) + MonthEnd(0)
        
        common_start = get_common_start_date(custom_data, selected_assets, start_date_user)
        if common_start is None: return None
        
        first_rebalance_date = common_start + pd.DateOffset(months=lookback_months)
        # Snap rebalance date to month end just in case
        first_rebalance_date = first_rebalance_date + MonthEnd(0)

        if first_rebalance_date > end_date_user:
            st.error(f"Not enough data for lookback. Need data until {first_rebalance_date.date()}")
            return None
            
        full_returns = custom_data[selected_assets].sort_index().loc[common_start:end_date_user]
        period_returns = full_returns.loc[first_rebalance_date:end_date_user]
        
        if period_returns.empty:
            st.error("Optimization period is empty.")
            return None
            
        period_dates = period_returns.index
        rebalance_indices = compute_rebalance_indices(period_dates, rebalance_freq)
        
        n = len(selected_assets)
        previous_weights = np.zeros(n)
        port_returns = pd.Series(index=period_dates, dtype=float).fillna(0.0)
        weights_over_time = {}
        country_exposure_over_time = {}
        total_tc = 0.0
        last_cov = None

        # --- REBALANCING LOOP ---
        for j, reb_idx in enumerate(rebalance_indices):
            rebal_date = period_dates[reb_idx]
            global_reb_pos = full_returns.index.get_loc(rebal_date)
            start_pos = max(0, global_reb_pos - lookback_months)
            
            # Estimate Covariance on Gross Returns
            est_window = full_returns.iloc[start_pos:global_reb_pos].dropna(how="any")
            
            if est_window.shape[0] < n + 1:
                st.warning(f"Insufficient data at {rebal_date.date()}, skipping rebalance.")
                weights = previous_weights
            else:
                lw = LedoitWolf().fit(est_window.values)
                cov = lw.covariance_ * ann_factor
                last_cov = cov
                try:
                    weights = solve_erc_weights(cov)
                except:
                    weights = previous_weights
            
            # Transaction Costs
            turnover = np.sum(np.abs(weights - previous_weights)) / 2
            cost = turnover * tc_rate
            total_tc += cost
            
            previous_weights = weights.copy()
            weights_over_time[rebal_date] = weights
            
            # Country Exposure
            country_exp = {}
            for asset, w in zip(selected_assets, weights):
                c = country_map.get(asset, "Unknown")
                country_exp[c] = country_exp.get(c, 0) + w
            country_exposure_over_time[rebal_date] = country_exp

            # Calculate Returns (Gross of RF, Net of Fees)
            if j == len(rebalance_indices) - 1:
                end_slice = len(period_dates)
            else:
                end_slice = rebalance_indices[j+1]
                
            sub_ret = period_returns.iloc[reb_idx:end_slice].fillna(0.0)
            if not sub_ret.empty:
                period_port_ret = sub_ret.values @ weights
                if len(period_port_ret) > 0:
                    period_port_ret[0] -= cost 
                port_returns.iloc[reb_idx:end_slice] = period_port_ret

        # --- CALCULATE METRICS ---
        
        # 1. Excess Returns
        if not rf_data.empty:
            aligned_rf = rf_data.reindex(port_returns.index, method='ffill').fillna(0.0)
            port_excess_returns = port_returns - aligned_rf
        else:
            st.warning("Risk Free Rate not found. Assuming 0%.")
            port_excess_returns = port_returns

        # 2. Volatility (Using GROSS returns)
        ann_vol = port_returns.std() * np.sqrt(ann_factor)
        
        # 3. Sharpe Ratio (Mean Excess / Gross Vol)
        ann_excess_ret = port_excess_returns.mean() * ann_factor
        sharpe = ann_excess_ret / ann_vol if ann_vol > 0 else 0.0
        
        # 4. Cumulative Return (Excess)
        cum_port_excess = (1 + port_excess_returns).cumprod()
        max_drawdown = compute_max_drawdown(cum_port_excess)

        if last_cov is not None:
            port_var = weights @ last_cov @ weights
            sigma_p = np.sqrt(port_var)
            mrc = last_cov @ weights
            rc_abs = weights * mrc / sigma_p
            rc_pct = (rc_abs / rc_abs.sum()) * 100
        else:
            rc_pct = np.zeros(n)

        return {
            "selected_assets": selected_assets,
            "weights": weights,
            "risk_contrib_pct": rc_pct,
            "expected_return": ann_excess_ret * 100, 
            "volatility": ann_vol * 100,             
            "sharpe": sharpe,
            "port_returns": port_excess_returns,
            "cum_port": cum_port_excess,
            "total_tc": total_tc * 100,
            "weights_df": pd.DataFrame(weights_over_time, index=selected_assets).T.sort_index(),
            "corr_matrix": est_window.corr(),
            "country_exposure_over_time": country_exposure_over_time,
            "max_drawdown": max_drawdown
        }

    except Exception as e:
        st.error(f"Optimization Error: {e}")
        return None

# --- VISUALIZATION ---

def plot_final_weights(results):
    df = pd.DataFrame({"Asset": results["selected_assets"], "Weight (%)": results["weights"] * 100})
    df = df.sort_values("Weight (%)", ascending=True)
    fig = px.bar(df, x="Weight (%)", y="Asset", orientation="h")
    fig.update_traces(marker_color="#0D6EFD", texttemplate="%{x:.2f}%")
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_risk_contributions(results):
    df = pd.DataFrame({"Asset": results["selected_assets"], "Risk Contribution (%)": results["risk_contrib_pct"]})
    df = df.sort_values("Risk Contribution (%)", ascending=True)
    fig = px.bar(df, x="Risk Contribution (%)", y="Asset", orientation="h")
    fig.update_traces(marker_color="#00C2FF", texttemplate="%{x:.2f}%")
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_cumulative_performance(results):
    cum_series = results["cum_port"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_series.index, y=cum_series.values, mode="lines", name="Portfolio", line=dict(color="#0D6EFD", width=3)))
    
    # Calculate dynamic dtick for cleaner log scale
    min_val = cum_series.min()
    max_val = cum_series.max()
    if min_val > 0 and max_val > 0:
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        log_range = log_max - log_min
        
        # Aim for approx 5-8 ticks
        raw_dtick = log_range / 6
        # Snap to a "nice" interval
        magnitude = 10 ** np.floor(np.log10(raw_dtick))
        normalized = raw_dtick / magnitude
        if normalized < 1.5: nice_dtick = 1.0 * magnitude
        elif normalized < 3.5: nice_dtick = 2.0 * magnitude
        elif normalized < 7.5: nice_dtick = 5.0 * magnitude
        else: nice_dtick = 10.0 * magnitude
    else:
        nice_dtick = 1 # Fallback
        
    fig.update_layout(
        title="Cumulative Excess Return (Wealth vs Cash) - Log Scale", 
        paper_bgcolor="#000", 
        plot_bgcolor="#000", 
        font=dict(color="#E0E0E0", family="Times New Roman"),
        yaxis_title="Growth of $1 (Log)",
        yaxis=dict(
            type="log",
            dtick=nice_dtick,
            tickformat=".2f",
            minor=dict(showgrid=False) # Hides messy minor lines
        )
    )
    return fig

def plot_weights_over_time(results):
    df = results["weights_df"]
    fig = px.area(df, x=df.index, y=df.columns)
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_correlation_matrix(results):
    fig = px.imshow(results["corr_matrix"], color_continuous_scale=["#0D6EFD", "#FFFFFF"], aspect="auto")
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_country_exposure_over_time(results):
    df = pd.DataFrame(results["country_exposure_over_time"]).T
    df.index = pd.to_datetime(df.index)
    fig = go.Figure()
    for country in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[country]*100, mode="lines", name=str(country)))
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"), yaxis_title="Exposure (%)")
    return fig

# --- MAIN APP ---

tab0, tab1, tab2, tab3 = st.tabs(["How to Use", "Asset Selection", "Portfolio Results", "About Us"])

with tab0:
    st.title("How to Use")
    st.markdown("""
    1. **Asset Selection**: Choose your date range and assets.
    2. **Optimization**: The app calculates the Equal Risk Contribution (ERC) portfolio.
    3. **Risk-Free Rate**: Automatically extracted from the dataset.
    4. **Results**: 
        - **Sharpe Ratio** = Avg Excess Return / Gross Volatility
        - **Performance** = Cumulative Excess Return (Log Scale)
    """)

with tab1:
    st.title("Asset Selection")
    custom_data, rf_data = load_data_bundle()
    
    if custom_data.empty:
        st.error("Data not loaded. Check 'compustat_git.parquet' and 'etf_git.parquet'.")
    else:
        min_date = custom_data.index.min().date()
        max_date = datetime(2024, 12, 31).date()
        
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = col2.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        if start_date < end_date:
            all_assets = custom_data.columns.tolist()
            # Snap dates for validation
            valid = get_valid_assets(custom_data, start_date, end_date)
            
            col1, col2 = st.columns(2)
            selected_assets = col1.multiselect("Select Assets", all_assets)
            
            rebalance_freq = st.selectbox("Rebalance Frequency", ["Quarterly", "Semi-Annually", "Annually"], index=2)
            
            if st.button("Optimize My Portfolio"):
                if not selected_assets:
                    st.error("Select at least one asset.")
                else:
                    with st.spinner("Optimizing..."):
                        results = perform_optimization(
                            selected_assets, 
                            start_date, 
                            end_date, 
                            rebalance_freq, 
                            custom_data, 
                            rf_data
                        )
                        if results:
                            st.session_state.results = results
                            st.success("Done! Go to Results tab.")
        else:
            st.error("End Date must be after Start Date.")

with tab2:
    st.title("Portfolio Results (Excess Return)")
    if "results" in st.session_state:
        res = st.session_state.results
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Ann. Excess Return", f"{res['expected_return']:.2f}%")
        col2.metric("Ann. Gross Volatility", f"{res['volatility']:.2f}%")
        col3.metric("Sharpe Ratio", f"{res['sharpe']:.2f}")
        col4.metric("Max Drawdown (Excess)", f"{res['max_drawdown']:.2f}%")
        col5.metric("Total Trans. Costs", f"{res['total_tc']:.2f}%")
        
        st.plotly_chart(plot_cumulative_performance(res), use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.subheader("Weights")
        c1.plotly_chart(plot_final_weights(res), use_container_width=True)
        c2.subheader("Risk Contributions")
        c2.plotly_chart(plot_risk_contributions(res), use_container_width=True)
        
        st.subheader("Weights Evolution")
        st.plotly_chart(plot_weights_over_time(res), use_container_width=True)
        
        st.subheader("Country Exposure")
        st.plotly_chart(plot_country_exposure_over_time(res), use_container_width=True)
    else:
        st.info("Run optimization first.")

with tab3:
    st.title("About Us")
    st.write("Pension Fund Optimizer Team.")
